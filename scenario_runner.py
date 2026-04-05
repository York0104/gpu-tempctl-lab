import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from pynvml import (
    NVML_TEMPERATURE_GPU,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetPowerManagementDefaultLimit,
    nvmlDeviceGetTemperature,
    nvmlInit,
    nvmlShutdown,
)

from fan_fault_injector import (
    restore_default_fan,
    set_manual_fan_percent,
    nvml_init as fan_nvml_init,
    nvml_shutdown as fan_nvml_shutdown,
)
from workload_profiles import get_workload_profile


def run_cmd(cmd):
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"STDOUT:\n{res.stdout}\n"
            f"STDERR:\n{res.stderr}"
        )
    return res


def _signal_to_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt()


def build_nvidia_smi_cmd(*args):
    base = ["nvidia-smi", *args]
    if os.geteuid() == 0:
        return base
    return ["sudo", "-n", *base]


def set_power_limit(gpu_index: int, watts: int):
    run_cmd(build_nvidia_smi_cmd("-i", str(gpu_index), "-pl", str(int(watts))))


def start_proc(cmd, cwd=None, stdout_path: Path | None = None):
    stdout_fp = stdout_path.open("w") if stdout_path else subprocess.DEVNULL
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=stdout_fp,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True,
    )
    return proc, stdout_fp


def stop_proc(proc, timeout=10):
    if proc is None or proc.poll() is not None:
        return
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


def write_event(writer, start_ts, event, value):
    writer.writerow([time.time(), round(time.time() - start_ts, 3), event, value])


def tail_text_file(path: Path, n_lines: int = 80) -> str:
    if not path.exists():
        return f"[log file not found] {path}"
    with path.open("r", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-n_lines:])


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", errors="replace") as f:
        return list(csv.DictReader(f))


def read_last_csv_row(path: Path):
    rows = read_csv_rows(path)
    return rows[-1] if rows else None


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _send_signal_to_pid(pid: int, sig: int) -> bool:
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return True
    except PermissionError:
        sig_flag = "-TERM" if sig == signal.SIGTERM else "-KILL"
        res = subprocess.run(
            ["sudo", "-n", "kill", sig_flag, str(pid)],
            capture_output=True,
            text=True,
        )
        return res.returncode == 0


def terminate_pid(pid: int, timeout_s: float = 5.0) -> bool:
    _send_signal_to_pid(pid, signal.SIGTERM)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.2)

    _send_signal_to_pid(pid, signal.SIGKILL)
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.2)
    return not _pid_exists(pid)


def find_workload_pids(workload_script: str) -> set[int]:
    token = Path(workload_script).name
    res = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return set()

    pids = set()
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        args = parts[1]
        if "python" not in args:
            continue
        if (
            f"/{token}" not in args
            and f" {token}" not in args
            and not args.endswith(token)
        ):
            continue
        pids.add(pid)
    return pids


def cleanup_orphan_workloads(workload_script: str, baseline_pids: set[int], meta: dict):
    current = find_workload_pids(workload_script)
    orphan_pids = sorted(current - baseline_pids)
    if not orphan_pids:
        return

    cleaned = []
    failed = []
    for pid in orphan_pids:
        if terminate_pid(pid):
            cleaned.append(pid)
        else:
            failed.append(pid)

    if cleaned:
        meta["result"]["orphan_workload_pids_cleaned"] = cleaned
        meta["result"]["notes"].append(f"cleaned orphan workload pids: {cleaned}")
    if failed:
        meta["result"]["orphan_workload_pids_failed"] = failed
        meta["result"]["notes"].append(f"failed to clean orphan workload pids: {failed}")


def restore_default_fan_policy(args, meta: dict):
    script = str(Path(__file__).with_name("fan_fault_injector.py"))
    cmd = [
        sys.executable,
        script,
        "--gpu-index", str(args.gpu_index),
        "--fan-index", str(args.fan_index),
        "--restore-default",
    ]
    if os.geteuid() != 0:
        cmd = ["sudo", "-n", "-E", *cmd]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        meta["result"]["notes"].append(
            "fan restore failed: "
            f"code={res.returncode}, stderr={res.stderr.strip() or 'N/A'}"
        )
        return False
    return True


def restore_default_power_limit(args, meta: dict):
    default_pl = meta.get("gpu", {}).get("default_pl_w")
    if default_pl is None:
        try:
            default_pl = get_gpu_info(args.gpu_index)["default_pl_w"]
        except Exception as e:
            meta["result"]["notes"].append(f"cannot query default PL for cleanup: {e}")
            return False

    try:
        set_power_limit(args.gpu_index, int(default_pl))
        return True
    except Exception as e:
        meta["result"]["notes"].append(f"default PL restore failed: {e}")
        return False


def get_gpu_info(gpu_index: int):
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetName,
        nvmlDeviceGetUUID,
        nvmlDeviceGetPowerManagementLimitConstraints,
        nvmlDeviceGetPowerManagementDefaultLimit,
    )

    nvmlInit()
    try:
        handle = nvmlDeviceGetHandleByIndex(gpu_index)

        name = nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()

        uuid = nvmlDeviceGetUUID(handle)
        if isinstance(uuid, bytes):
            uuid = uuid.decode()

        min_pl_mw, max_pl_mw = nvmlDeviceGetPowerManagementLimitConstraints(handle)
        default_pl_w = int(nvmlDeviceGetPowerManagementDefaultLimit(handle) / 1000)

        return {
            "gpu_index": gpu_index,
            "name": name,
            "uuid": uuid,
            "pl_min_w": int(min_pl_mw / 1000),
            "pl_max_w": int(max_pl_mw / 1000),
            "default_pl_w": default_pl_w,
        }
    finally:
        nvmlShutdown()


def build_metadata_base(args, run_dir: Path):
    import socket

    gpu_info = get_gpu_info(args.gpu_index)

    meta = {
        "schema_version": "scenario_runner.v2",
        "scenario": args.scenario,
        "run_id": run_dir.name,
        "timestamps": {
            "start_ts": time.time(),
            "end_ts": None,
            "duration_s": None,
        },
        "host": {
            "hostname": socket.gethostname(),
            "cwd": str(Path.cwd()),
            "python_executable": sys.executable,
            "runner_euid": os.geteuid(),
        },
        "gpu": gpu_info,
        "requested": {},
        "effective": {},
        "artifacts": {},
        "result": {
            "status": "running",
            "return_code": None,
            "success_condition": None,
            "notes": [],
        },
    }

    if args.scenario == "natural_high_heat":
        meta["requested"] = {
            "scenario": args.scenario,
            "target": args.target,
            "hold_seconds": args.hold_seconds,
            "max_run_seconds": args.max_run_seconds,
        }
    elif args.scenario == "cooling_anomaly":
        meta["requested"] = {
            "scenario": args.scenario,
            "workload_mode": args.workload_mode,
            "workload_profile": args.workload_profile,
            "workload_size": args.workload_size,
            "workload_run_style": args.workload_run_style,
            "workload_period_ms": args.workload_period_ms,
            "workload_compute_budget_ms": args.workload_compute_budget_ms,
            "workload_warmup_seconds": args.workload_warmup_seconds,
            "workload_cycle_report_every": args.workload_cycle_report_every,
            "power_limit": args.power_limit,
            "run_seconds": args.run_seconds,
            "fault_start": args.fault_start,
            "fault_seconds": args.fault_seconds,
            "fault_fan_percent": args.fault_fan_percent,
            "fan_index": args.fan_index,
            "sample_interval": args.sample_interval,
            "safety_temp": args.safety_temp,
        }

    return meta


def write_metadata(run_dir: Path, meta: dict):
    (run_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def generate_temperature_plot(run_dir: Path, meta: dict):
    controller_csv = run_dir / "controller.csv"
    thermal_csv = run_dir / "thermal.csv"

    controller_rows = read_csv_rows(controller_csv)
    thermal_rows = read_csv_rows(thermal_csv)
    if not controller_rows and not thermal_rows:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        meta["result"]["notes"].append(
            "temperature plot skipped: matplotlib is not available"
        )
        return

    traces = []

    if controller_rows:
        xs = []
        ys = []
        first_ts = None
        for row in controller_rows:
            y = _to_float(row.get("temp_c"))
            if y is None:
                continue
            x = _to_float(row.get("elapsed_s"))
            if x is None:
                ts = _to_float(row.get("ts"))
                if ts is None:
                    continue
                if first_ts is None:
                    first_ts = ts
                x = ts - first_ts
            xs.append(x)
            ys.append(y)
        if xs and ys:
            traces.append(("controller temp", xs, ys))

    if thermal_rows:
        xs = []
        ys = []
        first_ts = None
        for idx, row in enumerate(thermal_rows):
            y = _to_float(
                row.get("temp_c")
                or row.get("temperature_c")
                or row.get("gpu_temp_c")
            )
            if y is None:
                continue
            ts = _to_float(row.get("ts"))
            if ts is None:
                x = float(idx)
            else:
                if first_ts is None:
                    first_ts = ts
                x = ts - first_ts
            xs.append(x)
            ys.append(y)
        if xs and ys:
            traces.append(("thermal temp", xs, ys))

    if not traces:
        return

    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        for label, xs, ys in traces:
            ax.plot(xs, ys, label=label, linewidth=2)

        controller_effective = meta.get("effective", {}).get("controller", {})
        target = _to_float(controller_effective.get("target"))
        band_low = _to_float(controller_effective.get("band_low"))
        band_high = _to_float(controller_effective.get("band_high"))
        if target is not None:
            ax.axhline(target, linestyle="--", linewidth=1, label=f"target {target:.0f}C")
        if band_low is not None and band_high is not None:
            ax.axhspan(band_low, band_high, alpha=0.12, label="target band")

        ax.set_xlabel("Elapsed (s)")
        ax.set_ylabel("Temperature (C)")
        ax.set_title(f"{meta.get('scenario', 'scenario')} | {meta.get('result', {}).get('status', 'unknown')}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        out_path = run_dir / "temperature_curve.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        meta["artifacts"]["temperature_plot"] = str(out_path)
    except Exception as e:
        meta["result"]["notes"].append(f"temperature plot generation failed: {e}")


def print_run_summary(run_dir: Path):
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        print(f"[summary] metadata.json not found: {meta_path}", flush=True)
        return

    meta = json.loads(meta_path.read_text())
    scenario = meta.get("scenario", "unknown")
    status = meta.get("result", {}).get("status", "unknown")

    print("\n===== Scenario Result Summary =====", flush=True)
    print(f"run_id   : {meta.get('run_id', run_dir.name)}", flush=True)
    print(f"scenario : {scenario}", flush=True)
    print(f"status   : {status}", flush=True)

    if meta.get("timestamps"):
        print(f"duration : {meta['timestamps'].get('duration_s')}", flush=True)

    # natural_high_heat: controller.csv
    controller_csv = run_dir / "controller.csv"
    last_ctl = read_last_csv_row(controller_csv)
    if last_ctl is not None:
        print(f"final_temp_c           : {last_ctl.get('temp_c')}", flush=True)
        print(f"final_temp_avg_3s      : {last_ctl.get('temp_avg_3s')}", flush=True)
        print(f"final_controller_pl_w  : {last_ctl.get('controller_pl_w')}", flush=True)
        print(f"final_controller_mode  : {last_ctl.get('controller_mode')}", flush=True)
        print(f"final_in_band_seconds  : {last_ctl.get('in_band_consecutive_s')}", flush=True)

    # cooling_anomaly: thermal.csv
    thermal_csv = run_dir / "thermal.csv"
    last_thermal = read_last_csv_row(thermal_csv)
    if last_thermal is not None:
        temp_val = (
            last_thermal.get("temp_c")
            or last_thermal.get("temperature_c")
            or last_thermal.get("gpu_temp_c")
        )
        pl_val = (
            last_thermal.get("power_limit_w")
            or last_thermal.get("pl_w")
        )
        fan_val = (
            last_thermal.get("fan_percent")
            or last_thermal.get("fan_speed_percent")
        )
        print(f"final_temp_c           : {temp_val}", flush=True)
        print(f"final_power_limit_w    : {pl_val}", flush=True)
        print(f"final_fan_percent      : {fan_val}", flush=True)

    temp_plot = meta.get("artifacts", {}).get("temperature_plot")
    if temp_plot:
        print(f"temperature_plot       : {temp_plot}", flush=True)

    print(f"run_dir  : {run_dir}", flush=True)
    print("==================================\n", flush=True)


def finalize_metadata(meta: dict, status: str, return_code: int | None = None, notes=None):
    meta["timestamps"]["end_ts"] = time.time()
    meta["timestamps"]["duration_s"] = round(
        meta["timestamps"]["end_ts"] - meta["timestamps"]["start_ts"], 3
    )
    meta["result"]["status"] = status
    meta["result"]["return_code"] = return_code
    if notes:
        meta["result"]["notes"].extend(notes)


def parse_args():
    p = argparse.ArgumentParser(description="Run predefined GPU thermal scenarios")
    p.add_argument("--scenario", choices=["natural_high_heat", "cooling_anomaly"], required=True)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--output-root", type=str, default="logs")
    p.add_argument("--session-name", type=str, default=None)

    # common scripts
    p.add_argument("--controller-script", type=str, default="controller.py")
    p.add_argument("--workload-script", type=str, default="workload_torch.py")
    p.add_argument("--logger-script", type=str, default="gpu_logger.py")

    # natural_high_heat
    p.add_argument("--target", type=int, default=80)
    p.add_argument("--hold-seconds", type=int, default=600)
    p.add_argument("--max-run-seconds", type=int, default=1800)

    # cooling_anomaly
    p.add_argument("--workload-mode", choices=["low", "mid", "high"], default="mid")
    p.add_argument(
        "--workload-profile",
        type=str,
        default=None,
        help="Named workload profile, e.g. timeslice30_openloop_icclz1",
    )
    p.add_argument("--workload-size", type=int, default=None)
    p.add_argument("--workload-run-style", choices=["continuous", "timeslice"], default=None)
    p.add_argument("--workload-period-ms", type=float, default=None)
    p.add_argument("--workload-compute-budget-ms", type=float, default=None)
    p.add_argument("--workload-warmup-seconds", type=float, default=None)
    p.add_argument("--workload-cycle-report-every", type=int, default=None)
    p.add_argument("--power-limit", type=int, default=200)
    p.add_argument("--run-seconds", type=int, default=900)
    p.add_argument("--fault-start", type=int, default=300)
    p.add_argument("--fault-seconds", type=int, default=300)
    p.add_argument("--fault-fan-percent", type=int, default=50)
    p.add_argument("--fan-index", type=int, default=0)
    p.add_argument("--sample-interval", type=float, default=1.0)
    p.add_argument("--safety-temp", type=float, default=83.0)
    return p.parse_args()


def resolve_workload_config(args) -> dict:
    if args.workload_profile:
        profile = get_workload_profile(args.workload_profile)
        profile_args = profile.get("args", {})
        workload_cfg = {
            "profile_name": args.workload_profile,
            "profile_type": profile.get("profile_type"),
            "validated_on": profile.get("validated_on"),
            "validation": profile.get("validation"),
            "mode": profile_args.get("mode"),
            "run_style": profile_args.get("run_style", "continuous"),
            "size": profile_args.get("size"),
            "period_ms": profile_args.get("period_ms"),
            "compute_budget_ms": profile_args.get("compute_budget_ms"),
            "warmup_seconds": profile_args.get("warmup_seconds"),
            "cycle_report_every": profile_args.get("cycle_report_every"),
        }
    else:
        workload_cfg = {
            "profile_name": None,
            "profile_type": "adhoc",
            "validated_on": None,
            "validation": None,
            "mode": args.workload_mode,
            "run_style": "continuous",
            "size": None,
            "period_ms": None,
            "compute_budget_ms": None,
            "warmup_seconds": None,
            "cycle_report_every": None,
        }

    cli_overrides = {
        "run_style": args.workload_run_style,
        "size": args.workload_size,
        "period_ms": args.workload_period_ms,
        "compute_budget_ms": args.workload_compute_budget_ms,
        "warmup_seconds": args.workload_warmup_seconds,
        "cycle_report_every": args.workload_cycle_report_every,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            workload_cfg[key] = value

    run_style = workload_cfg.get("run_style")
    if run_style not in {"continuous", "timeslice"}:
        raise ValueError(f"unsupported workload run_style: {run_style}")

    if run_style == "continuous":
        if not workload_cfg.get("mode"):
            workload_cfg["mode"] = args.workload_mode
        workload_cfg["size"] = None
        workload_cfg["period_ms"] = None
        workload_cfg["compute_budget_ms"] = None
        workload_cfg["warmup_seconds"] = None
        workload_cfg["cycle_report_every"] = None
    else:
        workload_cfg["mode"] = None
        required_fields = [
            "size",
            "period_ms",
            "compute_budget_ms",
            "warmup_seconds",
            "cycle_report_every",
        ]
        missing = [k for k in required_fields if workload_cfg.get(k) is None]
        if missing:
            raise ValueError(
                "timeslice workload config is missing required fields: "
                + ", ".join(missing)
            )
        if workload_cfg["period_ms"] <= 0:
            raise ValueError("timeslice workload period_ms must be > 0")
        if workload_cfg["compute_budget_ms"] < 0:
            raise ValueError("timeslice workload compute_budget_ms must be >= 0")
        if workload_cfg["warmup_seconds"] < 0:
            raise ValueError("timeslice workload warmup_seconds must be >= 0")
        if workload_cfg["cycle_report_every"] <= 0:
            raise ValueError("timeslice workload cycle_report_every must be > 0")

    return workload_cfg


def run_natural_high_heat(args, run_dir: Path, meta: dict):
    out_csv = run_dir / "controller.csv"
    stdout_log = run_dir / "controller_stdout.log"

    if args.target == 60:
        controller_effective = {
            "target": 60,
            "band_low": 55,
            "band_high": 65,
            "stable_low": 58,
            "stable_high": 62,
            "pl_init_w": 150,
            "workload_mode": "mid",
        }
    else:
        controller_effective = {
            "target": 80,
            "band_low": 75,
            "band_high": 85,
            "stable_low": 78,
            "stable_high": 82,
            "pl_init_w": 300,
            "workload_mode": "mid",
        }

    meta["effective"]["controller"] = controller_effective
    meta["artifacts"]["controller_csv"] = str(out_csv)
    meta["artifacts"]["controller_stdout_log"] = str(stdout_log)
    meta["artifacts"]["command_json"] = str(run_dir / "command.json")
    write_metadata(run_dir, meta)

    if os.geteuid() == 0:
        cmd = [
            sys.executable,
            args.controller_script,
            "--target", str(args.target),
            "--hold-seconds", str(args.hold_seconds),
            "--max-run-seconds", str(args.max_run_seconds),
            "--gpu-index", str(args.gpu_index),
            "--workload-script", args.workload_script,
            "--output", str(out_csv),
        ]
    else:
        cmd = [
            "sudo", "-n", "-E", sys.executable, args.controller_script,
            "--target", str(args.target),
            "--hold-seconds", str(args.hold_seconds),
            "--max-run-seconds", str(args.max_run_seconds),
            "--gpu-index", str(args.gpu_index),
            "--workload-script", args.workload_script,
            "--output", str(out_csv),
        ]

    with (run_dir / "command.json").open("w") as f:
        json.dump({"cmd": cmd}, f, indent=2, ensure_ascii=False)

    proc, stdout_fp = start_proc(
        cmd,
        cwd=Path.cwd(),
        stdout_path=stdout_log,
    )

    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                if stdout_fp not in (None, subprocess.DEVNULL):
                    stdout_fp.flush()
                    stdout_fp.close()
                    stdout_fp = None

                if rc != 0:
                    meta["result"]["return_code"] = rc
                    meta["result"]["notes"].append(f"controller log: {stdout_log}")
                    log_tail = tail_text_file(stdout_log, 80)
                    raise RuntimeError(
                        f"natural_high_heat failed with code {rc}\n"
                        f"log: {stdout_log}\n"
                        f"---- controller_stdout.log (tail) ----\n"
                        f"{log_tail}"
                    )

                meta["result"]["success_condition"] = "in-band-hold"
                meta["result"]["return_code"] = 0

                last_row = read_last_csv_row(out_csv)
                if last_row:
                    try:
                        meta["result"]["final_temp_c"] = float(last_row["temp_c"])
                    except (KeyError, TypeError, ValueError):
                        pass
                    try:
                        meta["result"]["final_controller_pl_w"] = float(last_row["controller_pl_w"])
                    except (KeyError, TypeError, ValueError):
                        pass
                    if "controller_mode" in last_row:
                        meta["result"]["final_controller_mode"] = last_row["controller_mode"]
                    try:
                        meta["result"]["final_in_band_consecutive_s"] = int(
                            float(last_row["in_band_consecutive_s"])
                        )
                    except (KeyError, TypeError, ValueError):
                        pass
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping controller...", flush=True)
        stop_proc(proc)
        raise
    finally:
        if stdout_fp not in (None, subprocess.DEVNULL):
            try:
                stdout_fp.close()
            except Exception:
                pass
        if proc.poll() is None:
            stop_proc(proc)


def run_cooling_anomaly(args, run_dir: Path, meta: dict):
    thermal_csv = run_dir / "thermal.csv"
    perf_csv = run_dir / "perf.csv"
    events_csv = run_dir / "events.csv"
    timeslice_cycles_csv = run_dir / "timeslice_cycles.csv"
    fault_flag_file = run_dir / "fault_active.flag"
    fault_flag_file.write_text("0")
    workload_cfg = resolve_workload_config(args)

    meta["effective"]["fault_plan"] = {
        "workload_profile": workload_cfg.get("profile_name"),
        "workload_mode": workload_cfg.get("mode"),
        "workload_run_style": workload_cfg.get("run_style"),
        "workload_size": workload_cfg.get("size"),
        "workload_period_ms": workload_cfg.get("period_ms"),
        "workload_compute_budget_ms": workload_cfg.get("compute_budget_ms"),
        "workload_warmup_seconds": workload_cfg.get("warmup_seconds"),
        "workload_cycle_report_every": workload_cfg.get("cycle_report_every"),
        "power_limit": args.power_limit,
        "run_seconds": args.run_seconds,
        "fault_start": args.fault_start,
        "fault_seconds": args.fault_seconds,
        "fault_fan_percent": args.fault_fan_percent,
        "fan_index": args.fan_index,
        "safety_temp": args.safety_temp,
    }
    meta["effective"]["workload"] = workload_cfg
    meta["artifacts"]["thermal_csv"] = str(thermal_csv)
    meta["artifacts"]["perf_csv"] = str(perf_csv)
    meta["artifacts"]["events_csv"] = str(events_csv)
    meta["artifacts"]["fault_flag_file"] = str(fault_flag_file)
    meta["artifacts"]["workload_stdout_log"] = str(run_dir / "workload_stdout.log")
    meta["artifacts"]["logger_stdout_log"] = str(run_dir / "logger_stdout.log")
    if workload_cfg["run_style"] == "timeslice":
        meta["artifacts"]["timeslice_cycles_csv"] = str(timeslice_cycles_csv)
    write_metadata(run_dir, meta)

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(args.gpu_index)
    default_pl = int(nvmlDeviceGetPowerManagementDefaultLimit(handle) / 1000)
    nvmlShutdown()

    workload_proc = None
    workload_stdout = None
    logger_proc = None
    logger_stdout = None

    fan_nvml_init()
    try:
        with events_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ts", "elapsed_s", "event", "value"])
            start_ts = time.time()

            set_power_limit(args.gpu_index, args.power_limit)
            write_event(writer, start_ts, "power_limit_set", args.power_limit)
            f.flush()

            workload_cmd = [sys.executable, args.workload_script]
            if workload_cfg["run_style"] == "continuous":
                workload_cmd.extend([
                    "--mode", workload_cfg["mode"],
                    "--run-style", "continuous",
                ])
            else:
                workload_cmd.extend([
                    "--size", str(workload_cfg["size"]),
                    "--run-style", "timeslice",
                    "--period-ms", str(workload_cfg["period_ms"]),
                    "--compute-budget-ms", str(workload_cfg["compute_budget_ms"]),
                    "--warmup-seconds", str(workload_cfg["warmup_seconds"]),
                    "--cycle-report-every", str(workload_cfg["cycle_report_every"]),
                    "--timeslice-cycles-csv", str(timeslice_cycles_csv),
                ])
            workload_cmd.extend([
                "--perf-csv", str(perf_csv),
                "--scenario", args.scenario,
                "--fault-flag-file", str(fault_flag_file),
            ])
            workload_proc, workload_stdout = start_proc(
                workload_cmd,
                cwd=Path.cwd(),
                stdout_path=run_dir / "workload_stdout.log",
            )
            write_event(
                writer,
                start_ts,
                "workload_start",
                workload_cfg.get("profile_name") or workload_cfg["run_style"],
            )
            f.flush()

            logger_cmd = [
                sys.executable,
                args.logger_script,
                "--output", str(thermal_csv),
                "--gpu-index", str(args.gpu_index),
                "--sample-interval", str(args.sample_interval),
            ]
            logger_proc, logger_stdout = start_proc(
                logger_cmd,
                cwd=Path.cwd(),
                stdout_path=run_dir / "logger_stdout.log",
            )
            write_event(writer, start_ts, "thermal_logger_start", str(thermal_csv))
            f.flush()

            fault_active = False
            fault_injected_once = False
            fault_restored_once = False
            fault_end = args.fault_start + args.fault_seconds

            while True:
                elapsed = time.time() - start_ts
                if elapsed >= args.run_seconds:
                    write_event(writer, start_ts, "scenario_complete", args.run_seconds)
                    break

                nvmlInit()
                handle = nvmlDeviceGetHandleByIndex(args.gpu_index)
                temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                nvmlShutdown()

                if temp >= args.safety_temp:
                    if fault_active:
                        restore_default_fan(args.gpu_index, args.fan_index)
                        fault_flag_file.write_text("0")
                        fault_active = False
                        fault_restored_once = True
                        write_event(writer, start_ts, "fault_auto_restored_by_safety", temp)
                    write_event(writer, start_ts, "safety_temp_hit", temp)
                    break

                if (not fault_injected_once) and elapsed >= args.fault_start:
                    set_manual_fan_percent(args.gpu_index, args.fan_index, args.fault_fan_percent)
                    fault_flag_file.write_text("1")
                    fault_active = True
                    fault_injected_once = True
                    write_event(writer, start_ts, "fault_start", args.fault_fan_percent)
                    f.flush()

                if fault_active and elapsed >= fault_end:
                    restore_default_fan(args.gpu_index, args.fan_index)
                    fault_flag_file.write_text("0")
                    fault_active = False
                    fault_restored_once = True
                    write_event(writer, start_ts, "fault_end_restore_auto", temp)
                    f.flush()

                f.flush()
                time.sleep(args.sample_interval)

    finally:
        try:
            restore_default_fan(args.gpu_index, args.fan_index)
        except Exception:
            pass

        try:
            fault_flag_file.write_text("0")
        except Exception:
            pass

        stop_proc(logger_proc)
        stop_proc(workload_proc)

        if logger_stdout not in (None, subprocess.DEVNULL):
            try:
                logger_stdout.close()
            except Exception:
                pass
        if workload_stdout not in (None, subprocess.DEVNULL):
            try:
                workload_stdout.close()
            except Exception:
                pass

        try:
            set_power_limit(args.gpu_index, default_pl)
        except Exception:
            pass

        fan_nvml_shutdown()

    event_rows = read_csv_rows(events_csv)
    event_names = {row.get("event") for row in event_rows}

    fault_started = "fault_start" in event_names
    fault_restored = (
        "fault_end_restore_auto" in event_names
        or "fault_auto_restored_by_safety" in event_names
    )
    safety_stop = "safety_temp_hit" in event_names

    meta["result"]["fault_started"] = fault_started
    meta["result"]["fault_restored"] = fault_restored
    meta["result"]["safety_stop"] = safety_stop
    meta["result"]["return_code"] = 0

    if safety_stop:
        meta["result"]["status"] = "safety_stop"
        meta["result"]["success_condition"] = "stopped_by_safety_temp"
    elif fault_started and fault_restored:
        meta["result"]["success_condition"] = "fault_injected_and_restored"
    elif fault_started and not fault_restored:
        meta["result"]["success_condition"] = "fault_injected_not_restored"
        meta["result"]["notes"].append("fault started but no restore event found")
    else:
        meta["result"]["success_condition"] = "scenario_completed_without_fault"
        meta["result"]["notes"].append("fault_start event not found")

    last_thermal = read_last_csv_row(thermal_csv)
    if last_thermal:
        try:
            meta["result"]["final_temp_c"] = float(last_thermal["temp_c"])
        except (KeyError, TypeError, ValueError):
            pass


def ensure_sudo_for_runner():
    if os.geteuid() == 0:
        return
    res = subprocess.run(["sudo", "-v"])
    if res.returncode != 0:
        raise RuntimeError("sudo credential initialization failed")


def main():
    orig_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _signal_to_keyboard_interrupt)
    try:
        ensure_sudo_for_runner()
        args = parse_args()
        baseline_workload_pids = find_workload_pids(args.workload_script)
        ts = time.strftime("%Y%m%d_%H%M%S")
        base_dir = Path(args.output_root)
        if args.session_name:
            base_dir = base_dir / args.session_name
        run_dir = base_dir / f"{args.scenario}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        meta = build_metadata_base(args, run_dir)
        write_metadata(run_dir, meta)

        print(f"Run directory: {run_dir}", flush=True)

        try:
            if args.scenario == "natural_high_heat":
                run_natural_high_heat(args, run_dir, meta)
            elif args.scenario == "cooling_anomaly":
                run_cooling_anomaly(args, run_dir, meta)
            else:
                raise ValueError(f"unsupported scenario: {args.scenario}")

            if meta["result"]["status"] == "running":
                finalize_metadata(meta, status="success", return_code=0)
            else:
                finalize_metadata(
                    meta,
                    status=meta["result"]["status"],
                    return_code=meta["result"].get("return_code"),
                )

        except KeyboardInterrupt:
            finalize_metadata(meta, status="aborted", return_code=None, notes=["Interrupted by user"])
            raise
        except Exception as e:
            finalize_metadata(meta, status="failed", return_code=1, notes=[str(e)])
            raise
        finally:
            cleanup_orphan_workloads(args.workload_script, baseline_workload_pids, meta)
            restore_default_power_limit(args, meta)
            restore_default_fan_policy(args, meta)
            write_metadata(run_dir, meta)
            generate_temperature_plot(run_dir, meta)
            write_metadata(run_dir, meta)
            print_run_summary(run_dir)
    finally:
        signal.signal(signal.SIGTERM, orig_sigterm)


if __name__ == "__main__":
    main()
