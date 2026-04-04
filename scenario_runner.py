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


def run_cmd(cmd):
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"STDOUT:\n{res.stdout}\n"
            f"STDERR:\n{res.stderr}"
        )
    return res

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
    p.add_argument("--power-limit", type=int, default=200)
    p.add_argument("--run-seconds", type=int, default=900)
    p.add_argument("--fault-start", type=int, default=300)
    p.add_argument("--fault-seconds", type=int, default=300)
    p.add_argument("--fault-fan-percent", type=int, default=50)
    p.add_argument("--fan-index", type=int, default=0)
    p.add_argument("--sample-interval", type=float, default=1.0)
    p.add_argument("--safety-temp", type=float, default=83.0)
    return p.parse_args()


def run_natural_high_heat(args, run_dir: Path):
    out_csv = run_dir / "controller.csv"
    stdout_log = run_dir / "controller_stdout.log"

    if os.geteuid() == 0:
        cmd = [
            sys.executable, args.controller_script,
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

                if rc != 0:
                    log_tail = tail_text_file(stdout_log, 80)
                    raise RuntimeError(
                        f"natural_high_heat failed with code {rc}\n"
                        f"log: {stdout_log}\n"
                        f"---- controller_stdout.log (tail) ----\n"
                        f"{log_tail}"
                    )
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


def run_cooling_anomaly(args, run_dir: Path):
    thermal_csv = run_dir / "thermal.csv"
    perf_csv = run_dir / "perf.csv"
    events_csv = run_dir / "events.csv"
    fault_flag_file = run_dir / "fault_active.flag"
    fault_flag_file.write_text("0")

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

            workload_cmd = [
                sys.executable,
                args.workload_script,
                "--mode", args.workload_mode,
                "--perf-csv", str(perf_csv),
                "--scenario", args.scenario,
                "--fault-flag-file", str(fault_flag_file),
            ]
            workload_proc, workload_stdout = start_proc(
                workload_cmd,
                cwd=Path.cwd(),
                stdout_path=run_dir / "workload_stdout.log",
            )
            write_event(writer, start_ts, "workload_start", args.workload_mode)
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
                        write_event(writer, start_ts, "fault_auto_restored_by_safety", temp)
                    write_event(writer, start_ts, "safety_temp_hit", temp)
                    break

                if (not fault_active) and elapsed >= args.fault_start:
                    set_manual_fan_percent(args.gpu_index, args.fan_index, args.fault_fan_percent)
                    fault_flag_file.write_text("1")
                    fault_active = True
                    write_event(writer, start_ts, "fault_start", args.fault_fan_percent)
                    f.flush()

                if fault_active and elapsed >= fault_end:
                    restore_default_fan(args.gpu_index, args.fan_index)
                    fault_flag_file.write_text("0")
                    fault_active = False
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


def ensure_sudo_for_runner():
    res = subprocess.run(["sudo", "-v"])
    if res.returncode != 0:
        raise RuntimeError("sudo credential initialization failed")


def main():
    ensure_sudo_for_runner()
    args = parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.output_root)
    if args.session_name:
        base_dir = base_dir / args.session_name
    run_dir = base_dir / f"{args.scenario}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "metadata.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False)
    )

    print(f"Run directory: {run_dir}", flush=True)

    if args.scenario == "natural_high_heat":
        run_natural_high_heat(args, run_dir)
    elif args.scenario == "cooling_anomaly":
        run_cooling_anomaly(args, run_dir)
    else:
        raise ValueError(f"unsupported scenario: {args.scenario}")


if __name__ == "__main__":
    main()
