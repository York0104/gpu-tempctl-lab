import os
import csv
import json
import time
import signal
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from statistics import mean

import requests
import matplotlib.pyplot as plt


''''  
cd ~/gpu-tempctl-lab
source ../gpu-tempctl-1080ti/bin/activate
source fan_control_lab/env.sh
source "$HOME/.cargo/env"
export CCTV_DAEMON_PASSWORD='nctuiiot'
export MPLCONFIGDIR="$HOME/.config/matplotlib"
mkdir -p "$MPLCONFIGDIR"

python fan_control_lab/gpu_scenario_runner_auto.py \
  --tag exp_gpu_fault25_auto \
  --gpu-target 80 \
  --band 5 \
  --baseline-seconds 180 \
  --fault-seconds 300 \
  --recovery-seconds 180 \
  --baseline-mode-name GPU_BASELINE \
  --fault-mode-name GPU_FAULT_25 \
  --recovery-mode-name GPU_BASELINE
'''

def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def read_gpu_metrics():
    cmd = (
        "nvidia-smi --query-gpu="
        "temperature.gpu,utilization.gpu,power.draw,fan.speed,clocks.gr,clocks.mem "
        "--format=csv,noheader,nounits"
    )
    rc, out, err = sh(cmd)
    if rc != 0 or not out:
        return {
            "gpu_temp_c": None,
            "gpu_util_pct": None,
            "gpu_power_w": None,
            "gpu_fan_pct": None,
            "gpu_clock_mhz": None,
            "gpu_mem_clock_mhz": None,
            "gpu_error": err or "nvidia-smi failed",
        }

    parts = [x.strip() for x in out.split(",")]

    def conv(x):
        return None if x in ("N/A", "[Not Supported]") else float(x)

    return {
        "gpu_temp_c": conv(parts[0]),
        "gpu_util_pct": conv(parts[1]),
        "gpu_power_w": conv(parts[2]),
        "gpu_fan_pct": conv(parts[3]),
        "gpu_clock_mhz": conv(parts[4]),
        "gpu_mem_clock_mhz": conv(parts[5]),
        "gpu_error": "",
    }


def cc_get_devices(cc_base: str, cc_token: str):
    headers = {"Authorization": f"Bearer {cc_token}"}
    r = requests.get(f"{cc_base.rstrip('/')}/devices", headers=headers, timeout=5)
    r.raise_for_status()
    return r.json()


def gpu_state(temp, target, band):
    if temp is None:
        return "no_data"
    if temp > target + band:
        return "above_band"
    if temp < target - band:
        return "below_band"
    return "within_band"


def activate_mode(mode_name: str):
    rc, out, err = sh(f'cctv activate-mode "{mode_name}"')
    if rc != 0:
        raise RuntimeError(f"Failed to activate mode {mode_name}\nstdout={out}\nstderr={err}")


def wait_for_expected_fan(phase_name, timeout=20):
    start = time.time()
    while time.time() - start < timeout:
        g = read_gpu_metrics()
        fan = g["gpu_fan_pct"]
        print(f"[check] phase={phase_name} fan={fan} temp={g['gpu_temp_c']}")
        if phase_name == "fault":
            if fan is not None and abs(fan - 25) <= 2:
                return True
        else:
            if fan is not None and abs(fan - 25) > 3:
                return True
        time.sleep(1)
    return False


def start_workload(workdir: Path, seconds: int, size: int, duty: float, period_ms: int):
    stdout_f = open(workdir / "workload_stdout.log", "w", encoding="utf-8")
    stderr_f = open(workdir / "workload_stderr.log", "w", encoding="utf-8")
    cmd = [
        "python",
        "fan_control_lab/gpu_load_torch.py",
        "--seconds",
        str(seconds),
        "--size",
        str(size),
        "--duty",
        str(duty),
        "--period-ms",
        str(period_ms),
    ]
    p = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f)
    return p, stdout_f, stderr_f


def stop_process(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def summarize_phase(rows, phase_name):
    phase_rows = [r for r in rows if r["phase"] == phase_name and r["gpu_temp_c"] is not None]
    if not phase_rows:
        return {"samples": 0}

    temps = [r["gpu_temp_c"] for r in phase_rows]
    fans = [r["gpu_fan_pct"] for r in phase_rows if r["gpu_fan_pct"] is not None]
    powers = [r["gpu_power_w"] for r in phase_rows if r["gpu_power_w"] is not None]
    utils = [r["gpu_util_pct"] for r in phase_rows if r["gpu_util_pct"] is not None]

    tail = phase_rows[-min(30, len(phase_rows)) :]
    tail_temps = [r["gpu_temp_c"] for r in tail if r["gpu_temp_c"] is not None]

    return {
        "samples": len(phase_rows),
        "temp_min_c": min(temps),
        "temp_max_c": max(temps),
        "temp_mean_c": round(mean(temps), 3),
        "temp_last30_mean_c": round(mean(tail_temps), 3) if tail_temps else None,
        "fan_mean_pct": round(mean(fans), 3) if fans else None,
        "power_mean_w": round(mean(powers), 3) if powers else None,
        "util_mean_pct": round(mean(utils), 3) if utils else None,
        "within_band_ratio": round(
            sum(1 for r in phase_rows if r["gpu_state"] == "within_band") / len(phase_rows), 4
        ),
    }


def make_plot(rows, out_png: Path, target: float, band: float):
    ts = [r["elapsed_s"] for r in rows]
    temp = [r["gpu_temp_c"] for r in rows]
    fan = [r["gpu_fan_pct"] for r in rows]
    power = [r["gpu_power_w"] for r in rows]
    util = [r["gpu_util_pct"] for r in rows]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(ts, temp, label="GPU Temp")
    axes[0].axhline(target, linestyle="--", label="Target")
    axes[0].axhline(target + band, linestyle=":", label="Band +")
    axes[0].axhline(target - band, linestyle=":", label="Band -")
    axes[0].set_ylabel("Temp (C)")
    axes[0].legend()

    axes[1].plot(ts, fan, label="Fan %")
    axes[1].set_ylabel("Fan (%)")
    axes[1].legend()

    axes[2].plot(ts, power, label="Power (W)")
    axes[2].set_ylabel("Power (W)")
    axes[2].legend()

    axes[3].plot(ts, util, label="Util (%)")
    axes[3].set_ylabel("Util (%)")
    axes[3].set_xlabel("Elapsed Seconds")
    axes[3].legend()

    phase_boundaries = []
    last_phase = None
    for r in rows:
        if r["phase"] != last_phase:
            phase_boundaries.append((r["elapsed_s"], r["phase"]))
            last_phase = r["phase"]

    for ax in axes:
        for x, name in phase_boundaries:
            ax.axvline(x, linestyle=":", alpha=0.5)
            ax.text(x, ax.get_ylim()[1], name, fontsize=8, va="top")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gpu_scenario_manual")
    ap.add_argument("--gpu-target", type=float, default=float(os.getenv("GPU_TARGET", "80")))
    ap.add_argument("--band", type=float, default=float(os.getenv("BAND", "5")))
    ap.add_argument("--baseline-seconds", type=int, default=120)
    ap.add_argument("--fault-seconds", type=int, default=180)
    ap.add_argument("--recovery-seconds", type=int, default=120)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--duty", type=float, default=1.0)
    ap.add_argument("--period-ms", type=int, default=100)
    ap.add_argument("--sample-interval", type=float, default=1.0)
    ap.add_argument("--cc-base", default=os.getenv("CC_BASE", "http://localhost:11987"))
    ap.add_argument("--cc-token", default=os.getenv("CC_TOKEN", ""))
    ap.add_argument("--baseline-mode-name", default="GPU_BASELINE")
    ap.add_argument("--fault-mode-name", default="GPU_FAULT_25")
    ap.add_argument("--recovery-mode-name", default="GPU_BASELINE")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("fan_control_lab/logs") / f"{ts}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "tag": args.tag,
        "started_at": ts,
        "gpu_target": args.gpu_target,
        "band": args.band,
        "baseline_seconds": args.baseline_seconds,
        "fault_seconds": args.fault_seconds,
        "recovery_seconds": args.recovery_seconds,
        "size": args.size,
        "duty": args.duty,
        "period_ms": args.period_ms,
        "sample_interval": args.sample_interval,
        "cc_base": args.cc_base,
        "has_cc_token": bool(args.cc_token),
        "baseline_mode_name": args.baseline_mode_name,
        "fault_mode_name": args.fault_mode_name,
        "recovery_mode_name": args.recovery_mode_name,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    total_seconds = args.baseline_seconds + args.fault_seconds + args.recovery_seconds
    workload_proc, workload_stdout, workload_stderr = start_workload(
        run_dir, total_seconds + 180, args.size, args.duty, args.period_ms
    )

    rows = []
    events = []

    def log_event(elapsed_s, phase, event, detail=""):
        evt = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": round(elapsed_s, 3),
            "phase": phase,
            "event": event,
            "detail": detail,
        }
        events.append(evt)
        print(evt)

    phases = [
        ("baseline", args.baseline_seconds, args.baseline_mode_name),
        ("fault", args.fault_seconds, args.fault_mode_name),
        ("recovery", args.recovery_seconds, args.recovery_mode_name),
    ]

    start_time = time.time()

    try:
        for phase_name, phase_duration, mode_name in phases:
            print("\n" + "=" * 80)
            print(f"PHASE: {phase_name}")
            print(f"Activating CoolerControl mode: {mode_name}")
            activate_mode(mode_name)
            ok = wait_for_expected_fan(phase_name, timeout=20)
            log_event(time.time() - start_time, phase_name, "activate_mode", f"{mode_name}, ok={ok}")
            if not ok:
                raise RuntimeError(f"{phase_name} failed to reach expected fan state after mode switch")

            phase_start = time.time()
            while True:
                now = time.time()
                elapsed = now - start_time
                phase_elapsed = now - phase_start
                if phase_elapsed > phase_duration:
                    break

                g = read_gpu_metrics()
                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "elapsed_s": round(elapsed, 3),
                    "phase": phase_name,
                    "gpu_temp_c": g["gpu_temp_c"],
                    "gpu_state": gpu_state(g["gpu_temp_c"], args.gpu_target, args.band),
                    "gpu_util_pct": g["gpu_util_pct"],
                    "gpu_power_w": g["gpu_power_w"],
                    "gpu_fan_pct": g["gpu_fan_pct"],
                    "gpu_clock_mhz": g["gpu_clock_mhz"],
                    "gpu_mem_clock_mhz": g["gpu_mem_clock_mhz"],
                }
                rows.append(row)
                print(row)

                if args.cc_token:
                    try:
                        devices = cc_get_devices(args.cc_base, args.cc_token)
                        with (run_dir / "cc_devices.jsonl").open("a", encoding="utf-8") as rf:
                            rf.write(json.dumps({
                                "timestamp": row["timestamp"],
                                "phase": phase_name,
                                "devices": devices
                            }) + "\n")
                    except Exception as e:
                        with (run_dir / "cc_devices.jsonl").open("a", encoding="utf-8") as rf:
                            rf.write(json.dumps({
                                "timestamp": row["timestamp"],
                                "phase": phase_name,
                                "error": str(e)
                            }) + "\n")

                time.sleep(args.sample_interval)

    except KeyboardInterrupt:
        log_event(time.time() - start_time, "interrupted", "keyboard_interrupt", "")
    finally:
        stop_process(workload_proc)
        workload_stdout.close()
        workload_stderr.close()

    with (run_dir / "thermal.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "elapsed_s", "phase",
            "gpu_temp_c", "gpu_state", "gpu_util_pct",
            "gpu_power_w", "gpu_fan_pct",
            "gpu_clock_mhz", "gpu_mem_clock_mhz"
        ])
        writer.writeheader()
        writer.writerows(rows)

    with (run_dir / "events.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "elapsed_s", "phase", "event", "detail"])
        writer.writeheader()
        writer.writerows(events)

    summary = {
        "baseline": summarize_phase(rows, "baseline"),
        "fault": summarize_phase(rows, "fault"),
        "recovery": summarize_phase(rows, "recovery"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if rows:
        make_plot(rows, run_dir / "scenario_plot.png", args.gpu_target, args.band)

    print(f"\nRun complete: {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
