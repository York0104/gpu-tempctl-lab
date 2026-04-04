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
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlDeviceGetUUID,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetPowerManagementLimit,
    nvmlDeviceGetPowerManagementLimitConstraints,
    nvmlDeviceGetPowerManagementDefaultLimit,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetFanSpeed,
    nvmlDeviceGetClockInfo,
    NVML_TEMPERATURE_GPU,
    NVML_CLOCK_GRAPHICS,
    NVML_CLOCK_MEM,
)

def parse_args():
    p = argparse.ArgumentParser(description="Open-loop PL sweep for GPU thermal experiment")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--pls", type=str, default="250,300,350,400,450",
                   help="Comma-separated PL list in watts")
    p.add_argument("--run-seconds", type=int, default=180)
    p.add_argument("--sample-interval", type=float, default=1.0)
    p.add_argument("--cooldown-target", type=float, default=45.0,
                   help="Wait until temp <= this value before next PL run")
    p.add_argument("--cooldown-timeout", type=int, default=900,
                   help="Max wait seconds for cooldown")
    p.add_argument("--last-window-seconds", type=int, default=60,
                   help="Summary window length from tail of run phase")
    p.add_argument("--workload-script", type=str, default="workload_torch.py")
    p.add_argument("--output-root", type=str, default="logs")
    return p.parse_args()

def build_nvidia_smi_cmd(*args):
    base = ["nvidia-smi", *args]
    if os.geteuid() == 0:
        return base
    return ["sudo", "-n", *base]

def set_power_limit(gpu_index: int, watts: int):
    cmd = build_nvidia_smi_cmd("-i", str(gpu_index), "-pl", str(watts))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"set_power_limit failed for {watts}W\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}\n"
            f"Hint: either run the whole script as root, or allow passwordless sudo for /usr/bin/nvidia-smi."
        )

def sample_metrics(handle):
    util = nvmlDeviceGetUtilizationRates(handle)
    return {
        "ts": time.time(),
        "temp_c": nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU),
        "power_w": nvmlDeviceGetPowerUsage(handle) / 1000.0,
        "power_limit_w": nvmlDeviceGetPowerManagementLimit(handle) / 1000.0,
        "gpu_util_percent": util.gpu,
        "mem_util_percent": util.memory,
        "fan_percent": nvmlDeviceGetFanSpeed(handle),
        "graphics_clock_mhz": nvmlDeviceGetClockInfo(handle, NVML_CLOCK_GRAPHICS),
        "mem_clock_mhz": nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM),
    }

def write_raw(writer, run_idx, phase, target_pl, m, note=""):
    writer.writerow([
        run_idx,
        phase,
        target_pl,
        m["ts"],
        m["temp_c"],
        m["power_w"],
        m["power_limit_w"],
        m["gpu_util_percent"],
        m["mem_util_percent"],
        m["fan_percent"],
        m["graphics_clock_mhz"],
        m["mem_clock_mhz"],
        note,
    ])

def avg(xs):
    return sum(xs) / len(xs) if xs else None

def summarize_run(samples, last_window_n):
    temps = [x["temp_c"] for x in samples]
    powers = [x["power_w"] for x in samples]
    gpu_utils = [x["gpu_util_percent"] for x in samples]
    mem_utils = [x["mem_util_percent"] for x in samples]

    tail = samples[-last_window_n:] if len(samples) >= last_window_n else samples
    tail_temps = [x["temp_c"] for x in tail]
    tail_powers = [x["power_w"] for x in tail]
    tail_gpu_utils = [x["gpu_util_percent"] for x in tail]

    return {
        "run_samples": len(samples),
        "temp_avg_all": avg(temps),
        "temp_max": max(temps) if temps else None,
        "temp_min": min(temps) if temps else None,
        "power_avg_all": avg(powers),
        "gpu_util_avg_all": avg(gpu_utils),
        "mem_util_avg_all": avg(mem_utils),
        "temp_avg_last_window": avg(tail_temps),
        "power_avg_last_window": avg(tail_powers),
        "gpu_util_avg_last_window": avg(tail_gpu_utils),
        "end_temp_c": temps[-1] if temps else None,
    }

def start_workload(workload_script: str, cwd: Path):
    cmd = [sys.executable, workload_script]
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc

def stop_workload(proc: subprocess.Popen, timeout=5):
    if proc.poll() is not None:
        return
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()

def main():
    args = parse_args()
    pls = [int(x.strip()) for x in args.pls.split(",") if x.strip()]

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(args.gpu_index)

    gpu_name = nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    gpu_uuid = nvmlDeviceGetUUID(handle)
    if isinstance(gpu_uuid, bytes):
        gpu_uuid = gpu_uuid.decode()

    min_pl_mw, max_pl_mw = nvmlDeviceGetPowerManagementLimitConstraints(handle)
    min_pl_w = int(min_pl_mw / 1000)
    max_pl_w = int(max_pl_mw / 1000)
    default_pl_w = int(nvmlDeviceGetPowerManagementDefaultLimit(handle) / 1000)

    for pl in pls:
        if not (min_pl_w <= pl <= max_pl_w):
            raise ValueError(f"PL {pl}W out of range [{min_pl_w}, {max_pl_w}]")

    ts_str = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.output_root) / f"pl_sweep_{ts_str}"
    outdir.mkdir(parents=True, exist_ok=True)

    raw_csv = outdir / "sweep_raw.csv"
    summary_csv = outdir / "sweep_summary.csv"
    metadata_json = outdir / "metadata.json"

    metadata = {
        "gpu_index": args.gpu_index,
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "pl_range_w": [min_pl_w, max_pl_w],
        "default_pl_w": default_pl_w,
        "pls": pls,
        "run_seconds": args.run_seconds,
        "sample_interval": args.sample_interval,
        "cooldown_target_c": args.cooldown_target,
        "cooldown_timeout_s": args.cooldown_timeout,
        "last_window_seconds": args.last_window_seconds,
        "workload_script": args.workload_script,
        "output_dir": str(outdir),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    last_window_n = max(1, int(args.last_window_seconds / args.sample_interval))

    raw_fp = open(raw_csv, "w", newline="")
    summary_fp = open(summary_csv, "w", newline="")
    raw_writer = csv.writer(raw_fp)
    summary_writer = csv.writer(summary_fp)

    raw_writer.writerow([
        "run_idx",
        "phase",
        "target_pl_w",
        "ts",
        "temp_c",
        "power_w",
        "power_limit_w",
        "gpu_util_percent",
        "mem_util_percent",
        "fan_percent",
        "graphics_clock_mhz",
        "mem_clock_mhz",
        "note",
    ])

    summary_writer.writerow([
        "run_idx",
        "target_pl_w",
        "cooldown_wait_s",
        "cooldown_end_temp_c",
        "run_samples",
        "temp_avg_all",
        "temp_max",
        "temp_min",
        "power_avg_all",
        "gpu_util_avg_all",
        "mem_util_avg_all",
        "temp_avg_last_window",
        "power_avg_last_window",
        "gpu_util_avg_last_window",
        "end_temp_c",
    ])

    current_proc = None
    try:
        print(f"Output dir: {outdir}")
        print(f"GPU: {gpu_name} ({gpu_uuid})")
        print(f"PL range: {min_pl_w}W ~ {max_pl_w}W, default={default_pl_w}W")

        for run_idx, pl in enumerate(pls, start=1):
            print(f"\n=== Run {run_idx}/{len(pls)} | target PL = {pl}W ===")

            # 1) Cooldown
            cooldown_start = time.time()
            cooldown_status = "ok"
            while True:
                m = sample_metrics(handle)
                write_raw(raw_writer, run_idx, "cooldown", pl, m)
                raw_fp.flush()

                if m["temp_c"] <= args.cooldown_target:
                    break
                if time.time() - cooldown_start >= args.cooldown_timeout:
                    cooldown_status = "timeout"
                    break
                time.sleep(args.sample_interval)

            cooldown_wait_s = time.time() - cooldown_start
            cooldown_end_temp_c = m["temp_c"]
            print(f"Cooldown done: status={cooldown_status}, wait={cooldown_wait_s:.1f}s, temp={cooldown_end_temp_c:.1f}C")

            # 2) Set PL
            set_power_limit(args.gpu_index, pl)
            m = sample_metrics(handle)
            write_raw(raw_writer, run_idx, "set_pl", pl, m, note="after_set_pl")
            raw_fp.flush()
            print(f"PL set to {pl}W")

            # 3) Start workload
            current_proc = start_workload(args.workload_script, Path.cwd())
            print("Workload started.")

            run_samples = []
            run_start = time.time()
            while time.time() - run_start < args.run_seconds:
                m = sample_metrics(handle)
                run_samples.append(m)
                write_raw(raw_writer, run_idx, "run", pl, m)
                raw_fp.flush()
                time.sleep(args.sample_interval)

            # 4) Stop workload
            stop_workload(current_proc)
            current_proc = None
            print("Workload stopped.")

            # 5) Summary
            s = summarize_run(run_samples, last_window_n)
            summary_writer.writerow([
                run_idx,
                pl,
                round(cooldown_wait_s, 3),
                cooldown_end_temp_c,
                s["run_samples"],
                s["temp_avg_all"],
                s["temp_max"],
                s["temp_min"],
                s["power_avg_all"],
                s["gpu_util_avg_all"],
                s["mem_util_avg_all"],
                s["temp_avg_last_window"],
                s["power_avg_last_window"],
                s["gpu_util_avg_last_window"],
                s["end_temp_c"],
            ])
            summary_fp.flush()

            print(
                f"Summary | temp_avg_last_window={s['temp_avg_last_window']:.2f}C, "
                f"power_avg_last_window={s['power_avg_last_window']:.2f}W, "
                f"end_temp={s['end_temp_c']:.1f}C"
            )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if current_proc is not None:
            stop_workload(current_proc)

        try:
            set_power_limit(args.gpu_index, default_pl_w)
            print(f"Restored default PL = {default_pl_w}W")
        except Exception as e:
            print(f"Warning: failed to restore default PL automatically: {e}")

        raw_fp.close()
        summary_fp.close()
        nvmlShutdown()

if __name__ == "__main__":
    main()
