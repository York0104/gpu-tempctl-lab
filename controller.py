import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from collections import deque
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

# ----------------------------
# Config
# ----------------------------
PROFILE_60 = {
    "tset": 60,
    "band_low": 55,
    "band_high": 65,
    "stable_low": 58,
    "stable_high": 62,
    "pl_init": 150,
    "workload_mode": "mid",   # 固定中等強度
}

PROFILE_80 = {
    "tset": 80,
    "band_low": 75,
    "band_high": 85,
    "stable_low": 78,
    "stable_high": 82,
    "pl_init": 300,
    "workload_mode": "mid",   # 先用 mid，必要時切 high
}

WORKLOAD_MODES = {
    "low": 4096,
    "mid": 6144,
    "high": 8192,
}

# ----------------------------
# Helpers
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GTX 1080 Ti thermal controller v1")
    p.add_argument("--target", type=int, choices=[60, 80], required=True, help="Target temperature mode")
    p.add_argument("--hold-seconds", type=int, default=600, help="Required consecutive in-band seconds")
    p.add_argument("--max-run-seconds", type=int, default=1800, help="Fail-safe max runtime")
    p.add_argument("--sample-interval", type=float, default=1.0, help="Telemetry sampling interval")
    p.add_argument("--decision-window", type=int, default=3, help="Decision every N samples")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--workload-script", type=str, default="workload_torch.py")
    p.add_argument("--output", type=str, default="controller_log.csv")
    return p.parse_args()

def run_cmd(cmd):
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    return res


def _signal_to_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt()

def set_power_limit(gpu_index, watts):
    if os.geteuid() == 0:
        cmd = ["nvidia-smi", "-i", str(gpu_index), "-pl", str(int(watts))]
    else:
        cmd = ["sudo", "-n", "nvidia-smi", "-i", str(gpu_index), "-pl", str(int(watts))]
    run_cmd(cmd)

def start_workload(workload_script, size, cwd):
    cmd = [sys.executable, workload_script, "--size", str(size)]
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc

def stop_workload(proc, timeout=5):
    if proc is None:
        return
    if proc.poll() is not None:
        return
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()

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

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def avg(xs):
    return sum(xs) / len(xs) if xs else None

# ----------------------------
# Control Logic
# ----------------------------
def decide_60(temp_avg, current_pl, current_mode, min_pl, max_pl):
    """
    60C 模式 v2：
    - 主控制量：PL
    - 當 PL 已到底仍偏熱時，切 low mode
    """
    action = "hold"
    new_pl = current_pl
    new_mode = current_mode

    if temp_avg < 55:
        new_pl = clamp(current_pl + 10, min_pl, max_pl)
        new_mode = "mid"
        action = "PL+10"

    elif 55 <= temp_avg < 58:
        new_pl = clamp(current_pl + 5, min_pl, max_pl)
        new_mode = "mid"
        action = "PL+5"

    elif 58 <= temp_avg <= 62:
        action = "HOLD_STABLE"

    elif 62 < temp_avg <= 65:
        if current_pl > min_pl:
            new_pl = clamp(current_pl - 5, min_pl, max_pl)
            action = "PL-5"
        else:
            if current_mode != "low":
                new_mode = "low"
                action = "MODE_LOW"
            else:
                action = "HOT_AT_MIN_PL"

    else:  # temp_avg > 65
        if current_pl > min_pl:
            new_pl = clamp(current_pl - 10, min_pl, max_pl)
            action = "PL-10"
        else:
            if current_mode != "low":
                new_mode = "low"
                action = "MODE_LOW"
            else:
                action = "HOT_AT_MIN_PL"

    # 若已在 low mode 且偏冷，恢復 mid
    if current_mode == "low" and temp_avg < 58:
        new_mode = "mid"
        action = "MODE_MID"

    return new_pl, new_mode, action

def decide_80(temp_avg, current_pl, current_mode, min_pl, max_pl, low_count):
    """
    非對稱式 band control
    - 過熱：靠 PL 往下壓
    - 偏冷：優先維持 300W，持續偏冷再切 high workload
    """
    action = "hold"
    new_pl = current_pl
    new_mode = current_mode
    new_low_count = low_count

    if temp_avg < 75:
        new_low_count += 1
        action = "LOW_WAIT"

        # 已經 300W 又持續偏冷，才切高 workload
        if current_pl >= max_pl and current_mode != "high" and new_low_count >= 2:
            new_mode = "high"
            action = "MODE_HIGH"
        elif current_pl < max_pl:
            # 理論上 1080 Ti 這裡可能沒空間，但還是保留通用邏輯
            new_pl = clamp(current_pl + 5, min_pl, max_pl)
            action = "PL+5"

    elif 75 <= temp_avg < 78:
        new_low_count += 1
        action = "APPROACHING_LOW"

        if current_pl >= max_pl and current_mode != "high" and new_low_count >= 2:
            new_mode = "high"
            action = "MODE_HIGH"
        elif current_pl < max_pl:
            new_pl = clamp(current_pl + 5, min_pl, max_pl)
            action = "PL+5"

    elif 78 <= temp_avg <= 82:
        new_low_count = 0
        action = "HOLD_STABLE"

    elif 82 < temp_avg <= 85:
        new_low_count = 0
        new_pl = clamp(current_pl - 5, min_pl, max_pl)
        action = "PL-5"

        # 若還在 high mode 且偏熱，可先降回 mid
        if current_mode == "high":
            new_mode = "mid"
            action = "MODE_MID_AND_PL-5"

    else:  # > 85
        new_low_count = 0
        new_pl = clamp(current_pl - 10, min_pl, max_pl)
        action = "PL-10"

        if current_mode == "high":
            new_mode = "mid"
            action = "MODE_MID_AND_PL-10"

    return new_pl, new_mode, action, new_low_count

# ----------------------------
# Main
# ----------------------------
def main():
    orig_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _signal_to_keyboard_interrupt)

    args = parse_args()

    profile = PROFILE_60 if args.target == 60 else PROFILE_80

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(args.gpu_index)

    gpu_name = nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()

    gpu_uuid = nvmlDeviceGetUUID(handle)
    if isinstance(gpu_uuid, bytes):
        gpu_uuid = gpu_uuid.decode()

    min_pl_mw, max_pl_mw = nvmlDeviceGetPowerManagementLimitConstraints(handle)
    min_pl = int(min_pl_mw / 1000)
    max_pl = int(max_pl_mw / 1000)
    default_pl = int(nvmlDeviceGetPowerManagementDefaultLimit(handle) / 1000)

    pl = clamp(profile["pl_init"], min_pl, max_pl)
    mode = profile["workload_mode"]
    size = WORKLOAD_MODES[mode]

    temp_hist = deque(maxlen=args.decision_window)
    in_band_consecutive = 0
    low_count = 0

    workload_proc = None
    csv_fp = None

    try:
        print(f"GPU: {gpu_name} ({gpu_uuid})")
        print(f"PL range: {min_pl}~{max_pl}W, default={default_pl}W")
        print(f"Target mode: {args.target}C")
        print(f"Initial PL: {pl}W, workload mode: {mode}, size={size}")

        set_power_limit(args.gpu_index, pl)
        workload_proc = start_workload(args.workload_script, size, Path.cwd())

        csv_fp = open(args.output, "w", newline="")
        writer = csv.writer(csv_fp)
        writer.writerow([
            "ts",
            "elapsed_s",
            "target_c",
            "phase",
            "temp_c",
            "temp_avg_3s",
            "power_w",
            "power_limit_w",
            "gpu_util_percent",
            "mem_util_percent",
            "fan_percent",
            "graphics_clock_mhz",
            "mem_clock_mhz",
            "controller_pl_w",
            "controller_mode",
            "decision_action",
            "in_band_consecutive_s",
        ])

        start = time.time()
        last_decision_action = "INIT"

        while True:
            now = time.time()
            elapsed = now - start
            if elapsed >= args.max_run_seconds:
                print("Reached max_run_seconds. Stop.")
                break

            m = sample_metrics(handle)
            temp_hist.append(m["temp_c"])
            temp_avg = avg(temp_hist)

            band_low = profile["band_low"]
            band_high = profile["band_high"]

            if band_low <= m["temp_c"] <= band_high:
                in_band_consecutive += 1
                phase = "hold"
            else:
                if in_band_consecutive == 0:
                    phase = "warmup"
                else:
                    phase = "recover"
                in_band_consecutive = 0

            # 每 decision_window 筆做一次控制決策
            if len(temp_hist) == args.decision_window:
                if args.target == 60:
                    new_pl, new_mode, action = decide_60(
                        temp_avg=temp_avg,
                        current_pl=pl,
                        current_mode=mode,
                        min_pl=min_pl,
                        max_pl=max_pl,
                    )
                    low_count = 0
                else:
                    new_pl, new_mode, action, low_count = decide_80(
                        temp_avg=temp_avg,
                        current_pl=pl,
                        current_mode=mode,
                        min_pl=min_pl,
                        max_pl=max_pl,
                        low_count=low_count,
                    )

                # PL changes
                if new_pl != pl:
                    set_power_limit(args.gpu_index, new_pl)
                    pl = new_pl

                # Workload mode changes
                if new_mode != mode:
                    stop_workload(workload_proc)
                    mode = new_mode
                    size = WORKLOAD_MODES[mode]
                    workload_proc = start_workload(args.workload_script, size, Path.cwd())

                last_decision_action = action

            writer.writerow([
                m["ts"],
                round(elapsed, 3),
                profile["tset"],
                phase,
                m["temp_c"],
                round(temp_avg, 3) if temp_avg is not None else None,
                m["power_w"],
                m["power_limit_w"],
                m["gpu_util_percent"],
                m["mem_util_percent"],
                m["fan_percent"],
                m["graphics_clock_mhz"],
                m["mem_clock_mhz"],
                pl,
                mode,
                last_decision_action,
                in_band_consecutive,
            ])
            csv_fp.flush()

            if in_band_consecutive >= args.hold_seconds:
                print(f"Success: stayed within [{band_low}, {band_high}] for {args.hold_seconds} consecutive seconds.")
                break

            time.sleep(args.sample_interval)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        stop_workload(workload_proc)

        try:
            set_power_limit(args.gpu_index, default_pl)
            print(f"Restored default PL = {default_pl}W")
        except Exception as e:
            print(f"Warning: failed to restore default PL automatically: {e}")

        if csv_fp is not None:
            csv_fp.close()
        nvmlShutdown()
        signal.signal(signal.SIGTERM, orig_sigterm)

if __name__ == "__main__":
    main()
