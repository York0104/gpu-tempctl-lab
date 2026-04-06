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

CCTV_BIN = os.path.expanduser("~/.cargo/bin/cctv")


'''
cd ~/gpu-tempctl-lab
source ../gpu-tempctl-1080ti/bin/activate
source fan_control_lab/env.sh
source "$HOME/.cargo/env"

export CCTV_DAEMON_PASSWORD='nctuiiot'
export MPLCONFIGDIR="$HOME/.config/matplotlib"
mkdir -p "$MPLCONFIGDIR"

rm -f ~/.local/bin/cctv
hash -r

which -a cctv
~/.cargo/bin/cctv --help | head
'''

'''
cd ~/gpu-tempctl-lab
source ../gpu-tempctl-1080ti/bin/activate
source fan_control_lab/env.sh
source "$HOME/.cargo/env"

export CCTV_DAEMON_PASSWORD='nctuiiot'
export MPLCONFIGDIR="$HOME/.config/matplotlib"
mkdir -p "$MPLCONFIGDIR"

python fan_control_lab/gpu_supervisor_80.py \
  --tag sup70_run600 \
  --target 70 \
  --band 3 \
  --crit-temp 95 \
  --seconds 300 \
  --min-dwell-seconds 15 \
  --size 4096 \
  --duty 1.0 \
  --period-ms 100 \
  --initial-mode GPU_FAULT_5

'''


'''
cat fan_control_lab/logs/*sup80_demo_short*/summary.json
head -n 30 fan_control_lab/logs/*sup80_demo_short*/events.csv
tail -n 20 fan_control_lab/logs/*sup80_demo_short*/thermal.csv
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


def activate_mode(mode_name: str):
    rc, out, err = sh(f'"{CCTV_BIN}" activate-mode "{mode_name}"')
    if rc != 0:
        raise RuntimeError(f"Failed to activate mode {mode_name}\nstdout={out}\nstderr={err}")


def wait_for_mode_effect(mode_name: str, timeout: int = 20):
    expected_fixed = {
        "GPU_FAULT_5": 5,
        "GPU_FAULT_15": 15,
        "GPU_FAULT_20": 20,
        "GPU_FAULT_25": 25,
        "GPU_Cool_Max": 100,
    }

    start = time.time()
    while time.time() - start < timeout:
        g = read_gpu_metrics()
        fan = g["gpu_fan_pct"]
        if fan is None:
            time.sleep(1)
            continue

        if mode_name in expected_fixed:
            target = expected_fixed[mode_name]
            if abs(fan - target) <= 3:
                return True
        elif mode_name in ("GPU_BASELINE", "GPU_DEFAULT"):
            if all(abs(fan - x) > 3 for x in (5, 15, 20, 25)):
                return True

        time.sleep(1)

    return False


def get_current_mode_name():
    rc, out, err = sh(f'"{CCTV_BIN}" dump')
    if rc != 0:
        return None

    try:
        data = json.loads(out)
    except Exception:
        return None

    current_uid = None
    if isinstance(data, dict):
        current_uid = data.get("state", {}).get("current_mode_uid")

    if not current_uid:
        return None

    for mode in data.get("modes", []):
        if mode.get("uid") == current_uid:
            return mode.get("name")

    return None


def gpu_state(temp, target, band):
    if temp is None:
        return "no_data"
    if temp > target + band:
        return "above_band"
    if temp < target - band:
        return "below_band"
    return "within_band"


def choose_mode(temp, current_mode, target, band, crit_temp):
    if temp is None:
        return current_mode, "no_data_hold"

    lower = target - band
    upper = target + band

    if temp >= crit_temp:
        return "GPU_Cool_Max", "crit_temp"

    if temp < target - 30:
        return "GPU_FAULT_5", "preheat"
    elif temp < target - 15:
        return "GPU_FAULT_15", "too_cold"
    elif temp < lower:
        return "GPU_FAULT_20", "warming"
    elif temp < target - 1:
        return "GPU_FAULT_25", "fine_warming"
    elif temp <= upper:
        if current_mode == "GPU_Cool_Max":
            return "GPU_BASELINE", "cool_release"

        if current_mode == "GPU_BASELINE" and temp <= target:
            return "GPU_FAULT_25", "reenter_hold"

        return current_mode, "hold_zone"
    elif temp <= upper + 1:
        return "GPU_BASELINE", "cooling"
    else:
        return "GPU_Cool_Max", "strong_cooling"


def start_workload(workdir: Path, seconds: int, size: int, duty: float, period_ms: int):
    stdout_f = open(workdir / "workload_stdout.log", "w", encoding="utf-8")
    stderr_f = open(workdir / "workload_stderr.log", "w", encoding="utf-8")
    cmd = [
        "python",
        "fan_control_lab/gpu_load_torch.py",
        "--seconds", str(seconds),
        "--size", str(size),
        "--duty", str(duty),
        "--period-ms", str(period_ms),
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


def summarize_rows(rows):
    valid = [r for r in rows if r["gpu_temp_c"] is not None]
    if not valid:
        return {"samples": 0}

    temps = [r["gpu_temp_c"] for r in valid]
    fans = [r["gpu_fan_pct"] for r in valid if r["gpu_fan_pct"] is not None]
    powers = [r["gpu_power_w"] for r in valid if r["gpu_power_w"] is not None]
    utils = [r["gpu_util_pct"] for r in valid if r["gpu_util_pct"] is not None]

    tail = valid[-min(30, len(valid)) :]
    tail_temps = [r["gpu_temp_c"] for r in tail if r["gpu_temp_c"] is not None]

    return {
        "samples": len(valid),
        "temp_min_c": min(temps),
        "temp_max_c": max(temps),
        "temp_mean_c": round(mean(temps), 3),
        "temp_last30_mean_c": round(mean(tail_temps), 3) if tail_temps else None,
        "fan_mean_pct": round(mean(fans), 3) if fans else None,
        "power_mean_w": round(mean(powers), 3) if powers else None,
        "util_mean_pct": round(mean(utils), 3) if utils else None,
        "within_band_ratio": round(
            sum(1 for r in valid if r["gpu_state"] == "within_band") / len(valid), 4
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

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gpu_supervisor_80")
    ap.add_argument("--target", type=float, default=80.0)
    ap.add_argument("--band", type=float, default=5.0)
    ap.add_argument("--crit-temp", type=float, default=95.0)
    ap.add_argument("--seconds", type=int, default=900)
    ap.add_argument("--sample-interval", type=float, default=1.0)
    ap.add_argument("--min-dwell-seconds", type=int, default=12)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--duty", type=float, default=1.0)
    ap.add_argument("--period-ms", type=int, default=100)
    ap.add_argument("--cc-base", default=os.getenv("CC_BASE", "http://localhost:11987"))
    ap.add_argument("--cc-token", default=os.getenv("CC_TOKEN", ""))
    ap.add_argument("--initial-mode", default="GPU_FAULT_15")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("fan_control_lab/logs") / f"{ts}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    original_mode = get_current_mode_name()

    metadata = {
        "tag": args.tag,
        "started_at": ts,
        "target": args.target,
        "band": args.band,
        "crit_temp": args.crit_temp,
        "seconds": args.seconds,
        "sample_interval": args.sample_interval,
        "min_dwell_seconds": args.min_dwell_seconds,
        "size": args.size,
        "duty": args.duty,
        "period_ms": args.period_ms,
        "initial_mode": args.initial_mode,
        "original_mode": original_mode,
        "restore_mode": "GPU_DEFAULT",
        "cc_base": args.cc_base,
        "has_cc_token": bool(args.cc_token),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    workload_proc, workload_stdout, workload_stderr = start_workload(
        run_dir, args.seconds + 180, args.size, args.duty, args.period_ms
    )

    rows = []
    events = []

    def log_event(elapsed_s, event, detail=""):
        evt = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": round(elapsed_s, 3),
            "event": event,
            "detail": detail,
        }
        events.append(evt)
        print(evt)

    current_mode = args.initial_mode
    start_time = time.time()

    try:
        activate_mode(current_mode)
        ok = wait_for_mode_effect(current_mode)
        log_event(0.0, "activate_mode", f"{current_mode}, ok={ok}")
        if not ok:
            raise RuntimeError(f"Initial mode did not take effect: {current_mode}")

        last_switch_ts = time.time()
        while True:
            now = time.time()
            elapsed = now - start_time
            if elapsed > args.seconds:
                break

            g = read_gpu_metrics()
            temp = g["gpu_temp_c"]

            desired_mode, reason = choose_mode(
                temp,
                current_mode,
                args.target,
                args.band,
                args.crit_temp,
            )
            since_switch = now - last_switch_ts

            infeasible = ""
            if current_mode == "GPU_Cool_Max" and temp is not None and temp > args.target + args.band:
                infeasible = "infeasible_high_load"
            elif current_mode == "GPU_FAULT_5" and temp is not None and temp < args.target - args.band:
                infeasible = "infeasible_low_load"

            if desired_mode != current_mode and since_switch >= args.min_dwell_seconds:
                activate_mode(desired_mode)
                ok = wait_for_mode_effect(desired_mode)
                log_event(elapsed, "activate_mode", f"{desired_mode}, reason={reason}, ok={ok}")
                if ok:
                    current_mode = desired_mode
                    last_switch_ts = time.time()
                else:
                    log_event(elapsed, "mode_switch_failed", desired_mode)

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "elapsed_s": round(elapsed, 3),
                "gpu_temp_c": g["gpu_temp_c"],
                "gpu_state": gpu_state(g["gpu_temp_c"], args.target, args.band),
                "gpu_util_pct": g["gpu_util_pct"],
                "gpu_power_w": g["gpu_power_w"],
                "gpu_fan_pct": g["gpu_fan_pct"],
                "gpu_clock_mhz": g["gpu_clock_mhz"],
                "gpu_mem_clock_mhz": g["gpu_mem_clock_mhz"],
                "current_mode": current_mode,
                "desired_mode": desired_mode,
                "reason": reason,
                "infeasible": infeasible,
            }
            rows.append(row)
            print(row)

            if args.cc_token:
                try:
                    devices = cc_get_devices(args.cc_base, args.cc_token)
                    with (run_dir / "cc_devices.jsonl").open("a", encoding="utf-8") as rf:
                        rf.write(json.dumps({
                            "timestamp": row["timestamp"],
                            "devices": devices
                        }) + "\n")
                except Exception as e:
                    with (run_dir / "cc_devices.jsonl").open("a", encoding="utf-8") as rf:
                        rf.write(json.dumps({
                            "timestamp": row["timestamp"],
                            "error": str(e)
                        }) + "\n")

            time.sleep(args.sample_interval)

    except KeyboardInterrupt:
        log_event(time.time() - start_time, "keyboard_interrupt", "")
    finally:
        stop_process(workload_proc)
        workload_stdout.close()
        workload_stderr.close()

        restore_target = "GPU_DEFAULT"
        try:
            activate_mode(restore_target)
            ok = wait_for_mode_effect(restore_target, timeout=20)
            log_event(time.time() - start_time, "restore_mode", f"{restore_target}, ok={ok}")
        except Exception as e:
            log_event(time.time() - start_time, "restore_mode_failed", str(e))

    with (run_dir / "thermal.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "elapsed_s",
            "gpu_temp_c", "gpu_state", "gpu_util_pct",
            "gpu_power_w", "gpu_fan_pct",
            "gpu_clock_mhz", "gpu_mem_clock_mhz",
            "current_mode", "desired_mode", "reason", "infeasible"
        ])
        writer.writeheader()
        writer.writerows(rows)

    with (run_dir / "events.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "elapsed_s", "event", "detail"])
        writer.writeheader()
        writer.writerows(events)

    summary = summarize_rows(rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if rows:
        make_plot(rows, run_dir / "scenario_plot.png", args.target, args.band)

    report = [
        f"target={args.target}",
        f"band=±{args.band}",
        f"samples={summary.get('samples')}",
        f"temp_last30_mean_c={summary.get('temp_last30_mean_c')}",
        f"temp_max_c={summary.get('temp_max_c')}",
        f"fan_mean_pct={summary.get('fan_mean_pct')}",
        f"within_band_ratio={summary.get('within_band_ratio')}",
    ]
    (run_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\nRun complete: {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
