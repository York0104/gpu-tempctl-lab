import os
import csv
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
import requests

CC_BASE = os.getenv("CC_BASE", "http://localhost:11987")
CC_TOKEN = os.getenv("CC_TOKEN", "")
GPU_TARGET = float(os.getenv("GPU_TARGET", "80"))
BAND = float(os.getenv("BAND", "5"))
SECONDS = int(os.getenv("SECONDS", "300"))
TAG = os.getenv("TAG", "gpu_run")

def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def read_gpu_metrics():
    cmd = (
        "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw,fan.speed,clocks.gr,clocks.mem "
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

def state(temp, target, band):
    if temp is None:
        return "no_data"
    if temp > target + band:
        return "above_band"
    if temp < target - band:
        return "below_band"
    return "within_band"

def cc_get(path):
    if not CC_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {CC_TOKEN}"}
    url = f"{CC_BASE.rstrip('/')}/{path.lstrip('/')}"
    r = requests.get(url, headers=headers, timeout=5)
    r.raise_for_status()
    return r.json()

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path("fan_control_lab/logs") / f"{ts}_{TAG}"
    outdir.mkdir(parents=True, exist_ok=True)

    meta = {
        "started_at": ts,
        "gpu_target": GPU_TARGET,
        "band": BAND,
        "seconds": SECONDS,
        "cc_base": CC_BASE,
        "has_token": bool(CC_TOKEN),
        "tag": TAG,
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with (outdir / "thermal.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp",
            "gpu_temp_c",
            "gpu_state",
            "gpu_util_pct",
            "gpu_power_w",
            "gpu_fan_pct",
            "gpu_clock_mhz",
            "gpu_mem_clock_mhz",
        ])
        writer.writeheader()

        for _ in range(SECONDS):
            now = datetime.now().isoformat(timespec="seconds")
            g = read_gpu_metrics()
            row = {
                "timestamp": now,
                "gpu_temp_c": g["gpu_temp_c"],
                "gpu_state": state(g["gpu_temp_c"], GPU_TARGET, BAND),
                "gpu_util_pct": g["gpu_util_pct"],
                "gpu_power_w": g["gpu_power_w"],
                "gpu_fan_pct": g["gpu_fan_pct"],
                "gpu_clock_mhz": g["gpu_clock_mhz"],
                "gpu_mem_clock_mhz": g["gpu_mem_clock_mhz"],
            }
            writer.writerow(row)
            f.flush()
            print(row)

            if CC_TOKEN:
                try:
                    devices = cc_get("/devices")
                    with (outdir / "cc_devices.jsonl").open("a", encoding="utf-8") as rf:
                        rf.write(json.dumps({"timestamp": now, "devices": devices}) + "\n")
                except Exception as e:
                    with (outdir / "cc_devices.jsonl").open("a", encoding="utf-8") as rf:
                        rf.write(json.dumps({"timestamp": now, "error": str(e)}) + "\n")

            time.sleep(1)

if __name__ == "__main__":
    main()