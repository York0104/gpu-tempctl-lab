import os
import time
import subprocess

GPU_TARGET = float(os.getenv("GPU_TARGET", "80"))
BAND = float(os.getenv("BAND", "5"))

def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def gpu_temp():
    rc, out, _ = sh("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits")
    if rc != 0 or not out:
        return None
    try:
        return float(out.splitlines()[0].strip())
    except:
        return None

def judge(temp, target):
    if temp is None:
        return "no_data"
    if temp > target + BAND:
        return "above_band"
    if temp < target - BAND:
        return "below_band"
    return "within_band"

while True:
    gt = gpu_temp()
    print({"gpu_temp_c": gt, "gpu_state": judge(gt, GPU_TARGET)})
    time.sleep(1)