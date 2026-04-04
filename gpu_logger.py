import csv
import time
import sys
from pynvml import *

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

filename = sys.argv[1] if len(sys.argv) > 1 else "gpu_log.csv"

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "ts",
        "temp_c",
        "power_w",
        "power_limit_w",
        "gpu_util_percent",
        "mem_util_percent",
        "fan_percent",
        "graphics_clock_mhz",
        "mem_clock_mhz"
    ])

    print(f"Logging to {filename} ... Press Ctrl+C to stop.")

    try:
        while True:
            ts = time.time()
            temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
            power = nvmlDeviceGetPowerUsage(handle) / 1000.0
            power_limit = nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            util = nvmlDeviceGetUtilizationRates(handle)
            fan = nvmlDeviceGetFanSpeed(handle)
            gclk = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_GRAPHICS)
            mclk = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM)

            writer.writerow([
                ts,
                temp,
                power,
                power_limit,
                util.gpu,
                util.memory,
                fan,
                gclk,
                mclk
            ])
            f.flush()
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nLogger stopped.")
