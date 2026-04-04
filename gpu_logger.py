import argparse
import csv
import time
from pathlib import Path

from pynvml import (
    NVML_CLOCK_GRAPHICS,
    NVML_CLOCK_MEM,
    NVML_TEMPERATURE_GPU,
    nvmlDeviceGetClockInfo,
    nvmlDeviceGetFanSpeed,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetPowerManagementLimit,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)


def parse_args():
    p = argparse.ArgumentParser(description="Simple NVML telemetry logger")
    p.add_argument("--output", type=str, default="gpu_log.csv")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--sample-interval", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(args.gpu_index)

    with out.open("w", newline="") as f:
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
            "mem_clock_mhz",
        ])

        print(f"Logging to {out} ... Press Ctrl+C to stop.")

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
                    mclk,
                ])
                f.flush()
                time.sleep(args.sample_interval)

        except KeyboardInterrupt:
            print("\nLogger stopped.")
        finally:
            nvmlShutdown()


if __name__ == "__main__":
    main()
