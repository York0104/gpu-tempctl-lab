import argparse
import ctypes
import time


_nvml = ctypes.CDLL("libnvidia-ml.so.1")


def _check(ret: int, func_name: str) -> None:
    if ret != 0:
        _nvml.nvmlErrorString.restype = ctypes.c_char_p
        msg = _nvml.nvmlErrorString(ret)
        raise RuntimeError(f"{func_name} failed: ret={ret}, msg={msg.decode() if msg else 'unknown'}")


def nvml_init() -> None:
    _check(_nvml.nvmlInit_v2(), "nvmlInit_v2")


def nvml_shutdown() -> None:
    _check(_nvml.nvmlShutdown(), "nvmlShutdown")


def get_handle(gpu_index: int):
    handle = ctypes.c_void_p()
    _check(_nvml.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(gpu_index), ctypes.byref(handle)), "nvmlDeviceGetHandleByIndex_v2")
    return handle


def set_manual_fan_percent(gpu_index: int, fan_index: int, percent: int) -> None:
    if not (0 <= percent <= 100):
        raise ValueError("percent must be in [0, 100]")
    handle = get_handle(gpu_index)
    _check(
        _nvml.nvmlDeviceSetFanSpeed_v2(handle, ctypes.c_uint(fan_index), ctypes.c_uint(percent)),
        "nvmlDeviceSetFanSpeed_v2",
    )


def restore_default_fan(gpu_index: int, fan_index: int) -> None:
    handle = get_handle(gpu_index)
    _check(
        _nvml.nvmlDeviceSetDefaultFanSpeed_v2(handle, ctypes.c_uint(fan_index)),
        "nvmlDeviceSetDefaultFanSpeed_v2",
    )


def parse_args():
    p = argparse.ArgumentParser(description="Set GPU fan to manual percentage, then optionally restore default policy")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--fan-index", type=int, default=0)
    p.add_argument("--percent", type=int, help="Manual fan speed percent [0,100]")
    p.add_argument("--duration", type=int, default=0, help="If >0, hold this many seconds then restore default fan policy")
    p.add_argument("--restore-default", action="store_true", help="Restore default fan policy and exit")
    return p.parse_args()


def main():
    args = parse_args()
    nvml_init()
    try:
        if args.restore_default:
            restore_default_fan(args.gpu_index, args.fan_index)
            print("Restored default fan policy")
            return

        if args.percent is None:
            raise ValueError("--percent is required unless --restore-default is used")

        set_manual_fan_percent(args.gpu_index, args.fan_index, args.percent)
        print(f"Set manual fan to {args.percent}%")

        if args.duration > 0:
            time.sleep(args.duration)
            restore_default_fan(args.gpu_index, args.fan_index)
            print("Restored default fan policy")
    finally:
        nvml_shutdown()


if __name__ == "__main__":
    main()
