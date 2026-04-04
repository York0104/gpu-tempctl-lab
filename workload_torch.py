import argparse
import signal
import sys
import time
from pathlib import Path

import torch

from perf_logger import PerfLogger

stop = False

WORKLOAD_MODES = {
    "low": 4096,
    "mid": 6144,
    "high": 8192,
}


def handle_stop(signum, frame):
    global stop
    stop = True


def parse_args():
    p = argparse.ArgumentParser(description="Torch GEMM workload generator with optional perf logging")
    p.add_argument("--size", type=int, default=None, help="Matrix size. If omitted, derive from --mode.")
    p.add_argument("--mode", choices=["low", "mid", "high"], default="mid", help="Named workload mode")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--report-every", type=int, default=50, help="Print every N iterations")
    p.add_argument("--perf-csv", type=str, default=None, help="Optional performance CSV path")
    p.add_argument("--perf-interval", type=float, default=1.0, help="Performance log interval in seconds")
    p.add_argument("--scenario", type=str, default="standalone")
    p.add_argument("--fault-flag-file", type=str, default=None, help="Optional file path; fault_active=1 when file contains 1")
    return p.parse_args()


def read_fault_active(flag_file: str | None) -> bool:
    if not flag_file:
        return False
    p = Path(flag_file)
    if not p.exists():
        return False
    try:
        return p.read_text().strip() == "1"
    except Exception:
        return False


def main():
    global stop
    args = parse_args()

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    size = args.size if args.size is not None else WORKLOAD_MODES[args.mode]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This workload requires an NVIDIA GPU.")

    print("Using device:", torch.cuda.get_device_name(0), flush=True)
    print("Workload mode:", args.mode, flush=True)
    print("Matrix size:", size, flush=True)

    a = torch.randn(size, size, device=args.device)
    b = torch.randn(size, size, device=args.device)

    perf_logger = PerfLogger(args.perf_csv) if args.perf_csv else None

    iters = 0
    start = time.time()
    last_perf_log = start

    try:
        while not stop:
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            iters += 1

            if iters % args.report_every == 0:
                elapsed = time.time() - start
                print(f"iters={iters}, elapsed={elapsed:.1f}s", flush=True)

            now = time.time()
            if perf_logger is not None and now - last_perf_log >= args.perf_interval:
                perf_logger.log(
                    scenario=args.scenario,
                    fault_active=read_fault_active(args.fault_flag_file),
                    workload_mode=args.mode,
                    matrix_size=size,
                    iters_total=iters,
                )
                last_perf_log = now

    finally:
        elapsed = time.time() - start
        if perf_logger is not None:
            perf_logger.log(
                scenario=args.scenario,
                fault_active=read_fault_active(args.fault_flag_file),
                workload_mode=args.mode,
                matrix_size=size,
                iters_total=iters,
            )
            perf_logger.close()
        print(f"Stopped. iters={iters}, elapsed={elapsed:.1f}s", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
