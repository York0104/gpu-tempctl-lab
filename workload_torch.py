import argparse
import csv
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
    p.add_argument("--run-style", choices=["continuous", "timeslice"], default="continuous")
    p.add_argument("--period-ms", type=float, default=100.0)
    p.add_argument("--compute-budget-ms", type=float, default=30.0)
    p.add_argument("--warmup-seconds", type=float, default=2.0)
    p.add_argument("--cycle-report-every", type=int, default=10)
    p.add_argument("--report-every", type=int, default=50, help="Print every N iterations")
    p.add_argument("--perf-csv", type=str, default=None, help="Optional performance CSV path")
    p.add_argument(
        "--timeslice-cycles-csv",
        type=str,
        default=None,
        help="Optional timeslice cycle CSV path (auto: sibling of --perf-csv or ./timeslice_cycles.csv)",
    )
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


class TimesliceCycleLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.path.open("w", newline="")
        self.writer = csv.writer(self.fp)
        self.writer.writerow([
            "ts",
            "cycle_idx",
            "compute_actual_ms",
            "sleep_actual_ms",
            "period_actual_ms",
            "cycle_iters",
        ])

    def log(
        self,
        *,
        ts: float,
        cycle_idx: int,
        compute_actual_ms: float,
        sleep_actual_ms: float,
        period_actual_ms: float,
        cycle_iters: int,
    ) -> None:
        self.writer.writerow([
            ts,
            cycle_idx,
            compute_actual_ms,
            sleep_actual_ms,
            period_actual_ms,
            cycle_iters,
        ])
        self.fp.flush()

    def close(self) -> None:
        if not self.fp.closed:
            self.fp.close()


def resolve_timeslice_cycles_csv_path(args) -> Path:
    if args.timeslice_cycles_csv:
        return Path(args.timeslice_cycles_csv)
    if args.perf_csv:
        return Path(args.perf_csv).with_name("timeslice_cycles.csv")
    return Path("timeslice_cycles.csv")


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
    print("Run style:", args.run_style, flush=True)

    a = torch.randn(size, size, device=args.device)
    b = torch.randn(size, size, device=args.device)

    perf_logger = PerfLogger(args.perf_csv) if args.perf_csv else None
    cycle_logger = None

    iters = 0
    start = time.time()
    last_perf_log = start

    def maybe_log_perf(now: float) -> float:
        if perf_logger is not None and now - last_perf_log >= args.perf_interval:
            perf_logger.log(
                scenario=args.scenario,
                fault_active=read_fault_active(args.fault_flag_file),
                workload_mode=args.mode,
                matrix_size=size,
                iters_total=iters,
            )
            return now
        return last_perf_log

    try:
        if args.run_style == "continuous":
            while not stop:
                _ = torch.matmul(a, b)
                torch.cuda.synchronize()
                iters += 1

                if iters % args.report_every == 0:
                    elapsed = time.time() - start
                    print(f"iters={iters}, elapsed={elapsed:.1f}s", flush=True)

                now = time.time()
                last_perf_log = maybe_log_perf(now)
        else:
            if args.period_ms <= 0:
                raise ValueError("--period-ms must be > 0")
            if args.compute_budget_ms < 0:
                raise ValueError("--compute-budget-ms must be >= 0")
            if args.warmup_seconds < 0:
                raise ValueError("--warmup-seconds must be >= 0")

            period_s = args.period_ms / 1000.0
            compute_budget_s = args.compute_budget_ms / 1000.0

            print(
                f"Timeslice config: period_ms={args.period_ms:.1f}, "
                f"compute_budget_ms={args.compute_budget_ms:.1f}, "
                f"warmup_seconds={args.warmup_seconds:.1f}",
                flush=True,
            )
            cycle_csv_path = resolve_timeslice_cycles_csv_path(args)
            cycle_logger = TimesliceCycleLogger(cycle_csv_path)
            print(f"Timeslice cycle log: {cycle_csv_path}", flush=True)

            if args.warmup_seconds > 0:
                warmup_end = time.perf_counter() + args.warmup_seconds
                while not stop and time.perf_counter() < warmup_end:
                    _ = torch.matmul(a, b)
                    torch.cuda.synchronize()

                # Drop warmup from runtime/perf stats to keep timeslice measurements stable.
                iters = 0
                start = time.time()
                last_perf_log = start
                if perf_logger is not None:
                    perf_logger.last_ts = start
                    perf_logger.last_iters = 0

            cycle_idx = 0
            while not stop:
                cycle_idx += 1
                cycle_start = time.perf_counter()
                busy_start = cycle_start
                cycle_iters = 0

                # Keep issuing GEMM work until this cycle consumes its compute budget.
                while not stop:
                    _ = torch.matmul(a, b)
                    torch.cuda.synchronize()
                    iters += 1
                    cycle_iters += 1

                    if time.perf_counter() - busy_start >= compute_budget_s:
                        break

                compute_actual_s = time.perf_counter() - cycle_start
                remain_s = period_s - compute_actual_s
                if remain_s > 0:
                    time.sleep(remain_s)

                cycle_total_s = time.perf_counter() - cycle_start
                sleep_actual_s = max(cycle_total_s - compute_actual_s, 0.0)
                now_wall = time.time()

                if cycle_logger is not None:
                    cycle_logger.log(
                        ts=now_wall,
                        cycle_idx=cycle_idx,
                        compute_actual_ms=compute_actual_s * 1000.0,
                        sleep_actual_ms=sleep_actual_s * 1000.0,
                        period_actual_ms=cycle_total_s * 1000.0,
                        cycle_iters=cycle_iters,
                    )

                if args.cycle_report_every > 0 and cycle_idx % args.cycle_report_every == 0:
                    print(
                        f"cycle={cycle_idx}, "
                        f"compute_ms={compute_actual_s * 1000:.1f}, "
                        f"sleep_ms={sleep_actual_s * 1000:.1f}, "
                        f"period_ms={cycle_total_s * 1000:.1f}, "
                        f"cycle_iters={cycle_iters}",
                        flush=True,
                    )

                last_perf_log = maybe_log_perf(now_wall)

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
        if cycle_logger is not None:
            cycle_logger.close()
        print(f"Stopped. iters={iters}, elapsed={elapsed:.1f}s", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
