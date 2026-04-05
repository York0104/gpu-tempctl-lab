import argparse
import csv
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run timeslice workload + gpu logger, then summarize results.")
    p.add_argument("--duration-sec", type=int, default=120, help="How long to run the test.")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--size", type=int, default=2560)
    p.add_argument("--period-ms", type=float, default=100.0)
    p.add_argument("--compute-budget-ms", type=float, default=30.0)
    p.add_argument("--warmup-seconds", type=float, default=2.0)
    p.add_argument("--sample-interval", type=float, default=1.0)
    p.add_argument("--cycle-report-every", type=int, default=100)
    p.add_argument("--output-root", type=str, default="logs")
    p.add_argument("--session-name", type=str, default="timeslice_eval")
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--plot", action="store_true", help="Generate plots if matplotlib is available.")
    return p.parse_args()


def start_proc(cmd, cwd: Path, stdout_path: Path):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    fp = stdout_path.open("w")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=fp,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True,
    )
    return proc, fp


def stop_proc(proc, fp=None, timeout=10):
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    if fp is not None and not fp.closed:
        fp.close()


def read_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="", errors="replace") as f:
        return list(csv.DictReader(f))


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarize_numeric(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {}
    s = sorted(vals)
    return {
        "count": len(vals),
        "mean": statistics.mean(vals),
        "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "p50": percentile(s, 0.50),
        "p95": percentile(s, 0.95),
        "max": max(vals),
    }


def print_metric_block(title, d, unit=""):
    if not d:
        print(f"{title}: no data")
        return
    suffix = f" {unit}" if unit else ""
    print(
        f"{title}: "
        f"count={d['count']}, "
        f"mean={d['mean']:.3f}{suffix}, "
        f"stdev={d['stdev']:.3f}{suffix}, "
        f"min={d['min']:.3f}{suffix}, "
        f"p50={d['p50']:.3f}{suffix}, "
        f"p95={d['p95']:.3f}{suffix}, "
        f"max={d['max']:.3f}{suffix}"
    )


def maybe_plot(run_dir: Path, thermal_rows, perf_rows, cycle_rows):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    out_paths = []

    # Thermal plot
    if thermal_rows:
        ts0 = to_float(thermal_rows[0].get("ts"))
        xs = []
        util = []
        temp = []
        gclk = []
        for r in thermal_rows:
            ts = to_float(r.get("ts"))
            if ts is None or ts0 is None:
                continue
            xs.append(ts - ts0)
            util.append(to_float(r.get("gpu_util_percent")))
            temp.append(to_float(r.get("temp_c")))
            gclk.append(to_float(r.get("graphics_clock_mhz")))

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(xs, util)
        axes[0].set_ylabel("GPU util (%)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(xs, temp)
        axes[1].set_ylabel("Temp (C)")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(xs, gclk)
        axes[2].set_ylabel("Graphics clock (MHz)")
        axes[2].set_xlabel("Elapsed (s)")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle("Thermal / Utilization Summary")
        fig.tight_layout()

        out_path = run_dir / "thermal_summary.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        out_paths.append(str(out_path))

    # Timeslice cycle plot
    if cycle_rows:
        xs = []
        compute = []
        sleep = []
        period = []
        cycle_iters = []
        for r in cycle_rows:
            idx = to_float(r.get("cycle_idx"))
            if idx is None:
                continue
            xs.append(idx)
            compute.append(to_float(r.get("compute_actual_ms")))
            sleep.append(to_float(r.get("sleep_actual_ms")))
            period.append(to_float(r.get("period_actual_ms")))
            cycle_iters.append(to_float(r.get("cycle_iters")))

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(xs, compute, label="compute_actual_ms")
        axes[0].plot(xs, sleep, label="sleep_actual_ms")
        axes[0].plot(xs, period, label="period_actual_ms")
        axes[0].set_ylabel("ms")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(xs, cycle_iters)
        axes[1].set_ylabel("cycle_iters")
        axes[1].set_xlabel("Cycle idx")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle("Timeslice Cycle Summary")
        fig.tight_layout()

        out_path = run_dir / "timeslice_summary.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        out_paths.append(str(out_path))

    return out_paths


def main():
    args = parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / args.session_name / f"ts_eval_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    thermal_csv = run_dir / "thermal.csv"
    perf_csv = run_dir / "perf.csv"
    cycle_csv = run_dir / "timeslice_cycles.csv"
    logger_stdout = run_dir / "gpu_logger_stdout.log"
    workload_stdout = run_dir / "workload_stdout.log"
    summary_json = run_dir / "summary.json"

    print(f"Run directory: {run_dir}", flush=True)

    logger_cmd = [
        args.python_bin,
        "gpu_logger.py",
        "--output", str(thermal_csv),
        "--gpu-index", str(args.gpu_index),
        "--sample-interval", str(args.sample_interval),
    ]

    workload_cmd = [
        args.python_bin,
        "workload_torch.py",
        "--size", str(args.size),
        "--run-style", "timeslice",
        "--period-ms", str(args.period_ms),
        "--compute-budget-ms", str(args.compute_budget_ms),
        "--warmup-seconds", str(args.warmup_seconds),
        "--cycle-report-every", str(args.cycle_report_every),
        "--perf-csv", str(perf_csv),
        "--scenario", "timeslice_eval",
    ]

    logger_proc = logger_fp = None
    workload_proc = workload_fp = None
    t0 = time.time()

    try:
        logger_proc, logger_fp = start_proc(logger_cmd, Path.cwd(), logger_stdout)
        time.sleep(1.0)  # let logger start

        workload_proc, workload_fp = start_proc(workload_cmd, Path.cwd(), workload_stdout)

        while True:
            elapsed = time.time() - t0
            if elapsed >= args.duration_sec:
                break
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("Interrupted by user, stopping processes...", flush=True)

    finally:
        stop_proc(workload_proc, workload_fp)
        stop_proc(logger_proc, logger_fp)

    thermal_rows = read_csv_rows(thermal_csv)
    perf_rows = read_csv_rows(perf_csv)
    cycle_rows = read_csv_rows(cycle_csv)

    util_stats = summarize_numeric([to_float(r.get("gpu_util_percent")) for r in thermal_rows])
    temp_stats = summarize_numeric([to_float(r.get("temp_c")) for r in thermal_rows])
    gclk_stats = summarize_numeric([to_float(r.get("graphics_clock_mhz")) for r in thermal_rows])
    perf_iter_stats = summarize_numeric([to_float(r.get("iter_per_sec")) for r in perf_rows])
    perf_ms_stats = summarize_numeric([to_float(r.get("avg_iter_ms")) for r in perf_rows])

    # ignore obviously truncated final cycle rows
    valid_cycle_rows = []
    for r in cycle_rows:
        c = to_float(r.get("compute_actual_ms"))
        p = to_float(r.get("period_actual_ms"))
        iters = to_float(r.get("cycle_iters"))
        if c is None or p is None or iters is None:
            continue
        if p < 80 or iters < 1:
            continue
        valid_cycle_rows.append(r)

    compute_stats = summarize_numeric([to_float(r.get("compute_actual_ms")) for r in valid_cycle_rows])
    sleep_stats = summarize_numeric([to_float(r.get("sleep_actual_ms")) for r in valid_cycle_rows])
    period_stats = summarize_numeric([to_float(r.get("period_actual_ms")) for r in valid_cycle_rows])
    iter_stats = summarize_numeric([to_float(r.get("cycle_iters")) for r in valid_cycle_rows])

    status = "UNKNOWN"
    recommendation = ""

    if util_stats:
        avg_util = util_stats["mean"]
        if 28.0 <= avg_util <= 35.0:
            status = "PASS"
            recommendation = "Average GPU utilization is in the first-pass acceptable range (28~35%)."
        elif avg_util < 28.0:
            status = "LOW"
            new_budget = args.compute_budget_ms * 30.0 / max(avg_util, 1e-6)
            recommendation = (
                f"Average GPU utilization is too low. "
                f"Try increasing compute_budget_ms from {args.compute_budget_ms} to about {new_budget:.1f}."
            )
        else:
            status = "HIGH"
            new_budget = args.compute_budget_ms * 30.0 / avg_util
            recommendation = (
                f"Average GPU utilization is too high. "
                f"Try reducing compute_budget_ms from {args.compute_budget_ms} to about {new_budget:.1f}."
            )

    plots = []
    if args.plot:
        plots = maybe_plot(run_dir, thermal_rows, perf_rows, valid_cycle_rows) or []

    summary = {
        "run_dir": str(run_dir),
        "args": vars(args),
        "status": status,
        "recommendation": recommendation,
        "thermal": {
            "gpu_util_percent": util_stats,
            "temp_c": temp_stats,
            "graphics_clock_mhz": gclk_stats,
        },
        "perf": {
            "iter_per_sec": perf_iter_stats,
            "avg_iter_ms": perf_ms_stats,
        },
        "timeslice_cycles": {
            "compute_actual_ms": compute_stats,
            "sleep_actual_ms": sleep_stats,
            "period_actual_ms": period_stats,
            "cycle_iters": iter_stats,
        },
        "artifacts": {
            "thermal_csv": str(thermal_csv),
            "perf_csv": str(perf_csv),
            "timeslice_cycles_csv": str(cycle_csv),
            "gpu_logger_stdout": str(logger_stdout),
            "workload_stdout": str(workload_stdout),
            "plots": plots,
        },
    }

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n===== Final Result =====")
    print(f"Status: {status}")
    print(f"Recommendation: {recommendation}")
    print(f"Run dir: {run_dir}\n")

    print_metric_block("GPU util", util_stats, "%")
    print_metric_block("GPU temp", temp_stats, "C")
    print_metric_block("Graphics clock", gclk_stats, "MHz")
    print_metric_block("Perf iter_per_sec", perf_iter_stats, "iter/s")
    print_metric_block("Perf avg_iter_ms", perf_ms_stats, "ms")
    print_metric_block("Cycle compute_actual_ms", compute_stats, "ms")
    print_metric_block("Cycle sleep_actual_ms", sleep_stats, "ms")
    print_metric_block("Cycle period_actual_ms", period_stats, "ms")
    print_metric_block("Cycle cycle_iters", iter_stats)

    if plots:
        print("\nGenerated plots:")
        for p in plots:
            print(f"  - {p}")

    print(f"\nSummary JSON: {summary_json}")


if __name__ == "__main__":
    main()
