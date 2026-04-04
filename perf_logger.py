import csv
import time
from pathlib import Path


class PerfLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.path.open("w", newline="")
        self.writer = csv.writer(self.fp)
        self.writer.writerow([
            "ts",
            "scenario",
            "fault_active",
            "workload_mode",
            "matrix_size",
            "iters_total",
            "iters_last_interval",
            "iter_per_sec",
            "avg_iter_ms",
        ])
        self.last_ts = time.time()
        self.last_iters = 0

    def log(
        self,
        *,
        scenario: str,
        fault_active: bool,
        workload_mode: str,
        matrix_size: int,
        iters_total: int,
    ) -> None:
        now = time.time()
        dt = max(now - self.last_ts, 1e-9)
        delta_iters = iters_total - self.last_iters
        iter_per_sec = delta_iters / dt
        avg_iter_ms = (1000.0 / iter_per_sec) if iter_per_sec > 0 else 0.0

        self.writer.writerow([
            now,
            scenario,
            int(fault_active),
            workload_mode,
            matrix_size,
            iters_total,
            delta_iters,
            iter_per_sec,
            avg_iter_ms,
        ])
        self.fp.flush()

        self.last_ts = now
        self.last_iters = iters_total

    def close(self) -> None:
        if not self.fp.closed:
            self.fp.close()
