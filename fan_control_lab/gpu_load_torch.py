import argparse
import time
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=300)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--duty", type=float, default=1.0)
    ap.add_argument("--period-ms", type=int, default=100)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    a = torch.randn(args.size, args.size, device="cuda")
    b = torch.randn(args.size, args.size, device="cuda")

    period_s = args.period_ms / 1000.0
    busy_s = period_s * args.duty
    idle_s = max(0.0, period_s - busy_s)
    end_t = time.time() + args.seconds

    while time.time() < end_t:
        t0 = time.time()
        while time.time() - t0 < busy_s:
            c = a @ b
            _ = c.mean().item()
        if idle_s > 0:
            time.sleep(idle_s)

if __name__ == "__main__":
    main()