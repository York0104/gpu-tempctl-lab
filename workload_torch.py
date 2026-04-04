import argparse
import signal
import sys
import time
import torch

stop = False

def handle_stop(signum, frame):
    global stop
    stop = True

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=6144, help="Matrix size")
    p.add_argument("--report-every", type=int, default=50, help="Print every N iterations")
    return p.parse_args()

def main():
    global stop
    args = parse_args()

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    device = "cuda:0"
    print("Using device:", torch.cuda.get_device_name(0), flush=True)
    print("Matrix size:", args.size, flush=True)

    a = torch.randn(args.size, args.size, device=device)
    b = torch.randn(args.size, args.size, device=device)

    iters = 0
    start = time.time()

    while not stop:
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        iters += 1

        if iters % args.report_every == 0:
            elapsed = time.time() - start
            print(f"iters={iters}, elapsed={elapsed:.1f}s", flush=True)

    elapsed = time.time() - start
    print(f"Stopped. iters={iters}, elapsed={elapsed:.1f}s", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    main()
