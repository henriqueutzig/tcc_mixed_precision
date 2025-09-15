#!/usr/bin/env python3
import argparse
import time
import sys

from utils import gpu_utils
from metrics_logger import MetricsLogger

# Import training functions
from train_pytorch import train_pytorch
from train_tensorflow import train_tensorflow

def parse_args():
    parser = argparse.ArgumentParser(description="Mixed Precision Benchmark Runner")
    parser.add_argument("--gpu-vendor", choices=["nvidia", "amd"], required=True,
                        help="Which GPU vendor to use: nvidia or amd")
    parser.add_argument("--framework", choices=["pytorch", "tensorflow"], required=True,
                        help="Deep learning framework to benchmark")
    parser.add_argument("--model", default="resnet50", type=str,
                        help="Model to benchmark (resnet50, bert, etc.)")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32",
                        help="Training precision")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--log-interval", type=int, default=5,
                        help="Seconds between GPU metric samples")
    return parser.parse_args()


def main():
    args = parse_args()

    # Detect and validate GPU
    if not gpu_utils.check_gpu_available(args.gpu_vendor):
        sys.exit(f"❌ No {args.gpu_vendor.upper()} GPU detected!")

    # Setup metrics logger
    metrics = MetricsLogger(
        gpu_vendor=args.gpu_vendor,
        framework=args.framework,
        model=args.model,
        precision=args.precision,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
    )

    metrics.start_logging()

    # Run training
    start = time.time()
    if args.framework == "pytorch":
        train_pytorch(args.model, args.batch_size, args.epochs, args.precision)
    elif args.framework == "tensorflow":
        train_tensorflow(args.model, args.batch_size, args.epochs, args.precision)
    end = time.time()

    metrics.stop_logging()
    metrics.finalize(total_time=end - start)

    print(f"✅ Benchmark completed in {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
