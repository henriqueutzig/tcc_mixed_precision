# main.py
import argparse
import pandas as pd
import os
import datetime
import torch
import tensorflow as tf

from metrics_logger import GpuMetricsLogger
import train_pytorch
import train_tensorflow

def main(args):
    # Setup device
    if args.framework == 'pytorch':
        if not torch.cuda.is_available():
            print("PyTorch CUDA/ROCm not available. Exiting.")
            return
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    elif args.framework == 'tensorflow':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # FIX: Iterate over the list of GPUs and set memory growth for each one.
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
            
            device = '/GPU:0'
            device_name = gpus.name # FIX: Get name from the first GPU object in the list
    else:
        print(f"Framework {args.framework} not supported.")
        return

    print(f"Running on device: {device_name}")

    # Start metrics logger
    metrics_logger = GpuMetricsLogger(gpu_vendor=args.gpu_vendor)
    metrics_logger.start()

    # Run training
    training_results = {}
    if args.framework == 'pytorch':
        training_results = train_pytorch.train_model(
            model_name=args.model,
            precision=args.precision,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=device
        )
    elif args.framework == 'tensorflow':
        training_results = train_tensorflow.train_model(
            model_name=args.model,
            precision=args.precision,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device_name=device
        )
    
    # Stop metrics logger and get results
    metrics_logger.stop()
    hardware_results = metrics_logger.get_results()

    # Combine results
    all_results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'gpu_vendor': args.gpu_vendor,
        'gpu_name': device_name,
        'framework': args.framework,
        'model': args.model,
        'precision': args.precision,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        **training_results,
        **hardware_results
    }
    
    # Clean up large fields for CSV logging
    if 'loss_history' in all_results:
        del all_results['loss_history']

    # Save results to CSV
    df = pd.DataFrame([all_results])
    if not os.path.exists(args.output_file):
        df.to_csv(args.output_file, index=False)
    else:
        df.to_csv(args.output_file, mode='a', header=False, index=False)

    print("\n--- Benchmark Complete ---")
    for key, value in all_results.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"Results saved to {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mixed Precision Benchmark Suite")
    parser.add_argument('--gpu-vendor', type=str, required=True, choices=['nvidia', 'amd'], help='GPU manufacturer')
    parser.add_argument('--framework', type=str, required=True, choices=['pytorch', 'tensorflow'], help='ML Framework')
    parser.add_argument('--model', type=str, required=True, choices=['resnet50', 'bert-large', 'tacotron2'], help='Model to benchmark')
    parser.add_argument('--precision', type=str, required=True, choices=['fp32', 'fp16', 'bf16'], help='Training precision')
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to run')
    parser.add_argument('--output-file', type=str, default='benchmark_results.csv', help='File to save results')
    
    args = parser.parse_args()
    main(args)