# Inference benchmark with pretrained models
import argparse
import json
import time
import torch
import timm
from statistics import mean
from train_pytorch import get_pytorch_dtype, set_seed
from metrics_logger import GpuMetricsLogger

def make_resnet_iter(device, batch_size, num_batches):
    for _ in range(num_batches):
        yield torch.randn(batch_size, 3, 224, 224, device=device)

def make_bert_iter(device, batch_size, num_batches, vocab=30522, seq_len=128):
    for _ in range(num_batches):
        yield torch.randint(0, vocab, (batch_size, seq_len), device=device)

def make_gpt2_iter(device, batch_size, num_batches, vocab, seq_len=128, eos_id=None):
    for _ in range(num_batches):
        input_ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)
        attn = (input_ids != (eos_id if eos_id is not None else 0)).long()
        yield {"input_ids": input_ids, "attention_mask": attn}

def load_model(model_name, device, use_pretrained=True, bert_num_labels=2):
    if model_name == "resnet50":
        # timm pretrained=True loads ImageNet weights
        model = timm.create_model("resnet50", pretrained=use_pretrained, num_classes=1000)
        example_iter_builder = lambda batch_size, num_batches: make_resnet_iter(device, batch_size, num_batches)
    elif model_name == "bert-large":
        from transformers import BertForSequenceClassification
        if use_pretrained:
            # Loads pretrained base weights; classification head (num_labels) randomly init if different
            model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=bert_num_labels)
        else:
            from transformers import BertConfig
            config = BertConfig.from_pretrained("bert-large-uncased", num_labels=bert_num_labels)
            model = BertForSequenceClassification(config)
        example_iter_builder = lambda batch_size, num_batches: make_bert_iter(device, batch_size, num_batches)
    elif model_name == "gpt2":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained("gpt2") if use_pretrained else AutoModelForCausalLM.from_config(model.config_class())
        vocab = tok.vocab_size
        eos_id = tok.eos_token_id
        example_iter_builder = lambda batch_size, num_batches: make_gpt2_iter(device, batch_size, num_batches, vocab=vocab, eos_id=eos_id)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.eval().to(device)
    return model, example_iter_builder

def percentile(data, p):
    if not data:
        return 0.0
    idx = (len(data)-1) * p / 100.0
    lo = int(idx)
    hi = min(lo+1, len(data)-1)
    if hi == lo:
        return data[lo]
    frac = idx - lo
    return data[lo]*(1-frac) + data[hi]*frac

def benchmark_inference(model_name, precision, batch_size, num_batches, warmup, device,
                        use_gpu_logger=False, gpu_vendor="nvidia", gpu_index=0,
                        use_pretrained=True, bert_num_labels=2):
    set_seed(42)

    if precision not in ["fp32", "fp16", "bf16"]:
        raise ValueError("precision must be one of fp32|fp16|bf16")

    torch.backends.cuda.matmul.allow_tf32 = (precision == "fp32")
    torch.backends.cudnn.allow_tf32 = (precision == "fp32")

    dtype = get_pytorch_dtype(precision)
    use_amp = precision in ["fp16", "bf16"]

    model, iter_builder = load_model(model_name, device, use_pretrained=use_pretrained, bert_num_labels=bert_num_labels)

    warmup_iter = iter_builder(batch_size, warmup)
    bench_iter = iter_builder(batch_size, num_batches)

    gpu_logger = None
    if use_gpu_logger and device.type == "cuda":
        gpu_logger = GpuMetricsLogger(gpu_vendor=gpu_vendor, gpu_index=gpu_index, interval=0.5)
        gpu_logger.start()

    latencies = []
    total_samples = 0

    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        for batch in warmup_iter:
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                if model_name == "gpt2":
                    model(**batch)
                else:
                    model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    with torch.no_grad():
        for batch in bench_iter:
            t0 = time.time()
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                if model_name == "gpt2":
                    _ = model(**batch)
                else:
                    _ = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            latencies.append(t1 - t0)
            total_samples += batch_size
    end_time = time.time()

    if gpu_logger:
        gpu_logger.stop()

    wall_time = end_time - start_time
    throughput = total_samples / wall_time if wall_time > 0 else 0.0
    lat_sorted = sorted(latencies)
    metrics = {
        "model": model_name,
        "precision": precision,
        "pretrained": use_pretrained,
        "device": str(device),
        "batch_size": batch_size,
        "num_batches": num_batches,
        "warmup_batches": warmup,
        "total_samples": total_samples,
        "total_inference_time_sec": wall_time,
        "throughput_samples_per_sec": throughput,
        "mean_batch_latency_ms": (mean(latencies) * 1000) if latencies else 0.0,
        "p50_latency_ms": percentile(lat_sorted, 50) * 1000,
        "p90_latency_ms": percentile(lat_sorted, 90) * 1000,
        "p95_latency_ms": percentile(lat_sorted, 95) * 1000,
        "p99_latency_ms": percentile(lat_sorted, 99) * 1000,
        "peak_memory_mb": (torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else 0
    }

    if gpu_logger:
        agg_gpu, _ = gpu_logger.get_results()
        metrics.update({
            "gpu_peak_memory_logger_mb": agg_gpu['peak_memory_mb'],
            "gpu_avg_util_percent": agg_gpu['avg_utilization_percent'],
            "gpu_avg_power_watts": agg_gpu['avg_power_watts']
        })
    return metrics

def parse_args():
    ap = argparse.ArgumentParser(description="PyTorch Inference Mixed Precision Benchmark (Pretrained)")
    ap.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "bert-large", "gpt2"])
    ap.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-batches", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--gpu-metrics", action="store_true")
    ap.add_argument("--gpu-vendor", type=str, default="nvidia", choices=["nvidia", "amd"])
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true", help="Disable pretrained weights (for comparison).")
    ap.add_argument("--bert-num-labels", type=int, default=2, help="Num labels for BERT classification head.")
    ap.add_argument("--json", action="store_true", help="Output JSON only.")
    ap.add_argument("--experiment-num", type=int, default=1, help="Experiment number for logging.")
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = benchmark_inference(
        model_name=args.model,
        precision=args.precision,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        warmup=args.warmup,
        device=device,
        use_gpu_logger=args.gpu_metrics,
        gpu_vendor=args.gpu_vendor,
        gpu_index=args.gpu_index,
        use_pretrained=not args.no_pretrained,
        bert_num_labels=args.bert_num_labels   
    )
    metrics["experiment_num"] = args.experiment_num

    if args.json:
        gpu_name = torch.cuda.get_device_name(device) 
        safe_gpu_name = gpu_name.replace(" ", "_") if gpu_name else "unknown"
        file_name = f"infer_metrics_{safe_gpu_name}_{args.model}_pr{args.precision}_bs{args.batch_size}.json"
        with open(file_name, "a") as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps(metrics, indent=2))
        print(f"Saved metrics JSON to {file_name}")
    else:
        print("Inference Benchmark Results (Pretrained)")
        for k, v in metrics.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()