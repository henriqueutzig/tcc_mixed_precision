# train_pytorch.py
import torch
import time
from tqdm import tqdm
import timm

def get_pytorch_dtype(precision):
    if precision == 'fp16':
        return torch.float16
    elif precision == 'bf16':
        return torch.bfloat16
    return torch.float32

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_model(model_name, precision, batch_size, epochs, device, num_batches=100, gpu_vendor=None, experiment_num=None):
    """Main training function for PyTorch models.

    Supported model_name values:
      - resnet50
      - bert-large
      - gpt2 (causal LM)
    """
    set_seed(42)
    print(f"--- Training PyTorch Model: {model_name} | Precision: {precision} | Batch: {batch_size} ---")

    torch.backends.cuda.matmul.allow_tf32 = (precision == 'fp32')
    torch.backends.cudnn.allow_tf32 = (precision == 'fp32')

    dtype = get_pytorch_dtype(precision)
    use_amp = precision in ['fp16', 'bf16']
    scaler_enabled = (precision == 'fp16')

    model = None
    optimizer = None
    loss_fn = None
    causal_lm = False
    nan_inf_count = 0
    skipped_steps = 0
    loss_history = []
    total_samples = 0

    if model_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained=False, num_classes=10).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()
        def batch_iter():
            for _ in range(num_batches):
                data = torch.randn(batch_size, 3, 224, 224, device=device)
                target = torch.randint(0, 10, (batch_size,), device=device)
                yield data, target

    elif model_name == 'bert-large':
        from transformers import BertForSequenceClassification, BertConfig
        config = BertConfig.from_pretrained('bert-large-uncased', num_labels=2)
        model = BertForSequenceClassification(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()
        vocab = 30522
        seq_len = 128
        def batch_iter():
            for _ in range(num_batches):
                data = torch.randint(0, vocab, (batch_size, seq_len), device=device)
                target = torch.randint(0, 2, (batch_size,), device=device)
                yield data, target

    elif model_name == 'gpt2':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        causal_lm = True
        seq_len = 128
        eos_id = tokenizer.eos_token_id
        vocab = tokenizer.vocab_size
        def batch_iter():
            for _ in range(num_batches):
                data = torch.randint(0, vocab, (batch_size, seq_len), device=device)
                # Next-token prediction: labels are input_ids shifted (simplified synthetic)
                labels = data.clone()
                yield {'input_ids': data, 'attention_mask': (data != eos_id).long()}, labels
    elif model_name == '': 
        pass  # Placeholder for additional models
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    start_time = time.time()
    torch.cuda.synchronize()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        epoch_start = time.time()
        for step, batch in enumerate(batch_iter()):
            if causal_lm:
                batch_inputs, labels = batch
            else:
                batch_inputs, labels = batch

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                if causal_lm:
                    output = model(**batch_inputs, labels=labels)
                    loss = output.loss
                else:
                    output = model(batch_inputs)
                    logits = output.logits if hasattr(output, 'logits') else output
                    loss = loss_fn(logits, labels)

            if not torch.isfinite(loss):
                nan_inf_count += 1
                skipped_steps += 1
                continue

            if scaler_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_history.append(loss.item())
            total_samples += batch_size

        torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} time: {epoch_time:.3f}s | Last loss: {loss_history[-1]:.4f}")
        log_epoch_metrics(batch_size=batch_size, num_batches=num_batches, epoch=epoch,
                          epoch_time=epoch_time, loss_history=loss_history,
                          nan_inf_count=nan_inf_count, skipped_steps=skipped_steps, device=device,
                          model=model_name, precision=precision, epochs=epochs, gpu_vendor=gpu_vendor, experiment_num=experiment_num)

    torch.cuda.synchronize()
    total_time = time.time() - start_time
    avg_throughput = total_samples / total_time if total_time > 0 else 0.0

    return {
        'total_time_sec': total_time,
        'avg_throughput_samples_per_sec': avg_throughput,
        'final_loss': loss_history[-1] if loss_history else -1,
        'nan_inf_count': nan_inf_count,
        'skipped_steps': skipped_steps,
        'loss_history': loss_history
    }

def log_epoch_metrics(batch_size, num_batches, epoch, epoch_time, loss_history, nan_inf_count, skipped_steps, device, model, precision, epochs, gpu_vendor, experiment_num):
    # Per-epoch performance / GPU efficiency logging
        if 'epoch_logs' not in locals():
            epoch_logs = []
        epoch_samples = batch_size * num_batches
        epoch_throughput = epoch_samples / epoch_time if epoch_time > 0 else 0.0

        # GPU memory stats
        mem_alloc = torch.cuda.memory_allocated(device)
        mem_reserved = torch.cuda.memory_reserved(device)
        max_mem_alloc = torch.cuda.max_memory_allocated(device)
        max_mem_reserved = torch.cuda.max_memory_reserved(device)

        # GPU utilization via NVML (optional)
        gpu_util = None
        mem_util = None
        power = None
        if gpu_vendor == 'nvidia':
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util_rates.gpu          # %
                mem_util = util_rates.memory       # %
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
            except Exception:
                pass
        elif gpu_vendor == 'amd':
            try:
                from pyrsmi import rocml
                rocml.smi_initialize()
                gpu_index = torch.cuda.current_device()
                gpu_util = rocml.smi_get_device_utilization(gpu_index)  # %
                mem_util = rocml.smi_get_device_memory_used(self.gpu_index)
                power = rocml.smi_get_device_average_power(gpu_index)    # W
            except Exception:
                pass

        epoch_log = {
            'experiment_num': experiment_num,
            'gpu_vendor': gpu_vendor,
            'model': model,
            'precision': precision,
            'batch_size': batch_size,
            'epoch': epoch + 1,
            'loss': loss_history[-1],
            'epoch_time_sec': epoch_time,
            'samples': epoch_samples,
            'samples_per_sec': epoch_throughput,
            'nan_inf_count_so_far': nan_inf_count,
            'skipped_steps_so_far': skipped_steps,
            'mem_alloc_bytes': mem_alloc,
            'mem_reserved_bytes': mem_reserved,
            'max_mem_alloc_bytes': max_mem_alloc,
            'max_mem_reserved_bytes': max_mem_reserved,
            'gpu_util_percent': gpu_util,
            'mem_util_percent': mem_util,
            'power_watts': power
        }
        epoch_logs.append(epoch_log)

        # Console summary
        print(f"[Epoch {epoch+1}] throughput={epoch_throughput:.2f} samples/s | "
              f"loss={loss_history[-1]:.4f} | alloc={mem_alloc/1e6:.1f}MB "
              f"peak_alloc={max_mem_alloc/1e6:.1f}MB | gpu_util={gpu_util if gpu_util is not None else 'NA'}%")

        # Append to log file
        try:
            with open(f"epoch_metrics_{model}_bs{batch_size}_pr{precision}_e{epochs}.json", "a") as f:
                f.write(str(epoch_log) + "\n")
        except Exception:
            pass