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

def train_model(model_name, precision, batch_size, epochs, device, num_batches=100):
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