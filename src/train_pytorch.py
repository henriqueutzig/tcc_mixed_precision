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

def train_model(model_name, precision, batch_size, epochs):
    """Main training function for PyTorch models.
    
    Supported model_name values:
      - resnet50
      - bert-large
      - tacotron2 (inference benchmark)
      - olmo-1b (causal LM training benchmark)
      - olmo-7b (causal LM training benchmark)
    """

    print(f"--- Training PyTorch Model: {model_name} | Precision: {precision} | Batch: {batch_size} ---")

    dtype = get_pytorch_dtype(precision)
    use_amp = precision in ['fp16', 'bf16']
    # GradScaler not needed for bf16
    scaler_enabled = (precision == 'fp16')
    
    # Initialize variables
    model = None
    optimizer = None
    loss_fn = None
    dummy_data = None
    dummy_targets = None
    sequences, lengths = None, None # For Tacotron2
    num_batches = 100
    causal_lm = False  # Flag for causal language models (OLMo)

    # --- Model-Specific Setup ---
    if model_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained=False, num_classes=10).to(device)
        input_shape = (3, 224, 224)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()
        dummy_data = [torch.randn(batch_size, *input_shape, device=device) for _ in range(num_batches)]
        dummy_targets = [torch.randint(0, 10, (batch_size,), device=device) for _ in range(num_batches)]
    
    elif model_name == 'bert-large':
        from transformers import BertForSequenceClassification, BertConfig
        config = BertConfig.from_pretrained('bert-large-uncased', num_labels=2)
        model = BertForSequenceClassification(config).to(device)
        input_shape = (128,) # Sequence length
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()
        dummy_data = [torch.randint(0, 30522, (batch_size, *input_shape), device=device) for _ in range(num_batches)]
        dummy_targets = [torch.randint(0, 2, (batch_size,), device=device) for _ in range(num_batches)]

    elif model_name in ['olmo-1b', 'olmo-7b']:
        # Causal LM benchmark using synthetic token batches
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf_name = 'allenai/OLMo-1B' if model_name == 'olmo-1b' else 'allenai/OLMo-7B'
        print(f"Loading {hf_name} ...")
        # Load tokenizer (no need to move to device)
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
        seq_len = 512
        # Load model (dtype handling: pass torch_dtype only for fp16/bf16, else default fp32)
        load_kwargs = {}
        if use_amp:
            load_kwargs['torch_dtype'] = dtype
        model = AutoModelForCausalLM.from_pretrained(hf_name, **load_kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        causal_lm = True
        # Synthetic random token sequences; targets = inputs (standard LM objective)
        dummy_data = [torch.randint(0, vocab_size, (batch_size, seq_len), device=device) for _ in range(num_batches)]
        dummy_targets = dummy_data  # Same tensors (do NOT clone to save memory)

    elif model_name == 'tacotron2':
        model_math_precision = 'fp16' if use_amp else 'fp32'
        tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math=model_math_precision).to(device)
        waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math=model_math_precision).to(device)
        model = {'tacotron2': tacotron2, 'waveglow': waveglow}
        utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
        dummy_texts = ["hello world, this is a benchmark"] * batch_size
        sequences, lengths = utils.prepare_input_sequence(dummy_texts, device=device)
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    
    # --- Training / Inference Loop ---
    total_samples = 0
    nan_inf_count = 0
    loss_history = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        if model_name == 'tacotron2': # Inference benchmark for TTS
            with torch.no_grad():
                for _ in tqdm(range(50)):
                    with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                        mel, _, _ = model['tacotron2'].infer(sequences, lengths)
                        _ = model['waveglow'].infer(mel)
                    total_samples += len(dummy_texts)
        else: # Training benchmark
            model.train()
            data_iter = zip(dummy_data, dummy_targets) if dummy_targets is not None else dummy_data
            for data, target in tqdm(data_iter, total=len(dummy_data)):
                optimizer.zero_grad(set_to_none=True)
                
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                    if causal_lm:
                        # Pass labels for internal loss computation
                        output = model(data, labels=target)
                        loss = output.loss
                    else:
                        output = model(data)
                        logits = output.logits if hasattr(output, 'logits') else output
                        loss = loss_fn(logits, target)

                if not torch.isfinite(loss):
                    nan_inf_count += 1
                    continue

                if scaler_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                loss_history.append(loss.item())
                total_samples += data.size(0)

    end_time = time.time()
    total_time = end_time - start_time
    avg_throughput = total_samples / total_time if total_time > 0 else 0

    return {
        'total_time_sec': total_time,
        'avg_throughput_samples_per_sec': avg_throughput,
        'final_loss': loss_history[-1] if loss_history else -1,
        'nan_inf_count': nan_inf_count,
        'loss_history': loss_history
    }