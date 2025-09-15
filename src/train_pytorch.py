import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import time


def train_pytorch(model_name="resnet50", batch_size=32, epochs=1, precision="fp32"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    if model_name.lower() == "resnet50":
        model = models.resnet50()
    elif model_name.lower() == "mobilenet_v2":
        model = models.mobilenet_v2()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.to(device)

    # Mixed precision setup
    dtype = torch.float32
    if precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()
        use_amp = True
    elif precision == "bf16":
        scaler = None
        use_amp = True
        dtype = torch.bfloat16
    else:
        scaler = None
        use_amp = False

    # Synthetic data
    data = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)
    target = torch.randint(0, 1000, (batch_size,), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    warmup_steps = 10
    measured_steps = 50  # limit for fairness
    step_times = []

    model.train()
    step = 0
    for epoch in range(epochs):
        while step < (warmup_steps + measured_steps):
            start = time.time()

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    output = model(data)
                    loss = criterion(output, target)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            end = time.time()
            if step >= warmup_steps:
                step_times.append(end - start)

            step += 1

    avg_time = sum(step_times) / len(step_times)
    throughput = batch_size / avg_time
    print(f"[PyTorch] Model={model_name}, Precision={precision}, "
          f"Batch={batch_size}, Throughput={throughput:.2f} samples/sec")

    return throughput
