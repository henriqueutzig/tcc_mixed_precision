import tensorflow as tf
import time


def train_tensorflow(model_name="resnet50", batch_size=32, epochs=1, precision="fp32"):
    # Mixed precision policy
    if precision == "fp16":
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
    elif precision == "bf16":
        policy = tf.keras.mixed_precision.Policy("mixed_bfloat16")
        tf.keras.mixed_precision.set_global_policy(policy)
    else:
        policy = tf.keras.mixed_precision.Policy("float32")
        tf.keras.mixed_precision.set_global_policy(policy)

    # Model
    if model_name.lower() == "resnet50":
        model = tf.keras.applications.ResNet50(weights=None, classes=1000)
    elif model_name.lower() == "mobilenet_v2":
        model = tf.keras.applications.MobileNetV2(weights=None, classes=1000)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn)

    # Synthetic dataset
    input_shape = (224, 224, 3)
    x = tf.random.normal((batch_size, *input_shape))
    y = tf.random.uniform((batch_size,), maxval=1000, dtype=tf.int32)

    warmup_steps = 10
    measured_steps = 50
    step_times = []

    # Training loop (manual for precise timing)
    @tf.function(jit_compile=True)
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    step = 0
    for epoch in range(epochs):
        while step < (warmup_steps + measured_steps):
            start = time.time()
            _ = train_step(x, y)
            end = time.time()

            if step >= warmup_steps:
                step_times.append(end - start)
            step += 1

    avg_time = sum(step_times) / len(step_times)
    throughput = batch_size / avg_time
    print(f"[TensorFlow] Model={model_name}, Precision={precision}, "
          f"Batch={batch_size}, Throughput={throughput:.2f} samples/sec")

    return throughput
