# train_tensorflow.py
import tensorflow as tf
import tensorflow_hub as hub
import time
import numpy as np

def set_tf_policy(precision):
    if precision == 'fp16':
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
    elif precision == 'bf16':
        policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
    else:
        policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)

def train_model(model_name, precision, batch_size, epochs, device_name):
    """Main training function for TensorFlow models."""
    print(f"--- Training TensorFlow Model: {model_name} | Precision: {precision} | Batch: {batch_size} ---")
    
    set_tf_policy(precision)

    with tf.device(device_name):
        # 1. Load Model
        if model_name == 'resnet50':
            model = tf.keras.applications.ResNet50(weights=None, classes=10)
            input_shape = (224, 224, 3)
        elif model_name == 'bert-large':
            tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
            tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
            
            text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
            preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
            encoder_inputs = preprocessing_layer(text_input)
            encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
            outputs = encoder(encoder_inputs)
            net = outputs['pooled_output']
            net = tf.keras.layers.Dense(2, activation='softmax', name='classifier')(net)
            model = tf.keras.Model(text_input, net)
            input_shape = () # Takes string input
        elif model_name == 'tacotron2':
            print("Tacotron2 benchmark is not implemented for TensorFlow in this script.")
            return {'total_time_sec': -1, 'avg_throughput_samples_per_sec': 0, 'final_loss': -1, 'nan_inf_count': 0, 'loss_history': -1}
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        if precision == 'fp16':
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss_fn)

        # 2. Generate Dummy Data
        num_batches = 100
        if model_name == 'resnet50':
            dummy_data = np.random.rand(num_batches * batch_size, *input_shape).astype('float32')
            dummy_targets = np.random.randint(0, 2, size=(num_batches * batch_size, 10)).astype('float32')
        elif model_name == 'bert-large':
            dummy_data = np.array(["this is a sample sentence"] * (num_batches * batch_size))
            dummy_targets = np.random.randint(0, 2, size=(num_batches * batch_size, 2)).astype('float32')
        
        dataset = tf.data.Dataset.from_tensor_slices((dummy_data, dummy_targets)).batch(batch_size)

        # 3. Training Loop
        start_time = time.time()
        
        history = model.fit(dataset, epochs=epochs, verbose=1)
        
        end_time = time.time()
        total_time = end_time - start_time
        total_samples = num_batches * batch_size * epochs
        avg_throughput = total_samples / total_time if total_time > 0 else 0
        
        loss_history = history.history.get('loss',)
        nan_inf_count = np.sum(np.isnan(loss_history)) + np.sum(np.isinf(loss_history))

    return {
        'total_time_sec': total_time,
        'avg_throughput_samples_per_sec': avg_throughput,
        'final_loss': loss_history[-1] if loss_history else -1,
        'nan_inf_count': int(nan_inf_count),
        'loss_history': loss_history
    }