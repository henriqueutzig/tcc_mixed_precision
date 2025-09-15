#!/usr/bin/env bash
set -e

# Configuration
VENDORS=("nvidia" "amd")
FRAMEWORKS=("pytorch" "tensorflow")
PRECISIONS=("fp32" "fp16" "bf16")
MODELS=("resnet50" "mobilenet_v2")
BATCH_SIZE=32
EPOCHS=1
LOG_INTERVAL=5

RESULTS_DIR="results/metrics"
mkdir -p $RESULTS_DIR

timestamp=$(date +"%Y%m%d-%H%M%S")
SUMMARY_FILE="$RESULTS_DIR/summary_${timestamp}.csv"

# Write CSV header
echo "vendor,framework,model,precision,batch_size,throughput(samples/sec),training_time(s),metrics_file" > $SUMMARY_FILE

for vendor in "${VENDORS[@]}"; do
    for framework in "${FRAMEWORKS[@]}"; do
        for precision in "${PRECISIONS[@]}"; do
            for model in "${MODELS[@]}"; do
                echo "🚀 Running $vendor | $framework | $model | $precision"

                # Activate correct venv
                if [[ "$vendor" == "nvidia" ]]; then
                    source envs/nvidia/bin/activate
                else
                    source envs/amd/bin/activate
                fi

                # Run benchmark
                OUTPUT=$(python src/main.py \
                    --gpu-vendor $vendor \
                    --framework $framework \
                    --model $model \
                    --precision $precision \
                    --batch-size $BATCH_SIZE \
                    --epochs $EPOCHS \
                    --log-interval $LOG_INTERVAL 2>&1)

                deactivate

                # Extract throughput & training time from logs
                THROUGHPUT=$(echo "$OUTPUT" | grep -oP "Throughput=\K[0-9.]+")
                TIME=$(echo "$OUTPUT" | grep -oP "Benchmark completed in \K[0-9.]+")
                METRICS_FILE=$(echo "$OUTPUT" | grep -oP "results/metrics/\S+\.csv")

                # Append to summary CSV
                echo "$vendor,$framework,$model,$precision,$BATCH_SIZE,$THROUGHPUT,$TIME,$METRICS_FILE" >> $SUMMARY_FILE
            done
        done
    done
done

echo "✅ All benchmarks completed. Summary saved to $SUMMARY_FILE"
