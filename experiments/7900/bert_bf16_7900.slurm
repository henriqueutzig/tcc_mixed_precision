#!/bin/bash
#SBATCH --job-name=bert_bf16_bs96_7900
#SBATCH --partition=lunaris
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err


set -euo pipefail

PROJECT_NAME="tcc_mixed_precision"
RUN_DIR="$SCRATCH/${PROJECT_NAME}"

echo "Creating run dir: $RUN_DIR"
mkdir -p "$RUN_DIR"

# Sync source (faster & safer than cp -r -f)
rsync -a "$HOME/$PROJECT_NAME/src/"      "$RUN_DIR/src/"
rsync -a "$HOME/$PROJECT_NAME/scripts/"  "$RUN_DIR/scripts/"

if [ ! -e "$RUN_DIR/envs/" ]; then
    rsync -a "$HOME/$PROJECT_NAME/envs/"  "$RUN_DIR/envs/"
fi

# Ensure experiment script is executable
chmod +x "$RUN_DIR/scripts/run_experiments.sh"

cd "$RUN_DIR"

newgrp render <<'EOF'

# Activate existing env (not copied)
source envs/amd/bin/activate

echo "Python: $(which python)"
python --version

for experiment_num in $(seq 1 10); do
    echo "Running BERT experiment number: $experiment_num"
    python src/main.py --gpu-vendor amd \
      --model bert-large --precision bf16 \
      --batch-size 96 --epochs 100 \
      --output-file data/bert_bf16_bs96_7900_10e.csv --experiment-num $experiment_num
    rsync -a *.json "$HOME/$PROJECT_NAME/epoch_metrics/"
done

EOF

# (Optional) single inference benchmark example:
# python src/inference_pytorch.py --model resnet50 --precision fp16 --batch-size 256 --num-batches 100 --warmup 5 --gpu-metrics --gpu-vendor amd --json > results/inference_resnet50_fp16.json

# Sync results back
rsync -a results "$HOME/$PROJECT_NAME/"

echo "Done."