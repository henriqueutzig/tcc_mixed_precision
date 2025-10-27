#!/bin/bash
#SBATCH --job-name=infer_resnet50_fp16_bs32_7900
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

for experiment_num in $(seq 1 100); do
    echo "Running resnet50 experiment number: $experiment_num"
    python src/inference_pytorch.py --gpu-vendor amd \
      --model resnet50 --precision fp16 \
      --batch-size 32 \
      --experiment-num $experiment_num --json --gpu-index 1
    rsync -a *.json "$HOME/$PROJECT_NAME/infer_metrics/"
done

EOF

# (Optional) single inference benchmark example:
# python src/inference_pytorch.py --model resnet50 --precision bf16 --batch-size 256 --num-batches 100 --warmup 5 --gpu-metrics --gpu-vendor amd --json --gpu-index 1 > results/inference_resnet50_bf16.json

# Sync results back
rsync -a results "$HOME/$PROJECT_NAME/"

echo "Done."