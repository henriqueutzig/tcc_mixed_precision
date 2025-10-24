#!/bin/bash
#SBATCH --job-name=resnet50_fp16_b64_4090
#SBATCH --partition=tupi
#SBATCH --nodelist=tupi1,tupi2
#SBATCH --exclude=tupi3,tupi4,tupi5,tupi6
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
# Activate existing env (not copied)
source envs/nvidia/bin/activate

echo "Python: $(which python)"
python --version

for experiment_num in $(seq 1 100); do
    echo "Running resnet50 infer experiment number: $experiment_num"
    python src/inference_pytorch.py --gpu-vendor nvidia \
      --model resnet50 --precision fp16 \
      --batch-size 64 \
      --experiment-num $experiment_num --json
    rsync -a *.json "$HOME/$PROJECT_NAME/infer_metrics/"
done

# (Optional) single inference benchmark example:
# python src/inference_pytorch.py --model resnet50 --precision fp16 --batch-size 256 --num-batches 100 --warmup 5 --gpu-metrics --gpu-vendor amd --json > results/inference_resnet50_fp16.json

# Sync results back
rsync -a results "$HOME/$PROJECT_NAME/"

echo "Done."