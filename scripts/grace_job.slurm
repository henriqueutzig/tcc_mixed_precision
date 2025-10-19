#!/bin/bash
#SBATCH --job-name=mixed_precision_server_nvidia
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
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

# Detect architecture and set requirements file
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    REQUIREMENTS_FILE="scripts/ai.txt"
    echo "Using ARM-compatible requirements"
else
    REQUIREMENTS_FILE="scripts/ai.txt"
    echo "Using x86_64 requirements"
fi

if [ ! -e "$RUN_DIR/envs/ai_nvidia" ]; then
    cd "$RUN_DIR"
    python3 -m venv envs/ai_nvidia
    source envs/ai_nvidia/bin/activate

    # Upgrade pip first
    pip install --upgrade pip

    # Install PyTorch for ARM (ROCm/CPU fallback)
    if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi

    pip install -r "$REQUIREMENTS_FILE"
else
    cd "$RUN_DIR"
    source envs/ai_nvidia/bin/activate
    # pip install -r "$REQUIREMENTS_FILE"
fi

# pip install --extra-index-url https://pypi.nvidia.com nvidia-cudnn-cu12
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Ensure experiment script is executable
chmod +x "$RUN_DIR/scripts/run_experiments.sh"

echo "Python: $(which python)"
python --version
echo "Architecture: $(uname -m)"

# Test imports before running experiments
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run batch experiments
bash scripts/run_experiments.sh \
  --gpu-vendor nvidia \
  --repeat 1 \
  --output-dir results/ai_nvidia_runs

# Sync results back
rsync -a results "$HOME/$PROJECT_NAME/"

echo "Done."