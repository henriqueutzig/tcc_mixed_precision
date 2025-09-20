#!/usr/bin/env bash
set -Eeuo pipefail

# Default settings (override via CLI flags)
GPU_VENDOR="amd"
GLOBAL_EPOCHS=""        # If set, overrides per-model epochs
REPEAT=1                # How many times to repeat each combo
OUTPUT_DIR="data/batch"
MODELS=("resnet50" "bert-large" "gpt2")
PRECISIONS=("fp32" "fp16" "bf16")

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --gpu-vendor {amd|nvidia}   GPU vendor (default: amd)
  --models "m1 m2 ..."        Subset of models (default: resnet50 bert-large gpt2)
  --precisions "p1 p2 ..."    Subset of precisions (default: fp32 fp16 bf16)
  --epochs N                  Override epochs for all models
  --repeat N                  Repeat each combination N times (default: 1)
  --output-dir PATH           Where to store CSV outputs (default: data/batch)
  -h | --help                 Show this help
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-vendor) GPU_VENDOR="$2"; shift 2 ;;
    --models) read -r -a MODELS <<<"$2"; shift 2 ;;
    --precisions) read -r -a PRECISIONS <<<"$2"; shift 2 ;;
    --epochs) GLOBAL_EPOCHS="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$OUTPUT_DIR"

timestamp() { date +"%Y%m%d_%H%M%S"; }

# Batch size selection per (model, precision)
batch_size_for() {
  local model="$1" prec="$2"
  case "$model" in
    resnet50)
      case "$prec" in
        fp32) echo 128 ;;
        fp16|bf16) echo 256 ;;
      esac
      ;;
    bert-large)
      case "$prec" in
        fp32) echo 32 ;;
        fp16|bf16) echo 64 ;;
      esac
      ;;
    gpt2)
      case "$prec" in
        fp32) echo 64 ;;
        fp16|bf16) echo 96 ;;
      esac
      ;;
    *) echo 16 ;;
  esac
}

# Epochs per model (training length control)
epochs_for() {
  local model="$1"
  if [[ -n "$GLOBAL_EPOCHS" ]]; then
    echo "$GLOBAL_EPOCHS"; return
  fi
  case "$model" in
    resnet50) echo 30 ;;
    bert-large) echo 20 ;;
    gpt2) echo 15 ;;
    *) echo 10 ;;
  esac
}

log () { echo "[$(date +'%H:%M:%S')] $*"; }

python_cmd() {
  # Allow override via PYTHON env var
  if [[ -n "${PYTHON:-}" ]]; then echo "$PYTHON"; else echo "python3"; fi
}

# Verify main script exists
if [[ ! -f "src/main.py" ]]; then
  echo "src/main.py not found."; exit 1
fi

RUN_ID="$(timestamp)"
MASTER_LOG="$OUTPUT_DIR/experiment_matrix_${RUN_ID}.log"
echo "Experiment run started: $RUN_ID" | tee -a "$MASTER_LOG"
echo "GPU vendor: $GPU_VENDOR" | tee -a "$MASTER_LOG"

for model in "${MODELS[@]}"; do
  for prec in "${PRECISIONS[@]}"; do
    bs="$(batch_size_for "$model" "$prec")"
    epochs="$(epochs_for "$model")"
    for ((r=1; r<=REPEAT; r++)); do
      base="${GPU_VENDOR}_train_${model}_${prec}_bs${bs}_e${epochs}_r${r}"
      out_csv="${base}.csv"
      out_path="${OUTPUT_DIR}/${out_csv}"
      raw_csv="${OUTPUT_DIR}/${base}_raw_metrics.csv"

      if [[ -f "$out_path" ]]; then
        log "Skip existing: $out_path"
        continue
      fi

      log "Run: model=${model} precision=${prec} batch=${bs} epochs=${epochs} rep=${r}"
      set +e
      $(python_cmd) src/main.py \
        --gpu-vendor "$GPU_VENDOR" \
        --model "$model" \
        --precision "$prec" \
        --batch-size "$bs" \
        --epochs "$epochs" \
        --output-file "$out_path" \
        > "${out_path%.csv}.stdout" 2> "${out_path%.csv}.stderr"
      ec=$?
      set -e

      if [[ $ec -ne 0 ]]; then
        log "FAILED ec=$ec model=${model} precision=${prec} rep=${r}" | tee -a "$MASTER_LOG"
      else
        log "OK -> $out_path" | tee -a "$MASTER_LOG"
        # Move raw metrics if produced in CWD
        raw_generated="${base}_raw_metrics.csv"
        if [[ -f "$raw_generated" ]]; then
          mv "$raw_generated" "$raw_csv"
        fi
        
        # Also copy csv values to $HOME dir for easier access
        cp "$out_path" "$HOME/tcc_mixed_precision/data/batch/${out_csv}"
        if [[ -f "$raw_csv" ]]; then
            cp "$raw_csv" "$HOME/tcc_mixed_precision/data/batch/${base}_raw_metrics.csv"
        fi

      fi
    done
  done
done

log "All done." | tee -a "$MASTER_LOG"