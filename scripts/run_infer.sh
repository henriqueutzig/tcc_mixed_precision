#!/usr/bin/env bash
set -Eeuo pipefail

# Default settings (override via CLI flags)
GPU_VENDOR="amd"
NUM_BATCHES=""          # If set, overrides per-model num_batches
REPEAT=1                # How many times to repeat each combo
OUTPUT_DIR="data/inference"
MODELS=("resnet50" "bert-large" "gpt2")
PRECISIONS=("fp32" "fp16" "bf16")
USE_PRETRAINED=true
GPU_METRICS=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --gpu-vendor {amd|nvidia}   GPU vendor (default: amd)
  --models "m1 m2 ..."        Subset of models (default: resnet50 bert-large gpt2)
  --precisions "p1 p2 ..."    Subset of precisions (default: fp32 fp16 bf16)
  --num-batches N             Override num_batches for all models
  --repeat N                  Repeat each combination N times (default: 1)
  --output-dir PATH           Where to store CSV outputs (default: data/inference)
  --no-pretrained             Use randomly initialized models instead of pretrained
  --gpu-metrics               Enable GPU metrics logging
  -h | --help                 Show this help
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-vendor) GPU_VENDOR="$2"; shift 2 ;;
    --models) read -r -a MODELS <<<"$2"; shift 2 ;;
    --precisions) read -r -a PRECISIONS <<<"$2"; shift 2 ;;
    --num-batches) NUM_BATCHES="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --no-pretrained) USE_PRETRAINED=false; shift ;;
    --gpu-metrics) GPU_METRICS=true; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$OUTPUT_DIR"

timestamp() { date +"%Y%m%d_%H%M%S"; }

# Batch size selection per (model, precision) for inference
batch_size_for() {
  local model="$1" prec="$2"
  case "$model" in
    resnet50)
      case "$prec" in
        fp32) echo 64 ;;
        fp16|bf16) echo 128 ;;
      esac
      ;;
    bert-large)
      case "$prec" in
        fp32) echo 16 ;;
        fp16|bf16) echo 32 ;;
      esac
      ;;
    gpt2)
      case "$prec" in
        fp32) echo 32 ;;
        fp16|bf16) echo 48 ;;
      esac
      ;;
    *) echo 8 ;;
  esac
}

# Number of batches per model (inference benchmark length)
num_batches_for() {
  local model="$1"
  if [[ -n "$NUM_BATCHES" ]]; then
    echo "$NUM_BATCHES"; return
  fi
  case "$model" in
    resnet50) echo 100 ;;
    bert-large) echo 50 ;;
    gpt2) echo 30 ;;
    *) echo 50 ;;
  esac
}

# Warmup batches per model
warmup_for() {
  local model="$1"
  case "$model" in
    resnet50) echo 10 ;;
    bert-large) echo 5 ;;
    gpt2) echo 5 ;;
    *) echo 5 ;;
  esac
}

log () { echo "[$(date +'%H:%M:%S')] $*"; }

python_cmd() {
  # Allow override via PYTHON env var
  if [[ -n "${PYTHON:-}" ]]; then echo "$PYTHON"; else echo "python3"; fi
}

# Verify inference script exists
if [[ ! -f "src/inference_pytorch.py" ]]; then
  echo "src/inference_pytorch.py not found."; exit 1
fi

RUN_ID="$(timestamp)"
MASTER_LOG="$OUTPUT_DIR/inference_experiment_${RUN_ID}.log"
echo "Inference experiment run started: $RUN_ID" | tee -a "$MASTER_LOG"
echo "GPU vendor: $GPU_VENDOR" | tee -a "$MASTER_LOG"
echo "Use pretrained: $USE_PRETRAINED" | tee -a "$MASTER_LOG"
echo "GPU metrics: $GPU_METRICS" | tee -a "$MASTER_LOG"

for model in "${MODELS[@]}"; do
  for prec in "${PRECISIONS[@]}"; do
    bs="$(batch_size_for "$model" "$prec")"
    num_batches="$(num_batches_for "$model")"
    warmup="$(warmup_for "$model")"
    for ((r=1; r<=REPEAT; r++)); do
      pretrained_suffix=""
      if [[ "$USE_PRETRAINED" == "false" ]]; then
        pretrained_suffix="_nopretrained"
      fi
      
      base="${GPU_VENDOR}_inference_${model}_${prec}_bs${bs}_nb${num_batches}_r${r}${pretrained_suffix}"
      out_json="${base}.json"
      out_path="${OUTPUT_DIR}/${out_json}"

      if [[ -f "$out_path" ]]; then
        log "Skip existing: $out_path"
        continue
      fi

      log "Run: model=${model} precision=${prec} batch=${bs} num_batches=${num_batches} warmup=${warmup} rep=${r}"
      
      # Build command arguments
      cmd_args=(
        "src/inference_pytorch.py"
        "--model" "$model"
        "--precision" "$prec"
        "--batch-size" "$bs"
        "--num-batches" "$num_batches"
        "--warmup" "$warmup"
        "--gpu-vendor" "$GPU_VENDOR"
        "--json"
      )
      
      if [[ "$USE_PRETRAINED" == "false" ]]; then
        cmd_args+=("--no-pretrained")
      fi
      
      if [[ "$GPU_METRICS" == "true" ]]; then
        cmd_args+=("--gpu-metrics")
      fi
      
      set +e
      $(python_cmd) "${cmd_args[@]}" \
        > "$out_path" 2> "${out_path%.json}.stderr"
      ec=$?
      set -e

      if [[ $ec -ne 0 ]]; then
        log "FAILED ec=$ec model=${model} precision=${prec} rep=${r}" | tee -a "$MASTER_LOG"
        # Remove empty output file on failure
        [[ -f "$out_path" ]] && rm "$out_path"
      else
        log "OK -> $out_path" | tee -a "$MASTER_LOG"
        
        # Also copy json to $HOME dir for easier access
        mkdir -p "$HOME/tcc_mixed_precision/data/inference"
        cp "$out_path" "$HOME/tcc_mixed_precision/data/inference/${out_json}"
      fi
    done
  done
done

log "All inference experiments done." | tee -a "$MASTER_LOG"