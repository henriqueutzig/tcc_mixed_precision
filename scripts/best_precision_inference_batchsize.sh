#!/usr/bin/env bash
set -Eeuo pipefail

# Find best (batch_size, precision) for inference throughput per model.

# DEFAULTS ---------------------------------------------------------------------
GPU_VENDOR="nvidia"
MODELS=("resnet50" "bert-large" "gpt2")
PRECISIONS=("fp32" "fp16" "bf16")
BATCH_SIZES=(1024 768 512 384 256 192 128 96 64 48 32 24 16 12 8 4 2 1)
REPEAT=1
WARMUP=5
NUM_BATCHES=50
OUTPUT_DIR="results_inference_batch"
FORCE=0
DEVICE=""            # e.g. cuda:0 or cpu
EXTRA_ARGS=()        # passthrough

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]
  --gpu-vendor {nvidia|amd}
  --models "m1 m2 ..."
  --precisions "p1 p2 ..."
  --batch-sizes "b1 b2 ..."
  --repeat N
  --warmup N
  --num-batches N
  --device DEV              (e.g. cuda:0, cpu)
  --output-dir PATH
  --force
  --extra "ARGS"            (extra args passed to inference script)
  -h | --help
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-vendor) GPU_VENDOR="$2"; shift 2 ;;
    --models) read -r -a MODELS <<<"$2"; shift 2 ;;
    --precisions) read -r -a PRECISIONS <<<"$2"; shift 2 ;;
    --batch-sizes) read -r -a BATCH_SIZES <<<"$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --num-batches) NUM_BATCHES="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --extra) read -r -a EXTRA_ARGS <<<"$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

[[ -f "src/inference_pytorch.py" ]] || { echo "src/inference_pytorch.py not found"; exit 1; }

mkdir -p "$OUTPUT_DIR"
RUN_LOG="$OUTPUT_DIR/run_infer_matrix.log"

python_cmd() { if [[ -n "${PYTHON:-}" ]]; then echo "$PYTHON"; else echo "python3"; fi; }
log() { echo "[$(date +'%H:%M:%S')] $*"; }

echo "Run started $(date)" | tee -a "$RUN_LOG"
echo "Vendor=$GPU_VENDOR Repeat=$REPEAT" | tee -a "$RUN_LOG"

# EXECUTION LOOP ---------------------------------------------------------------
for model in "${MODELS[@]}"; do
  for prec in "${PRECISIONS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
      for ((r=1; r<=REPEAT; r++)); do
        base="${GPU_VENDOR}_infer_${model}_${prec}_bs${bs}_r${r}"
        out_json="${OUTPUT_DIR}/${base}.json"
        stdout_f="${OUTPUT_DIR}/${base}.stdout"
        stderr_f="${OUTPUT_DIR}/${base}.stderr"

        if [[ -f "$out_json" && $FORCE -eq 0 ]]; then
          log "Skip existing: $out_json"
          continue
        fi

        log "Run model=${model} prec=${prec} bs=${bs} rep=${r}"
        set +e
        $(python_cmd) src/inference_pytorch.py \
          --model "$model" \
          --precision "$prec" \
          --batch-size "$bs" \
          --num-batches "$NUM_BATCHES" \
          --warmup "$WARMUP" \
          --gpu-vendor "$GPU_VENDOR" \
          --json \
          ${DEVICE:+--device "$DEVICE"} \
          "${EXTRA_ARGS[@]}" \
          >"$out_json" 2>"$stderr_f"
        ec=$?
        set -e
        if [[ $ec -ne 0 ]]; then
          log "FAILED ec=$ec $base" | tee -a "$RUN_LOG"
          # Write minimal JSON so aggregation sees failure
          cat >"$out_json" <<EOF
{"model":"$model","precision":"$prec","batch_size":$bs,"throughput_samples_per_sec":0,"mean_batch_latency_ms":0,"total_inference_time_sec":0,"success":0}
EOF
        else
          log "OK -> $out_json" | tee -a "$RUN_LOG"
          # Add success flag if not present
          if ! grep -q '"success"' "$out_json"; then
            sed -i.bak 's/}$/,"success":1}/' "$out_json" 2>/dev/null || \
            $(python_cmd) - <<PY
import json,sys
p="$out_json"
d=json.load(open(p))
d.setdefault("success",1)
json.dump(d,open(p,"w"))
PY
            rm -f "$out_json.bak" 2>/dev/null || true
          fi
        fi
      done
    done
  done
done

# AGGREGATION ------------------------------------------------------------------
SUMMARY_CSV="$OUTPUT_DIR/summary.csv"
echo "model,precision,batch_size,throughput,mean_latency_ms,total_time_sec,success" > "$SUMMARY_CSV"

for f in "$OUTPUT_DIR"/*_infer_*_bs*_r*.json; do
  [[ -f "$f" ]] || continue
  $(python_cmd) - <<PY >> "$SUMMARY_CSV"
import json,sys
path="$f"
try:
    d=json.load(open(path))
except:
    sys.exit(0)
model=d.get("model","")
precision=d.get("precision","")
bs=d.get("batch_size",0)
thr=d.get("throughput_samples_per_sec", d.get("throughput",0))
lat=d.get("mean_batch_latency_ms", d.get("mean_latency_ms",0))
tot=d.get("total_inference_time_sec", d.get("total_time_sec",0))
success=int(d.get("success",1))
print(f"{model},{precision},{bs},{thr},{lat},{tot},{success}")
PY
done

# BEST SELECTION ---------------------------------------------------------------
BEST_CSV="$OUTPUT_DIR/best.csv"
echo "model,precision,best_batch_size,throughput" > "$BEST_CSV"

awk -F, 'NR>1 && $7==1 {
  key=$1 FS $2
  if ($4+0 > best_thr[key]+0) {
    best_thr[key]=$4; best_bs[key]=$3; m[key]=$1; p[key]=$2
  }
}
END {
  for (k in best_thr) {
    print m[k]","p[k]","best_bs[k]","best_thr[k]
  }
}' "$SUMMARY_CSV" >> "$BEST_CSV"

log "Summary CSV: $SUMMARY_CSV"
log "Best CSV:    $BEST_CSV"
column -t -s, "$BEST_CSV" 2>/dev/null || cat "$BEST_CSV"