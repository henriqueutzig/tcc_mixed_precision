#!/usr/bin/env bash
set -Eeuo pipefail

# PSEUDOCODE -------------------------------------------------------------------
# 1. Parse CLI options similar to run_experiment.sh:
#       --gpu-vendor, --models, --precisions, --batch-sizes, --epochs, --repeat, --output-dir
# 2. Default arrays: MODELS(resnet50 bert-large gpt2), PRECISIONS(fp32 fp16 bf16), BATCH_SIZES(descending).
# 3. For each (model, precision, batch_size) and repetition:
#       Build base name: <vendor>_train_<model>_<precision>_bs<batch>_e<epochs>_r<rep>
#       Output single-run CSV: OUTPUT_DIR/base.csv
#       Invoke python main.py with flags:
#           --gpu-vendor, --model, --precision, --batch-size, --epochs, --output-file
#       Capture stdout/stderr to sidecar files.
#       Skip if CSV already exists unless FORCE flag provided.
# 4. After all runs:
#       Aggregate all per-run CSVs into SUMMARY_CSV (model,precision,batch_size,elapsed_sec,throughput,success).
#       For each run file:
#           Derive model, precision, batch from filename (regex).
#           Use Python snippet to read the CSV:
#               - Load header
#               - Identify time column: prefer total_time_sec, elapsed_sec, time_sec else 0
#               - Identify throughput column: prefer avg_throughput_samples_per_sec, throughput, samples_per_sec else 0
#               - If no success column, assume success=1; else read (success or status).
#               - Use first non-header data row.
#           Emit standardized CSV row appended to SUMMARY_CSV (skip duplicates).
# 5. Compute BEST_CSV:
#       awk over SUMMARY_CSV selecting max throughput among success==1 per (model, precision).
# 6. Print paths and formatted best table.
# 7. Exit.

# CONFIG / DEFAULTS ------------------------------------------------------------
GPU_VENDOR="nvidia"
MODELS=("resnet50" "bert-large" "gpt2")
PRECISIONS=("fp32" "fp16" "bf16")
# Descending order to attempt largest first
BATCH_SIZES=(1024 768 512 384 256 192 128 96 64 48 32 24 16 12 8 4 2 1)
GLOBAL_EPOCHS="1"
REPEAT=1
OUTPUT_DIR="results_precision_batch"
FORCE=0          # If 1 re-run even if file exists

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]
  --gpu-vendor {amd|nvidia}
  --models "m1 m2 ..."
  --precisions "p1 p2 ..."
  --batch-sizes "b1 b2 ..."
  --epochs N
  --repeat N
  --output-dir PATH
  --force                 Re-run even if per-run CSV exists
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
    --epochs) GLOBAL_EPOCHS="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$OUTPUT_DIR"

python_cmd() { if [[ -n "${PYTHON:-}" ]]; then echo "$PYTHON"; else echo "python3"; fi; }
log() { echo "[$(date +'%H:%M:%S')] $*"; }

epochs_for() {
  local model="$1"
  if [[ -n "$GLOBAL_EPOCHS" ]]; then echo "$GLOBAL_EPOCHS"; return
  fi
  case "$model" in
    resnet50) echo 1 ;;        # keep very small for benchmarking synthetic
    bert-large) echo 1 ;;
    gpt2) echo 1 ;;
    *) echo 1 ;;
  esac
}

# Verify existence of python entry
if [[ ! -f "src/main.py" ]]; then
  echo "src/main.py not found."; exit 1
fi

RUN_LOG="$OUTPUT_DIR/run_matrix.log"
echo "Run started $(date)" | tee -a "$RUN_LOG"
echo "Vendor=$GPU_VENDOR Repeat=$REPEAT" | tee -a "$RUN_LOG"

# EXECUTION LOOP ---------------------------------------------------------------
for model in "${MODELS[@]}"; do
  for prec in "${PRECISIONS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
      epochs="$(epochs_for "$model")"
      for ((r=1; r<=REPEAT; r++)); do
        base="${GPU_VENDOR}_train_${model}_${prec}_bs${bs}_e${epochs}_r${r}"
        out_csv="${OUTPUT_DIR}/${base}.csv"
        stdout_f="${OUTPUT_DIR}/${base}.stdout"
        stderr_f="${OUTPUT_DIR}/${base}.stderr"

        if [[ -f "$out_csv" && $FORCE -eq 0 ]]; then
          log "Skip existing: $out_csv"
          continue
        fi

        log "Run model=${model} prec=${prec} bs=${bs} epochs=${epochs} rep=${r}"
        set +e
        $(python_cmd) src/main.py \
          --gpu-vendor "$GPU_VENDOR" \
          --model "$model" \
            --precision "$prec" \
          --batch-size "$bs" \
          --epochs "$epochs" \
          --output-file "$out_csv" \
          >"$stdout_f" 2>"$stderr_f"
        ec=$?
        set -e

        if [[ $ec -ne 0 ]]; then
          log "FAILED ec=$ec base=$base" | tee -a "$RUN_LOG"
        else
          log "OK -> $out_csv" | tee -a "$RUN_LOG"
        fi
      done
    done
  done
done

# AGGREGATION ------------------------------------------------------------------
SUMMARY_CSV="$OUTPUT_DIR/summary.csv"
echo "model,precision,batch_size,elapsed_sec,throughput,success" > "$SUMMARY_CSV"

# Iterate over generated CSVs and normalize metrics via Python
for f in "$OUTPUT_DIR"/*_train_*_bs*_e*_r*.csv; do
  [[ -f "$f" ]] || continue
  # Derive pieces from filename
  fname="$(basename "$f")"
  # pattern: vendor_train_model_precision_bsX_eY_rZ.csv
  model=$(echo "$fname" | sed -E 's/^[^_]+_train_([^_]+)_.*/\1/')
  precision=$(echo "$fname" | sed -E 's/^[^_]+_train_[^_]+_([^_]+)_.*/\1/')
  batch_size=$(echo "$fname" | sed -E 's/.*_bs([0-9]+)_e[0-9]+_r[0-9]+\.csv/\1/')

  # Avoid duplicate rows
  if grep -q "^${model},${precision},${batch_size}," "$SUMMARY_CSV"; then
    continue
  fi

  py_json="$(
    $(python_cmd) - <<PY
import csv, json, sys, re
path = "$f"
model="$model"; precision="$precision"; batch=int("$batch_size")
elapsed=0.0; throughput=0.0; success=1
try:
    with open(path, newline="") as fh:
        rd = list(csv.reader(fh))
    if not rd or len(rd[0])==0:
        raise ValueError("empty csv")
    header = rd[0]
    rows = rd[1:]
    if not rows:
        raise ValueError("no data rows")
    row = rows[0]
    idx = {h:i for i,h in enumerate(header)}
    # Candidate columns
    time_cols = [c for c in ["total_time_sec","elapsed_sec","time_sec"] if c in idx]
    thr_cols = [c for c in ["avg_throughput_samples_per_sec","throughput","samples_per_sec"] if c in idx]
    success_cols = [c for c in ["success","status"] if c in idx]
    if time_cols:
        try: elapsed = float(row[idx[time_cols[0]]])
        except: pass
    if thr_cols:
        try: throughput = float(row[idx[thr_cols[0]]])
        except: pass
    if success_cols:
        val = row[idx[success_cols[0]]].strip()
        if val.lower() in ("0","false","fail"): success=0
except Exception as e:
    success=0
print(json.dumps({"model":model,"precision":precision,"batch":batch,"elapsed":elapsed,"throughput":throughput,"success":success}))
PY
  )"

  m=$(echo "$py_json" | sed -n 's/.*"model":"\([^"]*\)".*/\1/p')
  p=$(echo "$py_json" | sed -n 's/.*"precision":"\([^"]*\)".*/\1/p')
  b=$(echo "$py_json" | sed -n 's/.*"batch":\([0-9]\+\).*/\1/p')
  e=$(echo "$py_json" | sed -n 's/.*"elapsed":\([0-9.]\+\).*/\1/p')
  t=$(echo "$py_json" | sed -n 's/.*"throughput":\([0-9.]\+\).*/\1/p')
  s=$(echo "$py_json" | sed -n 's/.*"success":\([01]\).*/\1/p')
  [[ -z "$m" ]] && m="$model"
  [[ -z "$p" ]] && p="$precision"
  [[ -z "$b" ]] && b="$batch_size"
  [[ -z "$e" ]] && e=0
  [[ -z "$t" ]] && t=0
  [[ -z "$s" ]] && s=0
  echo "${m},${p},${b},${e},${t},${s}" >> "$SUMMARY_CSV"
done

# BEST SELECTION ---------------------------------------------------------------
BEST_CSV="$OUTPUT_DIR/best.csv"
echo "model,precision,best_batch_size,elapsed_sec,throughput" > "$BEST_CSV"

awk -F, 'NR>1 {key=$1 FS $2; if ($6==1) {if ($5+0 > tp[key]+0) {tp[key]=$5; bs[key]=$3; tm[key]=$4; m[key]=$1; p[key]=$2}}}
END {for (k in tp) print m[k]","p[k]","bs[k]","tm[k]","tp[k]}' "$SUMMARY_CSV" >> "$BEST_CSV"

log "Summary CSV: $SUMMARY_CSV"
log "Best CSV:    $BEST_CSV"
column -t -s, "$BEST_CSV" 2>/dev/null || cat "$BEST_CSV"