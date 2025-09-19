#!/bin/bash
#SBATCH --job-name=mixed_precision_amd
#SBATCH --partition=lunaris
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

cp -r -f $HOME/tcc_mixed_precision/src $SCRATCH/tcc_mixed_precision

cd $SCRATCH
cd tcc_mixed_precision/
newgrp render <<'EOF'
source  envs/amd/bin/activate
python3 src/main.py --gpu-vendor amd --model gpt2 --precision fp16 --batch-size 32 --epochs 100 --output-file gpt2_fp16_32batch_100epoch_7900.csv
EOF

# pip install -r $SCRATCH/tcc_mixed_precision/src/requirements.txt

cp gpt2_fp16_32batch_100epoch_7900.csv $HOME/tcc_mixed_precision/data/batch/gpt2_fp16_32batch_100epoch_7900.csv
cp gpt2_fp16_32batch_100epoch_7900_raw_metrics.csv $HOME/tcc_mixed_precision/data/batch/gpt2_fp16_32batch_100epoch_7900_raw_metrics.csv