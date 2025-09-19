#!/bin/bash
#SBATCH --job-name=mixed_precision_nvidia
#SBATCH --partition=tupi
#SBATCH --nodelist=tupi2
#SBATCH --exclude=tupi3,tupi4,tupi5,tupi6
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

cp -r -f $HOME/tcc_mixed_precision/src $SCRATCH/tcc_mixed_precision

source $SCRATCH/tcc_mixed_precision/envs/nvidia/bin/activate

python3 $SCRATCH/tcc_mixed_precision/src/main.py --gpu-vendor nvidia --model gpt2 --precision fp16 --batch-size 32 --epochs 100 --output-file gpt2_fp16_32batch_100epoch_4090.csv

cp $SCRATCH/gpt2_fp16_32batch_100epoch_4090.csv $HOME/tcc_mixed_precision/data/batch/gpt2_fp16_32batch_100epoch_4090.csv