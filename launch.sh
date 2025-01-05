#!/bin/bash
#SBATCH -p edu-20h
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=20:00:00   # Set time limit to 30 minutes
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=40G
#SBATCH --nodes 1
#SBATCH --job-name=vlg-060
#SBATCH --output=logs/out060.out
#SBATCH --error=logs/err060.err
#SBATCH -N 1

python=/home/nicola.debole/anaconda3/envs/vlg-cbm/bin/python
$python -m train_cbm --config configs/celeba.json --annotation_dir annotations --num_workers 2 --cbl_batch_size 64 --cbl_confidence_threshold 0.6 --wandb
