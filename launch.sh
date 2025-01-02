#!/bin/bash
#SBATCH -p edu-20h
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=20:00:00   # Set time limit to 30 minutes
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --nodes 1
#SBATCH --job-name=vlg-annotate
#SBATCH --output=logs/out.out
#SBATCH --error=logs/err.err
#SBATCH -N 1

python=/home/nicola.debole/anaconda3/envs/vlg-cbm/bin/python

$python -m train_cbm.py --config configs/celeba.json --annotation_dir annotations --num_workers 4 --cbl_batch_size 512 --cbl_confidence_threshold 0.3
$python -m train_cbm.py --config configs/celeba.json --annotation_dir annotations --num_workers 4 --cbl_batch_size 512 --cbl_confidence_threshold 0.45
$python -m train_cbm.py --config configs/celeba.json --annotation_dir annotations --num_workers 4 --cbl_batch_size 512 --cbl_confidence_threshold 0.6
$python -m train_cbm.py --config configs/celeba.json --annotation_dir annotations --num_workers 4 --cbl_batch_size 512 --cbl_confidence_threshold 0.75
$python -m train_cbm.py --config configs/celeba.json --annotation_dir annotations --num_workers 4 --cbl_batch_size 512 --cbl_confidence_threshold 0.9