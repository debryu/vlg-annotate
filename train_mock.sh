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

$python -m train_cbm --dataset cifar10 --device cpu --num_workers 1 --cbl_batch_size 256 --concept_set concept_files/cifar10_filtered.txt --mock --annotation_dir annotations
