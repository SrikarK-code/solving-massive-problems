#!/bin/bash

#SBATCH --job-name=gmc_train
#SBATCH --output=protein_contrastive/logs/%x_%j.out
#SBATCH --error=protein_contrastive/logs/%x_%j.err
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --partition=scavenger-gpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

mkdir -p protein_contrastive/logs
mkdir -p protein_contrastive/gmc_checkpoints

export WANDB_API_KEY="8a1f77597d211a40626fad41f5281ea51614bcd0"
export PYTHONPATH=$PYTHONPATH:/hpc/group/chatterjee/srikar/storage/pbg/tong/tf-dplm-flow/TONG/protein_contrastive

python protein_contrastive/main_gmc.py

mkdir -p protein_contrastive/gmc_checkpoints
cp -r protein_contrastive/checkpoints/* protein_contrastive/gmc_checkpoints/
