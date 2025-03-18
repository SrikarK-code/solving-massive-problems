#!/bin/bash

#SBATCH --job-name=tf_flow
#SBATCH --output=flow_train_logs/%x_%j.out
#SBATCH --error=flow_train_logs/%x_%j.err
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --partition=scavenger-gpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Create logs directory if it doesn't exist
mkdir -p flow_train_logs
mkdir -p checkpoints

# Set WandB API key
export WANDB_API_KEY="8a1f77597d211a40626fad41f5281ea51614bcd0"

# Run training
python train_classifier_flow.py

# Copy checkpoints to separate directory
mkdir -p flow_checkpoints
cp -r checkpoints/* flow_checkpoints/

