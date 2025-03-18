#!/bin/bash
#SBATCH --job-name=esm_tf_data
#SBATCH --output=esm_tf_%j.out
#SBATCH --error=esm_tf_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=208G
#SBATCH --gres=gpu:RTX6000:2
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-common



# Run the script
python tf_flow.py
