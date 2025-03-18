#!/bin/bash
#SBATCH --job-name=ddp_job
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=scavenger-gpu
#SBATCH --output=ddp_job_%j.out
#SBATCH --error=ddp_job_%j.err

# Export required environment variables
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

# Run the distributed script
srun --export=ALL -n4 --gres=gpu:1 python /hpc/group/chatterjee/srikar/storage/pbg/tong/tf-dplm-flow/TONG/tf_flow.py
