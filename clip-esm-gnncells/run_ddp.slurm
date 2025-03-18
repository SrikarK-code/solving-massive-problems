#!/bin/bash
#SBATCH --job-name=ddp_job
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --partition=scavenger-gpu
#SBATCH --output=ddp_tf-flow_data/ddp_job_%j.out
#SBATCH --error=ddp_tf-flow_data/ddp_job_%j.err
#SBATCH --export=ALL

# Create the directory if it doesn't exist
mkdir -p ddp_tf-flow_data
# Identify the master node's address
master_addr=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)

# Debugging info
echo "Master node: $master_addr"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_PROCID: $SLURM_PROCID"

# Set MASTER_ADDR and MASTER_PORT
export MASTER_ADDR=$master_addr
export MASTER_PORT=29500

# Set LOCAL_RANK using SLURM_LOCALID
export LOCAL_RANK=$SLURM_LOCALID
echo "SLURM_LOCALID: $SLURM_LOCALID"

# Set PYTHONPATH for dependencies
export PYTHONPATH=$PYTHONPATH:/hpc/group/chatterjee/srikar/storage/pbg/tong/tf-dplm-flow/TONG/conditional_flow_matching
export PYTHONPATH=$PYTHONPATH:/hpc/group/chatterjee/srikar/storage/pbg/tong/tf-dplm-flow/TONG/

# Run your script with srun
srun --export=ALL -n4 python /hpc/group/chatterjee/srikar/storage/pbg/tong/tf-dplm-flow/TONG/tf_flow.py
