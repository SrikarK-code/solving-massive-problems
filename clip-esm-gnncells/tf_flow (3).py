import pickle
import os
import esm
import glob
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from data_encoder import DataEncoder

def cleanup():
    dist.destroy_process_group()

def process_dataset_ddp(encoder, rank, world_size, dataset_num, is_train=True):
    try:
        print(f"Rank {rank}: Starting dataset {dataset_num}", flush=True)
        adata, cell_data = encoder.module.load_dataset(dataset_num)
        print(f"Rank {rank}: Loaded dataset {dataset_num}, size: {len(adata)}", flush=True)
        total_cells = len(adata)
        indices = torch.arange(total_cells)
        sampler = DistributedSampler(indices, num_replicas=world_size, rank=rank)
        sampler.set_epoch(0)
        local_indices = list(iter(sampler))
        local_adata = adata[local_indices]
        local_cell_data = cell_data.iloc[local_indices]
        
        cell_emb = encoder.module.encode_cell_state(local_adata)
        protein_emb, pert_emb, esm_tokens, esm_embeddings, top5_tokens, top5_embeddings = \
            encoder.module.encode_protein_perturbation(local_cell_data, batch_size=8)
        
        gathered_results = [None for _ in range(world_size)]
        local_results = {
            'cell_emb': cell_emb,
            'protein_emb': protein_emb,
            'pert_emb': pert_emb,
            'esm_tokens': esm_tokens,
            'esm_embeddings': esm_embeddings,
            'top5_tokens': top5_tokens,
            'top5_embeddings': top5_embeddings,
            'indices': local_indices
        }
        print(f"Rank {rank}: Starting gather for dataset {dataset_num}")
        dist.all_gather_object(gathered_results, local_results)
        print(f"Rank {rank}: Completed gather for dataset {dataset_num}")
        
        if rank == 0:
            combined_results = {
                'adata': adata,
                'cell_data': cell_data,
                'embeddings': {}
            }
            
            for key in ['cell_emb', 'protein_emb', 'pert_emb', 'esm_tokens', 
                       'esm_embeddings', 'top5_tokens', 'top5_embeddings']:
                combined_results['embeddings'][key] = torch.zeros_like(
                    gathered_results[0][key].expand(total_cells, -1)
                )
            
            for proc_results in gathered_results:
                indices = proc_results['indices']
                for key in combined_results['embeddings']:
                    combined_results['embeddings'][key][indices] = proc_results[key]
            
            return combined_results
            
    except Exception as e:
        print(f"Error in rank {rank}: {str(e)}")
        raise
    # finally:
    #     cleanup()


def setup():
    # Debug SLURM environment variables
    print("MASTER_ADDR:", os.environ.get("MASTER_ADDR"))
    print("MASTER_PORT:", os.environ.get("MASTER_PORT"))
    print("SLURM_PROCID:", os.environ.get("SLURM_PROCID"))
    print("LOCAL_RANK:", os.environ.get("LOCAL_RANK"))
    print("SLURM_LOCALID:", os.environ.get("SLURM_LOCALID"))  # Debug LOCAL_RANK

    # Add NCCL debug info here
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
    os.environ["NCCL_SOCKET_TIMEOUT"] = "300"

    # Get rank and world size
    rank = int(os.environ.get("SLURM_PROCID", 0))  # Global rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Local rank per node
    world_size = int(os.environ.get("SLURM_NTASKS", 1))  # Total number of tasks

    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size,
    )

    # Set the current device for this process
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def main():
    rank, local_rank, world_size = setup()
    # print('setup done...')
    print(f'Rank {rank}: Setup complete, creating encoder...', flush=True)
    
    try:
        # Create and move encoder to the correct device
        encoder = DataEncoder().cuda(local_rank)
        print(f'Rank {rank}: Encoder created', flush=True)
        # Wrap with DDP
        encoder = DDP(encoder, device_ids=[local_rank])
        print(f'Rank {rank}: DDP wrapper created', flush=True)
        
        # Your existing main_ddp code here
        all_results = {'train': [], 'val': []}
        save_dir = "results"
        os.makedirs(save_dir, exist_ok=True)
        
        for dataset_num in [1,2,3,4,5,6,7,8,9]:
            checkpoint_dir = "checkpoints"
            if rank == 0:  # Only clean checkpoints on rank 0
                if os.path.exists(checkpoint_dir):
                    for f in glob.glob(f"{checkpoint_dir}/checkpoint_batch_*_dataset_{dataset_num}_*.pt"):
                        os.remove(f)
                        
            # Synchronize all processes before continuing
            torch.distributed.barrier()
                    
            is_train = dataset_num not in [2,9]
            try:
                adata, cell_data = encoder.module.load_dataset(dataset_num)
                total_cells = len(adata)
                
                # Process your data
                result = process_dataset_ddp(encoder, rank, world_size, dataset_num, is_train=True)

                if rank == 0:  # Only save on rank 0
                    dataset_path = os.path.join(save_dir, f"dataset_{dataset_num}_results.pt")
                    torch.save(result, dataset_path)
                    
                    if is_train:
                        all_results['train'].append(result)
                    else:
                        all_results['val'].append(result)
                        
            except Exception as e:
                print(f"Failed processing dataset {dataset_num} on rank {rank}: {e}")
                
            # Synchronize before next dataset
            torch.distributed.barrier()
            
        # Save final results on rank 0
        if rank == 0:
            if all_results['train'] or all_results['val']:
                torch.save(all_results, os.path.join(save_dir, "all_results.pt"))
    
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()



















