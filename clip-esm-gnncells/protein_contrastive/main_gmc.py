# import torch
# from protein_contrastive.trainer_gmc_v0 import GMCTrainer
# from config import config
# import wandb
# import umap
# import umap
# import matplotlib.pyplot as plt
# import seaborn as sns
# import torch
# import torch
# import umap
# import wandb
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import torch.nn.functional as F

# def main():
#     results = torch.load('results/encoder_results.pt')
#     train_data = {
#         'cell_state': torch.cat([emb['cell'] for emb in results['train']['embeddings']]),
#         'pert_protein': torch.cat([emb['protein'] for emb in results['train']['embeddings']]),
#         'perturbation': torch.cat([emb['perturbation'] for emb in results['train']['embeddings']]),
#     }

#     val_data = {
#         'cell_state': torch.cat([emb['cell'] for emb in results['val']['embeddings']]),
#         'pert_protein': torch.cat([emb['protein'] for emb in results['val']['embeddings']]),
#         'perturbation': torch.cat([emb['perturbation'] for emb in results['val']['embeddings']]),
#     }
    
    
#     trainer = GMCTrainer(config)
#     trainer.setup_data(train_data, val_data)  # Ensure data is set up
#     analyze_embeddings(trainer)
#     trainer.train(train_data, val_data, config.epochs)
#     wandb.finish()


# def analyze_embeddings(trainer, save_path="umap_plot.png"):
#     if not hasattr(trainer, 'train_loader'):
#         raise AttributeError("[ERROR] `train_loader` is missing in trainer. Call `setup_data` first.")

#     trainer.similarity_transformer.eval()
#     train_embeddings = torch.cat([b[0].detach() for b in trainer.train_loader], dim=0).cpu().numpy()
#     protein_embeddings = torch.cat([b[1].detach() for b in trainer.train_loader], dim=0).cpu().numpy()

#     train_proj = trainer.proj_cell(torch.tensor(train_embeddings, device=trainer.device)).detach().cpu().numpy()
#     protein_proj = trainer.proj_protein(torch.tensor(protein_embeddings, device=trainer.device)).detach().cpu().numpy()

#     print(f"[DEBUG] Number of train embeddings: {train_embeddings.shape[0]}, Number of protein embeddings: {protein_embeddings.shape[0]}")

#     sim_cp = F.cosine_similarity(torch.tensor(train_embeddings), torch.tensor(protein_embeddings), dim=-1)
#     sim_cell = torch.mm(torch.tensor(train_embeddings), torch.tensor(train_embeddings).T)
#     sim_protein = torch.mm(torch.tensor(protein_embeddings), torch.tensor(protein_embeddings).T)
    
#     alignment = (sim_cp ** 2).mean().item()
#     uniformity = (torch.pdist(torch.tensor(train_embeddings), p=2).pow(2).mul(-2).exp().mean().item() + 
#                   torch.pdist(torch.tensor(protein_embeddings), p=2).pow(2).mul(-2).exp().mean().item()) / 2

#     print(f"[DEBUG] Cosine Similarity (Cell-Protein) - Mean: {sim_cp.mean().item():.6f}, Min: {sim_cp.min().item():.6f}, Max: {sim_cp.max().item():.6f}")
#     print(f"[DEBUG] Intra-Cell Similarity: Mean={sim_cell.mean().item():.6f}, Min={sim_cell.min().item():.6f}, Max={sim_cell.max().item():.6f}")
#     print(f"[DEBUG] Intra-Protein Similarity: Mean={sim_protein.mean().item():.6f}, Min={sim_protein.min().item():.6f}, Max={sim_protein.max().item():.6f}")
#     print(f"[DEBUG] Embedding Alignment: {alignment:.6f}")
#     print(f"[DEBUG] Embedding Uniformity: {uniformity:.6f}")

#     umap_raw = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine").fit_transform(train_embeddings)
#     umap_proj = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine").fit_transform(train_proj)
#     umap_protein = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine").fit_transform(protein_proj)

#     fig, ax = plt.subplots(1, 3, figsize=(18, 5))
#     sns.scatterplot(x=umap_raw[:, 0], y=umap_raw[:, 1], ax=ax[0], alpha=0.7)
#     ax[0].set_title("Raw Cell Embeddings")
#     sns.scatterplot(x=umap_proj[:, 0], y=umap_proj[:, 1], ax=ax[1], alpha=0.7)
#     ax[1].set_title("Projected Cell Embeddings")
#     sns.scatterplot(x=umap_protein[:, 0], y=umap_protein[:, 1], ax=ax[2], alpha=0.7)
#     ax[2].set_title("Projected Protein Embeddings")
#     plt.savefig(save_path)
#     print(f"[INFO] UMAP plot saved at {save_path}")

#     if wandb.run:
#         wandb.log({"UMAP Visualization": wandb.Image(save_path),
#                    "Cosine Similarity Mean": sim_cp.mean().item(),
#                    "Cosine Similarity Min": sim_cp.min().item(),
#                    "Cosine Similarity Max": sim_cp.max().item(),
#                    "Intra-Cell Similarity Mean": sim_cell.mean().item(),
#                    "Intra-Protein Similarity Mean": sim_protein.mean().item(),
#                    "Embedding Alignment": alignment,
#                    "Embedding Uniformity": uniformity})
#         print("[INFO] Metrics logged to WandB.")

#     plt.show()

# if __name__ == "__main__":
#     main()


import torch
from data_encoder_gmc import CellProteinDataset
from trainer_gmc import GMCTrainer
from config import config
from torch.utils.data import ConcatDataset

def main():
    train_ds = [
        CellProteinDataset(data_dir="v1_data_aligned", dataset_num=num)
        for num in [1, 3, 4, 5, 6, 7, 8]
    ]
    val_ds = [
        CellProteinDataset(data_dir="v1_data_aligned", dataset_num=num)
        for num in [2, 9]
    ]
    train_dataset = ConcatDataset(train_ds)
    val_dataset = ConcatDataset(val_ds)
    trainer = GMCTrainer(config, train_dataset, val_dataset)
    trainer.train(config.epochs)

if __name__ == "__main__":
    main()

