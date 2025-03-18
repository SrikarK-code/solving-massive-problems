import torch
from umap import UMAP
import wandb
import torch.nn.functional as F


def visualize_embeddings(loader, model, device):
    embeddings, labels = [], []
    with torch.no_grad():
        for batch in loader:
            emb = F.normalize(batch[0].to(device), dim=1)
            embeddings.append(emb.cpu())
            labels.append(batch[1].cpu())

    embeddings, labels = torch.cat(embeddings), torch.cat(labels)
    umap_emb = UMAP(n_components=2).fit_transform(embeddings.numpy())
    
    wandb.log({"embeddings": wandb.Table(data=umap_emb, columns=["x", "y"])})
