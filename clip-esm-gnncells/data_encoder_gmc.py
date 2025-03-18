import os
import torch
from torch.utils.data import Dataset
import scanpy as sc
import pandas as pd
from scipy.sparse import issparse

class CellProteinDataset(Dataset):
    def __init__(self, data_dir, dataset_num):
        h5ad_file = os.path.join(data_dir, f"aligned_dataset_{dataset_num}.h5ad")
        self.adata = sc.read_h5ad(h5ad_file)
        
        # Convert gene expression data
        X = self.adata.X
        if issparse(X):
            X = X.toarray()
        self.cells = torch.tensor(X, dtype=torch.float32)
        
        # Convert gene mask from sparse to dense tensor
        gene_mask = self.adata.layers['gene_mask']
        if issparse(gene_mask):
            gene_mask = gene_mask.toarray()
        self.masks = torch.tensor(gene_mask, dtype=torch.bool)
        
        # Convert other modalities
        self.diffmap = torch.tensor(self.adata.obsm['X_diffmap'], dtype=torch.float32)
        self.proteins = torch.tensor(self.adata.obsm['esm_embeddings'], dtype=torch.float32)
        
        # Convert labels
        labels = pd.Categorical(self.adata.obs['leiden'])
        self.labels = torch.tensor(labels.codes, dtype=torch.long)

    def __len__(self):
        return self.cells.size(0)

    def __getitem__(self, idx):
        return (
            self.cells[idx],        # Gene expression [num_genes]
            self.masks[idx],        # Gene mask [num_genes]
            self.diffmap[idx],       # Diffusion pseudotime features
            self.proteins[idx],      # Protein embeddings
            self.labels[idx]         # Cell type labels
        )


