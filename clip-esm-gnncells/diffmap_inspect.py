import scanpy as sc
import numpy as np

adata = sc.read_h5ad("og_data_backup/dataset_1_single_gene_updated.h5ad")

if 'X_diffmap' in adata.obsm:
    diffmap = adata.obsm['X_diffmap']
    print("Diffmap shape:", diffmap.shape)
    print("Diffmap min:", np.min(diffmap), "max:", np.max(diffmap), "mean:", np.mean(diffmap))
else:
    print("No diffmap found in the AnnData object.")
