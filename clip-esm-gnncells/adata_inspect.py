import scanpy as sc
import pandas as pd

adata = sc.read_h5ad("og_data_backup/dataset_1_single_gene_updated.h5ad")

print("AnnData OBS columns:", adata.obs.columns.tolist())
print("AnnData VAR columns:", adata.var.columns.tolist())
print("AnnData OBSM keys:", list(adata.obsm.keys()))
print("AnnData UNS keys:", list(adata.uns.keys()))

#df = pd.read_csv("og_data_backup/dataset_1_single_gene_updated_cellwise_data.csv")
#print("CSV columns:", df.columns.tolist())

esm_embeddings = adata.obsm['esm_embeddings']
print("esm_embeddings shape:", esm_embeddings.shape)
