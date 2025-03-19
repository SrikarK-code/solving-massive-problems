import scanpy as sc
import scgpt as scg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import anndata as ad
import mygene

# Parameters
N_HVG = 4000  # Number of highly variable genes
BATCH_SIZE = 64
DATA_PATHS = [f"/path/to/plate_{i}.h5ad" for i in range(1, 15)]  # 14 plates
OUTPUT_PATH = "/path/to/processed_data.h5ad"

# Step 1: Load and Combine All 14 Plates
adata_list = [sc.read_h5ad(path) for path in DATA_PATHS]
adata = ad.concat(adata_list, join="outer", label="plate", keys=[f"plate_{i}" for i in range(1, 15)])
print(f"Combined AnnData: {adata.shape}")

# Step 2: Convert ENSG to Gene Names using mygene
# Assume genes are in adata.var.index as ENSG IDs (e.g., 'ENSG00000141510')
mg = mygene.MyGeneInfo()
ensg_ids = adata.var.index.tolist()  # 62,710 genes
gene_info = mg.querymany(ensg_ids, scopes="ensembl.gene", fields="symbol", species="human")

# Map ENSG to gene symbols
ensg_to_symbol = {}
for hit in gene_info:
    if "symbol" in hit:
        ensg_to_symbol[hit["query"]] = hit["symbol"]

# Add gene_col to adata.var
adata.var["gene_col"] = adata.var.index.map(lambda x: ensg_to_symbol.get(x, None))
print(f"Genes with symbols: {adata.var['gene_col'].notnull().sum()}/{len(adata.var)}")

# Filter out genes without symbols (~2% expected)
adata = adata[:, adata.var["gene_col"].notnull()]
print(f"After filtering unmapped genes: {adata.shape}")

# Step 3: Basic Preprocessing
# Ensure raw counts are in adata.X (scGPT expects raw counts)
if "counts" in adata.layers:
    adata.X = adata.layers["counts"]
else:
    print("Assuming adata.X contains raw counts")

# Filter low-quality cells (Tahoe-100M full filters: n_genes >= 250, mt% < 20, total_counts >= 700)
sc.pp.calculate_qc_metrics(adata, inplace=True)
adata = adata[
    (adata.obs["n_genes_by_counts"] >= 250) &
    (adata.obs["pct_counts_mt"] < 20) &
    (adata.obs["total_counts"] >= 700), :
]
print(f"After QC filtering: {adata.shape}")

# Identify highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat_v3")
adata = adata[:, adata.var["highly_variable"]]
print(f"After HVG filtering: {adata.shape}")

# Step 4: Generate scGPT Embeddings
# Use 'gene_col' for scGPT, pretrained model pulled automatically
embed_adata = scg.tasks.embed_data(
    adata,
    gene_col="gene_col",
    batch_size=BATCH_SIZE,
)
print(f"Embeddings generated: {embed_adata.obsm['X_scGPT'].shape}")

# Step 5: Compute Per-Cell-Line Difference Embeddings
# Assume metadata columns: 'cell_line', 'drug', 'treatment' (DMSO or drug)
cell_lines = embed_adata.obs["cell_line"].unique()  # 47 cell lines
drugs = embed_adata.obs["drug"].unique()  # 379 drugs
difference_embeddings = []
condition_labels = []

for cell_line in cell_lines:
    # Extract DMSO control for this cell line
    dmso_mask = (embed_adata.obs["cell_line"] == cell_line) & (embed_adata.obs["treatment"] == "DMSO")
    dmso_cells = embed_adata[dmso_mask]
    if dmso_cells.n_obs == 0:
        print(f"Warning: No DMSO cells for {cell_line}, skipping")
        continue
    dmso_mean = dmso_cells.obsm["X_scGPT"].mean(axis=0)  # Mean embedding for DMSO

    # Compute difference for each drug in this cell line
    for drug in drugs:
        drug_mask = (embed_adata.obs["cell_line"] == cell_line) & (embed_adata.obs["drug"] == drug)
        drug_cells = embed_adata[drug_mask]
        if drug_cells.n_obs == 0:
            continue
        drug_mean = drug_cells.obsm["X_scGPT"].mean(axis=0)  # Mean embedding for drug
        diff_embedding = drug_mean - dmso_mean
        difference_embeddings.append(diff_embedding)
        condition_labels.append(f"{cell_line}_{drug}")

difference_embeddings = np.stack(difference_embeddings)
print(f"Difference embeddings: {difference_embeddings.shape}")  # Expected: (17,813, embedding_dim)

# Step 6: Split Data by Drugs
condition_df = pd.DataFrame({
    "condition": condition_labels,
    "embedding": list(difference_embeddings),
    "drug": [label.split("_")[1] for label in condition_labels]
})

unique_drugs = condition_df["drug"].unique()  # 379 drugs
train_drugs, test_drugs = train_test_split(unique_drugs, test_size=0.2, random_state=42)  # 303 train, 76 test

train_mask = condition_df["drug"].isin(train_drugs)
test_mask = condition_df["drug"].isin(test_drugs)

train_data = condition_df[train_mask]
test_data = condition_df[test_mask]
print(f"Train conditions: {len(train_data)}, Test conditions: {len(test_data)}")

# Step 7: Save Processed Data
processed_adata = ad.AnnData(
    X=np.vstack(condition_df["embedding"].values),
    obs=condition_df[["condition", "drug"]]
)
processed_adata.write(OUTPUT_PATH)
print(f"Saved processed data to {OUTPUT_PATH}")

# Step 8: Prepare for Your Model
train_embeddings = np.vstack(train_data["embedding"].values)
test_embeddings = np.vstack(test_data["embedding"].values)
train_drugs = train_data["drug"].values
test_drugs = test_data["drug"].values

print("Data ready for model training!")
