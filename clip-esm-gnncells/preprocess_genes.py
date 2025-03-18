import os
import numpy as np
import scipy.sparse as sp
import scanpy as sc
from anndata import concat as ad_concat

def align_datasets(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    adatas = []
    for num in range(1, 10):
        h5ad_file = os.path.join(data_dir, f"dataset_{num}_single_gene_updated.h5ad")
        if not os.path.exists(h5ad_file):
            continue
            
        adata = sc.read_h5ad(h5ad_file)
        adata.var_names_make_unique()
        
        adata.var['original_gene'] = True
        adatas.append(adata)

    combined = ad_concat(adatas, join='outer', fill_value=0, index_unique=None)
    
    for i, orig in enumerate(adatas):
        dataset_key = f'dataset_{i+1}_mask'
        combined.var[dataset_key] = combined.var_names.isin(orig.var_names)
        
    combined.var.index = combined.var.index.astype(str)
    combined.var = combined.var.apply(lambda x: x.astype(str) if x.name != 'original_gene' else x)
    
    combined.obs = combined.obs.apply(lambda x: x.astype(str) if x.dtype != 'bool' else x)

    start_idx = 0
    for i, orig in enumerate(adatas):
        end_idx = start_idx + orig.n_obs
        aligned_adata = combined[start_idx:end_idx].copy()
        
        mask_matrix = sp.csr_matrix((aligned_adata.n_obs, aligned_adata.n_vars), dtype=int)
        mask_column_indices = np.where(aligned_adata.var[f'dataset_{i+1}_mask'].values)[0]
        
        for row_idx in range(aligned_adata.n_obs):
            mask_matrix[row_idx, mask_column_indices] = 1
            
        aligned_adata.layers['gene_mask'] = mask_matrix
        
        aligned_adata.obs['has_gene_mask'] = True
        
        for col in aligned_adata.obs.columns:
            if col != 'has_gene_mask':
                aligned_adata.obs[col] = aligned_adata.obs[col].astype(str)
        
        aligned_file = os.path.join(output_dir, f"aligned_dataset_{i+1}.h5ad")
        aligned_adata.write(
            aligned_file,
            as_dense=['X']
        )
        print(f"Successfully wrote {aligned_file}")
        start_idx = end_idx

if __name__ == "__main__":
    align_datasets(
        data_dir="v1_data",
        output_dir="v1_data_aligned"
    )
