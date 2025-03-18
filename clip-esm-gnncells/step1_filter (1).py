# step1_filter.py
import pandas as pd
import scanpy as sc
import os
import shutil
from typing import Tuple
import numpy as np

def filter_dataset(dataset_num: int, subset_pct: float) -> Tuple[pd.DataFrame, sc.AnnData]:
   # Load data
   csv_file = f"og_data/dataset_{dataset_num}_single_gene_updated_cellwise_data.csv"
   h5ad_file = f"og_data/dataset_{dataset_num}_single_gene_updated.h5ad"
   
   if subset_pct == 0:  # For dataset 6
       return None, None
       
   # Read files
   cell_data = pd.read_csv(csv_file)
   adata = sc.read_h5ad(h5ad_file)
   
   # Calculate number of rows to keep
   n_rows = int(len(cell_data) * subset_pct/100)
   
   # Filter both dataframes using same indices to maintain correspondence
   indices = np.random.choice(len(cell_data), n_rows, replace=False)
   cell_data_subset = cell_data.iloc[indices]
   adata_subset = adata[indices]
   
   return cell_data_subset, adata_subset

def main():   
   # Dictionary of dataset numbers and their subset percentages
   dataset_subsets = {
#       1: 8.0,
#       2: 10.0
#       3: 2.5,
#       4: 10.0,
#       5: 10.0,
#       6: 10.0,  # Skip this one
#       7: 2.5,
       8: 2.5
#       9: 8.0
   }
   
   for dataset_num, subset_pct in dataset_subsets.items():
       print(f"\nProcessing dataset {dataset_num}")
       
       cell_data_subset, adata_subset = filter_dataset(dataset_num, subset_pct)
       
       if cell_data_subset is not None:
           # Save filtered data
           cell_data_subset.to_csv(f"v1_data/dataset_{dataset_num}_single_gene_updated_cellwise_data.csv", index=False)
           adata_subset.write_h5ad(f"v1_data/dataset_{dataset_num}_single_gene_updated.h5ad")
           
           print(f"Original rows: {len(pd.read_csv(f'og_data/dataset_{dataset_num}_single_gene_updated_cellwise_data.csv'))}")
           print(f"Filtered rows: {len(cell_data_subset)}")
       else:
           print(f"Skipping dataset {dataset_num}")

if __name__ == "__main__":
   main()
