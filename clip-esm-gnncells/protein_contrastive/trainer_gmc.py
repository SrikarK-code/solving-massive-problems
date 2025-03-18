import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from losses_gmc import gmc_loss
from cell_encoder_gmc import CellEncoderGMC
from protein_encoder_gmc import ProteinEncoderGMC
from model import SimilarityTransformer  # Assumes this module exists

class GMCTrainer:
    def __init__(self, config, train_dataset, val_dataset):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.queue_size = config.queue_size
        self.latent_dim = config.latent_dim
        
        self.cell_encoder = CellEncoderGMC(config).to(self.device)
        self.protein_encoder = ProteinEncoderGMC(config).to(self.device)
        self.similarity_transformer = SimilarityTransformer(config.latent_dim).to(self.device)
        
        self.proj_cell = self.create_projection_layer(config.latent_dim, config.shared_dim).to(self.device)
        self.proj_protein = self.create_projection_layer(config.latent_dim, config.shared_dim).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            list(self.cell_encoder.parameters()) +
            list(self.similarity_transformer.parameters()) +
            list(self.proj_cell.parameters()) +
            list(self.proj_protein.parameters()),
            lr=3e-4,  # Changed to recommended learning rate
            weight_decay=0.01  # Added regularization
        )
            
        self.temperature = nn.Parameter(torch.tensor(0.5, device=self.device))  # Start higher: 0.5 instead of 0.07
        self.temperature_min = 0.1  # Prevent collapse
        
        # Initialize dynamic memory queue
        self.queue = torch.zeros(self.queue_size, config.shared_dim).to(self.device)
        self.dataset_queue = torch.zeros(self.queue_size, dtype=torch.long).to(self.device)
        self.queue_ptr = 0
        self.queue_full = False
        
        self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)


    def create_projection_layer(self, d_in, d_out, dropout=0.1, use_layernorm=True):
        class EnhancedProjectionLayer(nn.Module):
            def __init__(self, d_in, d_out, dropout, use_layernorm):
                super().__init__()
                self.pre_norm = nn.LayerNorm(d_in)  # NEW: Input normalization
                self.linear1 = nn.Linear(d_in, d_out)
                self.ln1 = nn.LayerNorm(d_out) if use_layernorm else nn.Identity()
                self.act = nn.GELU()
                self.dropout = nn.Dropout(dropout)
                self.linear2 = nn.Linear(d_out, d_out)
                self.ln2 = nn.LayerNorm(d_out) if use_layernorm else nn.Identity()
                self.linear3 = nn.Linear(d_out, d_out)
                self.ln3 = nn.LayerNorm(d_out) if use_layernorm else nn.Identity()
                self.gate = nn.Sequential(
                    nn.Linear(d_in, d_out),
                    nn.Sigmoid()
                )
                self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
            def forward(self, x):
                x = self.pre_norm(x)  # NEW: Normalize before projection
                identity = self.skip(x)
                gate_val = self.gate(x)
                out = self.linear1(x)
                out = self.ln1(out)
                out = self.act(out)
                out = self.dropout(out)
                out = self.linear2(out)
                out = self.ln2(out)
                out = self.act(out)
                out = self.dropout(out)
                out = self.linear3(out)
                out = self.ln3(out)
                return gate_val * out + (1 - gate_val) * identity
        return EnhancedProjectionLayer(d_in, d_out, dropout, use_layernorm)

    def compute_edge_index(self, dpt):
        batch_size = dpt.size(0)
        print("compute_edge_index: batch_size =", batch_size)
        dist = torch.cdist(dpt, dpt, p=2)
        print("compute_edge_index: dist shape =", dist.shape)
        
        conn = torch.exp(-dist)
        threshold = 0.3  # hyperparameter for connectivity threshold
        conn_mask = conn > threshold
        conn_mask.fill_diagonal_(False)
        source_conn, target_conn = conn_mask.nonzero(as_tuple=True)
        
        diag = torch.eye(batch_size, device=dpt.device) * 1e6
        knn_dist = dist + diag
        print("compute_edge_index: knn_dist shape =", knn_dist.shape)
        
        # Ensure k is no larger than batch_size.
        k_val = self.config.n_neighbors if self.config.n_neighbors <= batch_size else batch_size
        if k_val < self.config.n_neighbors:
            print(f"Warning: reducing n_neighbors from {self.config.n_neighbors} to {k_val} because batch_size is small.")
            
        knn_indices = torch.topk(knn_dist, k=k_val, largest=False).indices
        print("compute_edge_index: knn_indices shape =", knn_indices.shape)
        
        source_knn = torch.arange(batch_size, device=dpt.device).unsqueeze(1).repeat(1, k_val).reshape(-1)
        target_knn = knn_indices.reshape(-1)
        
        source_all = torch.cat([source_conn, source_knn], dim=0)
        target_all = torch.cat([target_conn, target_knn], dim=0)
        edge_index = torch.stack([source_all, target_all], dim=0)
        edge_index = torch.unique(edge_index, dim=1)
        
        return edge_index



    @torch.no_grad()
    def update_queue(self, protein_embeddings, dataset_labels):
        batch_size = protein_embeddings.size(0)
        protein_embeddings = F.normalize(protein_embeddings, dim=1)
        start_idx = self.queue_ptr
        end_idx = (self.queue_ptr + batch_size) % self.queue_size
        if start_idx + batch_size <= self.queue_size:
            self.queue[start_idx:start_idx + batch_size] = protein_embeddings
            self.dataset_queue[start_idx:start_idx + batch_size] = dataset_labels
        else:
            first_part = self.queue_size - start_idx
            self.queue[start_idx:] = protein_embeddings[:first_part]
            self.queue[:end_idx] = protein_embeddings[first_part:]
            self.dataset_queue[start_idx:] = dataset_labels[:first_part]
            self.dataset_queue[:end_idx] = dataset_labels[first_part:]
        self.queue_ptr = end_idx
        if self.queue_ptr < start_idx:
            self.queue_full = True

    def train_step(self, batch):
        for key, value in batch.items():
            print(f"{key}: {type(value)}")
        # gene_expr, mask, dpt, protein_raw, labels = [b.to(self.device) for b in batch]
        gene_expr = batch["cells"].to(self.device)
        mask = batch["masks"].to(self.device)
        dpt = batch["diffmap"].to(self.device)
        protein_raw = batch["proteins"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        batch_size = gene_expr.size(0)
        edge_index = self.compute_edge_index(dpt)
        batch_idx = torch.arange(batch_size, device=self.device)
        
        cell_emb = self.cell_encoder(gene_expr, dpt, edge_index, batch_idx, mask)
        protein_emb = self.protein_encoder(protein_raw)
        
        cell_proj = F.normalize(cell_emb, p=2, dim=-1)  # Enforce L2 norm
        protein_proj = F.normalize(protein_emb, p=2, dim=-1)  # Enforce L2 norm

        print(f"[DEBUG] Cell emb scale: mean={cell_emb.mean().item():.3f}, std={cell_emb.std().item():.3f}")
        print(f"[DEBUG] Protein emb scale: mean={protein_emb.mean().item():.3f}, std={protein_emb.std().item():.3f}")
                
        # Process cross-modal interactions
        cell_trans = self.similarity_transformer(cell_proj, protein_proj)  # x=cell, y=protein
        protein_trans = self.similarity_transformer(protein_proj, cell_proj)  # x=protein, y=cell
                
        loss = gmc_loss(cell_trans, protein_trans, self.queue if self.queue_full else None,
                        self.dataset_queue, labels, self.temperature)
                        
        self.optimizer.zero_grad()
        loss.backward()
        
        
        
        # Add gradient clipping before optimizer step
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        
        # Add gradient monitoring after clipping
        cell_grad = torch.norm(torch.cat([p.grad.flatten() for p in self.cell_encoder.parameters() if p.grad is not None]))
        prot_grad = torch.norm(torch.cat([p.grad.flatten() for p in self.protein_encoder.parameters() if p.grad is not None]))
        print(f"Gradients - Cell: {cell_grad:.2e}, Protein: {prot_grad:.2e}")
        
        # Add temperature monitoring
        print(f"Current temperature: {self.temperature.item():.3f}")
        
        # Add similarity monitoring
        with torch.no_grad():
            cp_sim = F.cosine_similarity(cell_trans, protein_trans)
            print(f"CP Similarity - Mean: {cp_sim.mean():.3f} ± {cp_sim.std():.3f}")

        self.optimizer.step()
        
        self.update_queue(protein_trans.detach(), labels)
        return loss.item()

    def update_temperature(self):
        # Dynamic temperature scheduling: reduce temperature over time, but not below a minimum threshold.
        if hasattr(self.config, 'temperature_decay') and hasattr(self.config, 'temperature_min'):
            new_temp = self.temperature.data * self.config.temperature_decay
            self.temperature.data = torch.max(new_temp, torch.tensor(self.config.temperature_min, device=self.device))
    
    def train_epoch(self):
        self.cell_encoder.train()
        self.similarity_transformer.train()
        total_loss = 0
        for batch in self.train_loader:
            loss = self.train_step(batch)
            total_loss += loss
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.cell_encoder.eval()
        self.similarity_transformer.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                for key, value in batch.items():
                    print(f"{key}: {type(value)}")

                # gene_expr, mask, dpt, protein_raw, labels = [b.to(self.device) for b in batch]
                gene_expr = batch["cells"].to(self.device)
                mask = batch["masks"].to(self.device)
                dpt = batch["diffmap"].to(self.device)
                protein_raw = batch["proteins"].to(self.device)
                labels = batch["labels"].to(self.device)
                        
                batch_size = gene_expr.size(0)
                edge_index = self.compute_edge_index(dpt)
                batch_idx = torch.arange(batch_size, device=self.device)
                cell_emb = self.cell_encoder(gene_expr, dpt, edge_index, batch_idx, mask)
                protein_emb = self.protein_encoder(protein_raw)
                cell_proj = F.normalize(cell_emb, p=2, dim=-1)  # Enforce L2 norm
                protein_proj = F.normalize(protein_emb, p=2, dim=-1)  # Enforce L2 norm
                # Process cross-modal interactions
                cell_trans = self.similarity_transformer(cell_proj, protein_proj)  # x=cell, y=protein
                protein_trans = self.similarity_transformer(protein_proj, cell_proj)  # x=protein, y=cell
                loss = gmc_loss(cell_trans, protein_trans, self.queue if self.queue_full else None,
                                self.dataset_queue, labels, self.temperature)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            # Update temperature dynamically after each epoch
            self.update_temperature()

def collate_fn(batch):
    """Handles sparse to dense conversion and type enforcement"""
    cells, masks, diffmaps, proteins, labels = zip(*batch)
    
    # Convert all to tensors first
    cells_tensor = torch.stack([t.clone().detach() for t in cells])
    masks_tensor = torch.stack([t.clone().detach() for t in masks])
    diffmap_tensor = torch.stack([t.clone().detach() for t in diffmaps])
    proteins_tensor = torch.stack([t.clone().detach() for t in proteins])
    labels_tensor = torch.stack([t.clone().detach() for t in labels])

    # Ensure correct types
    return {
        'cells': cells_tensor.float(),
        'masks': masks_tensor.bool(),
        'diffmap': diffmap_tensor.float(),
        'proteins': proteins_tensor.float(),
        'labels': labels_tensor.long()
    }