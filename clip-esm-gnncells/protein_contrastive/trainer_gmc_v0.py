import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from losses_gmc import gmc_loss
from model import SimilarityTransformer
import os
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import torch

class GMCTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.queue_size = config.queue_size
        self.latent_dim = config.latent_dim
        self.similarity_transformer = SimilarityTransformer(config.latent_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.similarity_transformer.parameters(), lr=self.config.learning_rate)
        self.temperature = nn.Parameter(torch.tensor(0.07, device=self.device))
        self.proj_cell = self.create_projection_layer(config.latent_dim, config.shared_dim).to(self.device)
        self.proj_protein = self.create_projection_layer(config.latent_dim, config.shared_dim).to(self.device)

        wandb.init(project="protein-contrastive-gmc", config=vars(config))
        self.setup_directories()
        self.setup_queue()

    def setup_directories(self):
        os.makedirs('checkpoints_gmc', exist_ok=True)

    def create_projection_layer(self, d_in, d_out, dropout=0.1, use_layernorm=True):
        """Creates a projection layer with optional layernorm."""
        return nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
            nn.LayerNorm(d_out) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
            nn.LayerNorm(d_out) if use_layernorm else nn.Identity()
        )


    def setup_queue(self):
        self.queue = torch.zeros(self.queue_size, self.latent_dim).to(self.device)
        self.dataset_queue = torch.zeros(self.queue_size, dtype=torch.long).to(self.device)
        self.queue_ptr = 0
        self.queue_full = False
        self.queue = F.normalize(self.queue, dim=1)


    def setup_data(self, train_data, val_data):
        train_tensors = [train_data[k] for k in ['cell_state', 'pert_protein', 'perturbation']]
        val_tensors = [val_data[k] for k in ['cell_state', 'pert_protein', 'perturbation']]
        train_dataset = TensorDataset(*train_tensors)
        val_dataset = TensorDataset(*val_tensors)
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=True)

    @torch.no_grad()
    def update_queue(self, embeddings, dataset_labels):
        batch_size = embeddings.shape[0]
        # embeddings = F.normalize(embeddings, dim=1)
        embeddings = F.normalize(self.proj_protein(embeddings), dim=1) 
        start_idx = self.queue_ptr
        end_idx = (self.queue_ptr + batch_size) % self.queue_size
        if start_idx + batch_size <= self.queue_size:
            self.queue[start_idx:start_idx + batch_size] = embeddings
            self.dataset_queue[start_idx:start_idx + batch_size] = dataset_labels
        else:
            first_part = self.queue_size - start_idx
            self.queue[start_idx:] = embeddings[:first_part]
            self.queue[:end_idx] = embeddings[first_part:]
            self.dataset_queue[start_idx:] = dataset_labels[:first_part]
            self.dataset_queue[:end_idx] = dataset_labels[first_part:]
        self.queue_ptr = end_idx
        if not self.queue_full and self.queue_ptr < start_idx:
            self.queue_full = True

    def train_step(self, batch_data, is_training=True):
        batch_data = [b.to(self.device) for b in batch_data]
        cell_state, protein, perturbation = batch_data
        # cell_state, protein = F.normalize(cell_state, dim=-1), F.normalize(protein, dim=-1)
        # print(f"[DEBUG] protein.device: {protein.device}, proj_protein.device: {next(self.proj_protein.parameters()).device}")
        # protein_proj = self.proj_protein(F.normalize(protein, dim=-1)).to(self.device)
        # cell_proj = self.proj_cell(F.normalize(cell_state, dim=-1)).to(self.device)
        protein_proj = F.normalize(self.proj_protein(protein), dim=-1).to(self.device)
        cell_proj = F.normalize(self.proj_cell(cell_state), dim=-1).to(self.device)
        total_loss = gmc_loss(cell_proj, protein_proj, self.queue if self.queue_full else None, self.dataset_queue, perturbation[:, 0], self.temperature)
        if is_training:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.update_queue(protein.detach(), perturbation[:, 0].detach())
        return total_loss.item()

    def train_epoch(self, loader, is_training=True):
        self.similarity_transformer.train() if is_training else self.similarity_transformer.eval()
        total_loss = 0
        with torch.set_grad_enabled(is_training):
            for batch_data in tqdm(loader):
                # total_loss += self.train_step(batch_data, is_training)
                step_loss = self.train_step(batch_data, is_training)
                wandb.log({'step_loss': step_loss})
                total_loss += step_loss

        return total_loss / len(loader)

    def train(self, train_data, val_data, n_epochs):
        self.setup_data(train_data, val_data)
        best_val_loss = float('inf')
        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch(self.train_loader, is_training=True)
            val_loss = self.train_epoch(self.val_loader, is_training=False)
            wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # torch.save(self.similarity_transformer.state_dict(), "checkpoints_gmc/best_model.pt")
                torch.save({
                    'similarity_transformer': self.similarity_transformer.state_dict(),
                    'proj_cell': self.proj_cell.state_dict(),
                     'proj_protein': self.proj_protein.state_dict()
                }, "checkpoints_gmc/best_model.pt")

        print("Training complete.")





