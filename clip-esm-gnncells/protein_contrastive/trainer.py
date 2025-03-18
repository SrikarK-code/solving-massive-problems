import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from tqdm import tqdm
from losses import barlow_twins_loss, vicreg_loss, wasserstein_loss, hyperbolic_loss, nt_xent_loss
from model import SimilarityTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

class ContrastiveTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"[INIT] Using device: {self.device}")
        print(f"[INIT] Queue size: {config.queue_size}, Latent dim: {config.latent_dim}")

        self.queue_size = config.queue_size  
        self.latent_dim = config.latent_dim 

        # Use a momentum encoder for queue embeddings
        self.setup_momentum_encoder()

        self.setup_directories()
        self.setup_queue()
        # self.setup_data()
        self.similarity_transformer = SimilarityTransformer(config.latent_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.similarity_transformer.parameters(), lr=self.config.learning_rate)

        # Learnable temperature for contrastive loss
        self.temperature = nn.Parameter(torch.tensor(0.07, device=self.device))

        wandb.init(project="protein-contrastive", config=vars(config))
        print(f"[INIT] WandB logging initialized.")

    def setup_directories(self):
        os.makedirs('checkpoints', exist_ok=True)
        print("[DEBUG] Directories set up.")

    def setup_data(self, train_data, val_data):
        """Setup data loaders for training and validation."""
        print("[DEBUG] Setting up training and validation data loaders.")

        try:
            train_tensors = [train_data[k] for k in ['cell_state', 'pert_protein', 'perturbation']]
            val_tensors = [val_data[k] for k in ['cell_state', 'pert_protein', 'perturbation']]
        except KeyError as e:
            print(f"[ERROR] Missing key in dataset: {e}")
            raise

        print(f"[DEBUG] Train data sizes: {[t.shape for t in train_tensors]}")
        print(f"[DEBUG] Val data sizes: {[t.shape for t in val_tensors]}")

        train_dataset = TensorDataset(*train_tensors)
        val_dataset = TensorDataset(*val_tensors)

        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=True)

        print("[DEBUG] Data loaders initialized successfully.")


    def setup_momentum_encoder(self):
        """Momentum encoder for queue updates (MoCo-like)."""
        print("[DEBUG] Initializing Momentum Encoder...")
        self.momentum_encoder = SimilarityTransformer(self.config.latent_dim).to(self.device)
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False  # Frozen for EMA updates
        print("[DEBUG] Momentum Encoder Initialized.")

    def setup_queue(self):
        """Memory queue for contrastive learning."""
        print(f"[DEBUG] Setting up queue: Size {self.queue_size}, Latent dim {self.latent_dim}")
        self.queue = torch.zeros(self.queue_size, self.latent_dim).to(self.device)
        self.dataset_queue = torch.zeros(self.queue_size, dtype=torch.long).to(self.device)  # ✅ Initialize dataset queue
        self.queue_ptr = 0
        self.queue_full = False
        self.queue = F.normalize(self.queue, dim=1)
        print("[DEBUG] Queue and dataset queue initialized and normalized.")

    @torch.no_grad()
    def update_queue(self, embeddings, dataset_labels):
        batch_size = embeddings.shape[0]
        embeddings = F.normalize(embeddings, dim=1)
        start_idx = self.queue_ptr
        end_idx = (self.queue_ptr + batch_size) % self.queue_size

        if start_idx + batch_size <= self.queue_size:
            self.queue[start_idx:start_idx + batch_size] = embeddings
            self.dataset_queue[start_idx:start_idx + batch_size] = dataset_labels  # Track dataset
        else:
            first_part = self.queue_size - start_idx
            self.queue[start_idx:] = embeddings[:first_part]
            self.queue[:end_idx] = embeddings[first_part:]
            self.dataset_queue[start_idx:] = dataset_labels[:first_part]
            self.dataset_queue[:end_idx] = dataset_labels[first_part:]

        self.queue_ptr = end_idx
        if not self.queue_full and self.queue_ptr < start_idx:
            self.queue_full = True
        
        print(f"[DEBUG] Queue updated: {start_idx} to {end_idx}, Queue Full: {self.queue_full}")
        print(f"[DEBUG] Unique datasets in queue: {len(torch.unique(self.dataset_queue))}")

        print(f"[DEBUG] Queue updated: {start_idx} to {end_idx}, Queue Full: {self.queue_full}")
        if self.dataset_queue.dim() > 1:
            self.dataset_queue = self.dataset_queue.view(-1)  # 🔥 Ensure 1D tensor

        # 🔥 Ensure only non-negative values before passing to bincount
        valid_indices = self.dataset_queue >= 0
        unique_counts = torch.bincount(self.dataset_queue[valid_indices])

        print(f"[DEBUG] Queue Dataset Distribution: {unique_counts.cpu().numpy()}")
        wandb.log({"Queue Dataset Distribution": wandb.Histogram(unique_counts.cpu().numpy())})


    def compute_losses(self, x, y, dataset_labels):
        """Compute multiple contrastive losses."""
        assert x.shape == y.shape, f"[ERROR] Mismatched shapes: x={x.shape}, y={y.shape}"

        print(f"[DEBUG] Computing losses for batch: x={x.shape}, y={y.shape}")
        losses = {}
        metrics = defaultdict(float)

        cont_loss = self.contrastive_loss(x, y, dataset_labels)  # 🔥 Pass dataset labels
        losses['contrastive'] = cont_loss

        losses['barlow'] = barlow_twins_loss(x, y) * 0.5  # Reduced Barlow Twins effect
        losses['vicreg'] = vicreg_loss(x, y)
        losses['wasserstein'] = wasserstein_loss(x, y)
        losses['hyperbolic'] = hyperbolic_loss(x, y)

        total_loss = sum(self.config.loss_weights[k] * v for k, v in losses.items())
        print(f"[DEBUG] Total loss: {total_loss.item():.6f}")

        return total_loss, losses, metrics


    def contrastive_loss(self, x, y, dataset_labels):
        """Uses NT-Xent loss with learnable temperature & hard negatives."""
        x, y = F.normalize(x, dim=1), F.normalize(y, dim=1)

        # Compute positive pair similarity
        sim_pos = torch.sum(x * y, dim=-1, keepdim=True)  # (batch_size, 1)
        print(f"[DEBUG] Positive similarities: Mean={sim_pos.mean().item():.6f}, Min={sim_pos.min().item():.6f}, Max={sim_pos.max().item():.6f}")

        if self.queue_full:
            print(f"[DEBUG] Queue is full, computing hard negatives.")
            sim_neg=torch.matmul(x,self.queue.T)
            print(f"[DEBUG] sim_neg shape: {sim_neg.shape}")
            print(f"[DEBUG] sim_neg Mean: {sim_neg.mean().item():.6f}, Std: {sim_neg.std().item():.6f}, Min: {sim_neg.min().item():.6f}, Max: {sim_neg.max().item():.6f}")
            threshold=sim_neg.mean()-0.1*sim_neg.std()
            print(f"[DEBUG] Hard Negative Threshold: {threshold:.6f}")
            hard_neg_indices=torch.nonzero(sim_neg>threshold,as_tuple=True)[1]
            print(f"[DEBUG] Number of negatives before thresholding: {sim_neg.numel()}")
            print(f"[DEBUG] Number of selected hard negatives: {hard_neg_indices.numel()}")
            print(f"[DEBUG] hard_neg_indices.shape before filtering: {hard_neg_indices.shape}")
            neg_datasets=self.dataset_queue[hard_neg_indices]
            print(f"[DEBUG] neg_datasets.shape before reshaping: {neg_datasets.shape}")
            min_neg_samples=5
            max_neg_samples=sim_neg.shape[1]
            while True:
                num_neg_samples=min(max_neg_samples,hard_neg_indices.numel()//x.shape[0])
                if num_neg_samples>=min_neg_samples:
                    break
                threshold*=0.95
                hard_neg_indices=torch.nonzero(sim_neg>threshold,as_tuple=True)[1]
                print(f"[DEBUG] Adjusting threshold to {threshold:.6f}, found {hard_neg_indices.numel()} negatives")
                if threshold<sim_neg.min():
                    print("[ERROR] Threshold is too low, breaking loop to prevent crash")
                    num_neg_samples=min_neg_samples
                    break
            print(f"[DEBUG] Final negative samples per sample: {num_neg_samples}")
            valid_negatives = (neg_datasets.numel() // x.shape[0]) * x.shape[0]  # Ensure it's a multiple of batch size
            neg_datasets = neg_datasets[:valid_negatives].view(x.shape[0], -1)  # Auto-adjust second dim
            print(f"[DEBUG] neg_datasets.shape after reshaping: {neg_datasets.shape}")
            mask=neg_datasets!=dataset_labels.unsqueeze(1)
            print(f"[DEBUG] mask.shape: {mask.shape}")
            valid_negatives = (hard_neg_indices.numel() // x.shape[0]) * x.shape[0]  # Ensure multiple of batch size
            hard_neg_indices = hard_neg_indices[:valid_negatives].view(x.shape[0], -1)[mask]
            print(f"[DEBUG] hard_neg_indices.shape after filtering: {hard_neg_indices.shape}")
            if hard_neg_indices.numel()<x.shape[0]*5:
                print(f"[WARNING] Found only {hard_neg_indices.numel()} diverse negatives, falling back to top-k.")
                hard_neg_indices=torch.topk(sim_neg,k=5,dim=1,largest=True)[1]
            print(f"[DEBUG] hard_neg_indices.shape before reshaping: {hard_neg_indices.shape}")
            hard_neg_indices=hard_neg_indices.view(x.shape[0],num_neg_samples).long()
            print(f"[DEBUG] self.queue.shape: {self.queue.shape}")
            print(f"[DEBUG] hard_neg_indices.shape before indexing: {hard_neg_indices.shape}")
            hard_negatives=self.queue[hard_neg_indices.view(-1)].view(x.shape[0],num_neg_samples,-1)
            print(f"[DEBUG] hard_negatives.shape: {hard_negatives.shape}")
            sim_neg=torch.matmul(x.unsqueeze(1),hard_negatives.transpose(1,2)).squeeze(1)
            print(f"[DEBUG] sim_neg.shape after matmul: {sim_neg.shape}")
            print(f"[DEBUG] Negative similarities: Mean={sim_neg.mean().item():.6f}, Min={sim_neg.min().item():.6f}, Max={sim_neg.max().item():.6f}")
            logits=torch.cat([sim_pos,sim_neg],dim=1)/(self.temperature.exp()*0.5)
            print(f"[DEBUG] logits shape after concatenation: {logits.shape}")
        else:
            logits=sim_pos/self.temperature.exp()
            print(f"[DEBUG] Using only sim_pos, queue not full yet.")

        # Cross-entropy loss
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=x.device)
        print(f"[DEBUG] Contrastive logits: Mean={logits.mean().item():.6f}, Min={logits.min().item():.6f}, Max={logits.max().item():.6f}")

        wandb.log({"Temperature": self.temperature.exp().item()})

        return F.cross_entropy(logits, labels)

    def train_step(self, batch_data, is_training=True):
        """Runs one step of training or validation using contrastive loss."""
        batch_data = [b.to(self.device) for b in batch_data]
        cell_state, protein, perturbation = batch_data

        # Normalize embeddings
        cell_state, protein = F.normalize(cell_state, dim=-1), F.normalize(protein, dim=-1)

        # Compute losses
        total_loss, loss_components, _ = self.compute_losses(cell_state, protein, perturbation[:, 0])

        if is_training:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            self.temperature.data = max(self.temperature.data * 0.99, torch.tensor(0.02, device=self.device))  # 🔥 Lower bound at 0.02

            # ✅ Ensure dataset_labels are passed correctly
            self.update_queue(protein.detach(), perturbation[:, 0].detach())

        print(f"[DEBUG] Step {'TRAIN' if is_training else 'VAL'} - Loss: {total_loss.item():.6f}")

        return total_loss.item(), loss_components


    def train_epoch(self, loader, is_training=True, epoch=0):
        """Runs one full epoch for training or validation."""
        self.similarity_transformer.train() if is_training else self.similarity_transformer.eval()

        total_losses = defaultdict(float)
        embeddings_list, labels_list = [], []

        with torch.set_grad_enabled(is_training):
            for batch_data in tqdm(loader, desc=f"{'Training' if is_training else 'Validation'} Epoch {epoch}"):
                batch_data = [b.to(self.device) for b in batch_data]
                cell_state, protein, perturbation = batch_data

                # Normalize and store embeddings
                cell_state, protein = F.normalize(cell_state, dim=-1), F.normalize(protein, dim=-1)
                embeddings_list.append(cell_state.detach().cpu())
                labels_list.append(perturbation[:, 0].detach().cpu())  # Ensure correct shape

                # Compute loss
                total_loss, loss_components = self.train_step(batch_data, is_training)
                total_losses['total_loss'] += total_loss
                for k, v in loss_components.items():
                    total_losses[k] += v.item()

        embeddings_tensor = torch.cat(embeddings_list, dim=0)  # (N, latent_dim)
        labels_tensor = torch.cat(labels_list, dim=0)  # (N,)

        print(f"[DEBUG] Pre-VIZ embeddings.shape: {embeddings_tensor.shape}, labels.shape: {labels_tensor.shape}")

        assert embeddings_tensor.shape[0] == labels_tensor.shape[0], f"[ERROR] Mismatch in embeddings ({embeddings_tensor.shape}) vs labels ({labels_tensor.shape})"

        if epoch % self.config.viz_frequency == 0:
            visualize_embeddings(embeddings_tensor, labels_tensor, title=f"t-SNE - {'Train' if is_training else 'Val'} Epoch {epoch}")
            plot_covariance_matrix(embeddings_tensor, title=f"VICReg Covariance - {'Train' if is_training else 'Val'} Epoch {epoch}")
        return {k: v / len(loader) for k, v in total_losses.items()}


    def train(self, train_data, val_data, n_epochs):
        """Runs full training pipeline with optional early stopping."""
        self.setup_data(train_data, val_data)

        best_val_loss = float('inf')
        patience = 0  # Counter for early stopping

        for epoch in range(1, n_epochs + 1):
            print(f"\n========== Epoch {epoch}/{n_epochs} ==========")

            # Train and validate
            train_metrics = self.train_epoch(self.train_loader, is_training=True, epoch=epoch)
            val_metrics = self.train_epoch(self.val_loader, is_training=False, epoch=epoch)

            # Log metrics
            wandb.log({
                'epoch': epoch,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            })

            # Model checkpointing
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                self.save_checkpoint(epoch, best_val_loss)
                print(f"[DEBUG] New best model saved at epoch {epoch}.")
                patience = 0  # Reset early stopping counter
            else:
                patience += 1
                print(f"[DEBUG] No improvement in validation loss ({patience}/{self.config.early_stop_patience}).")

            # Early stopping condition
            if patience >= self.config.early_stop_patience:
                print("[INFO] Early stopping triggered. Training terminated.")
                break

        # Final visualization after training
        print("[DEBUG] Running final embedding visualization...")
        self.final_embedding_visualization()
    
    def final_embedding_visualization(self):
        all_embeddings, all_labels = [], []
        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Extracting Final Embeddings"):
                batch_data = [b.to(self.device) for b in batch_data]
                cell_state, _, perturbation = batch_data
                all_embeddings.append(F.normalize(cell_state, dim=-1).cpu())
                all_labels.append(perturbation[:, 0].cpu())
        
        visualize_embeddings(torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0), title="Final t-SNE")

    def save_checkpoint(self, epoch, loss):
        """Saves model checkpoint with optimizer and queue state."""
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'model_state': self.similarity_transformer.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'queue': self.queue.clone().detach(),
            'queue_ptr': self.queue_ptr
        }
        torch.save(checkpoint, f"checkpoints/model_epoch_{epoch}.pt")
        print(f"[DEBUG] Model checkpoint saved at epoch {epoch} with loss {loss:.6f}.")

def visualize_embeddings(embeddings, dataset_labels, title="t-SNE of Embeddings by Dataset"):
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings.cpu().detach().numpy())

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=dataset_labels,
                        palette=sns.color_palette("husl", len(set(dataset_labels))), alpha=0.7, style=dataset_labels)
        plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title(title)
        wandb.log({title: wandb.Image(plt)})
        plt.show()

def compute_cosine_similarity(train_embeddings, val_embeddings):
    """Compute cosine similarity between train and validation embeddings."""
    cosine_sim = F.cosine_similarity(train_embeddings.unsqueeze(1), val_embeddings.unsqueeze(0), dim=-1)
    
    print(f"[DEBUG] Cosine Similarity - Mean: {cosine_sim.mean().item()}, Min: {cosine_sim.min().item()}, Max: {cosine_sim.max().item()}")

    sns.histplot(cosine_sim.flatten().cpu().numpy(), bins=50, kde=True)
    plt.title("Cosine Similarity Distribution (Train vs. Val)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    
    wandb.log({"Cosine Similarity Histogram": wandb.Image(plt)})
    plt.show()



def plot_covariance_matrix(embeddings, title="Covariance Matrix"):
    """Plots VICReg covariance matrix for analysis."""
    cov_matrix = torch.cov(embeddings.T).cpu().numpy()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cov_matrix, cmap="coolwarm", square=True)
    plt.title(title)
    wandb.log({title: wandb.Image(plt)})
    plt.close()




