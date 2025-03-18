import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader,TensorDataset
from torch.optim import Adam
import os


import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
import wandb
import esm
from tqdm import tqdm
import os
import numpy as np
import torch.nn.functional as F


class TransitionClassifier(nn.Module):
    """Classifier for protein-induced cell state transitions."""
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.use_trajectory = config.use_trajectory
        self.latent_dim = config.latent_dim
        self.hidden_dim = config.hidden_dim
 
        if self.use_trajectory:
            # Fix encoder layer input
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim*2,
                nhead=config.n_heads,
                dim_feedforward=self.hidden_dim,
                dropout=config.dropout,
                batch_first=True
            )
            # Fix projection dim
            self.trajectory_proj = nn.Sequential(
                nn.Linear(self.latent_dim*2, self.hidden_dim),  # Changed from latent_dim*2
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            )

            self.trajectory_encoder = nn.TransformerEncoder(encoder_layer,
                                                            num_layers=config.n_layers)
        
        self.state_proj = nn.Sequential(nn.Linear(self.latent_dim,self.hidden_dim),nn.LayerNorm(self.hidden_dim),nn.ReLU(),nn.Dropout(config.dropout))
        self.protein_proj = nn.Sequential(nn.Linear(self.latent_dim,self.hidden_dim),nn.LayerNorm(self.hidden_dim),nn.ReLU(),nn.Dropout(config.dropout))
        classifier_input_dim = self.hidden_dim * 2  # base dimension before adding trajectory
        if self.use_trajectory:
            classifier_input_dim += self.hidden_dim
        self.classifier = nn.Sequential(nn.Linear(classifier_input_dim,self.hidden_dim),nn.LayerNorm(self.hidden_dim),nn.ReLU(),nn.Dropout(config.dropout),nn.Linear(self.hidden_dim,self.hidden_dim//2),nn.ReLU(),nn.Linear(self.hidden_dim//2,1),nn.Sigmoid())

    def encode_trajectory(self, states, proteins):
        """Encodes full state and protein trajectories using transformer."""
        traj_features = torch.cat([states, proteins], dim=-1)
        encoded = self.trajectory_encoder(traj_features)
        projected = self.trajectory_proj(encoded)

        return projected


    def forward(self, inputs):
       # Just normalize vectors to unit length
    #    cell_state = F.normalize(inputs['cell_state'], dim=1)
       cell_state = F.normalize(inputs['cell_state'], dim=1)
    #    protein = F.normalize(inputs['protein'], dim=1)
       protein = F.normalize(inputs['protein'], dim=1)

       
       state_features = self.state_proj(cell_state)
       protein_features = self.protein_proj(protein)
       features = [state_features, protein_features]
       
       if self.use_trajectory and 'trajectory' in inputs:
           features.append(self.encode_trajectory(inputs['trajectory']['states'],inputs['trajectory']['proteins']))
       return self.classifier(torch.cat(features,dim=-1))
    
    def get_score_function(self,cell_state,protein,**kwargs):
        def score_fn(x_t,t):
            inputs = {'cell_state':cell_state,'protein':x_t}
            inputs.update(kwargs)
            with torch.no_grad():
                return self(inputs)
        return score_fn

class TrainFlow:
    """Training manager for the protein flow model."""
    def __init__(self,config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        self.setup_checkpointing()
        self.setup_queue()

    def setup_logging(self):
        """Initialize wandb logging."""
        self.logger = wandb.init(project="protein-flow",config=self.config)

    def setup_checkpointing(self):
        """Create checkpoint directory."""
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir,exist_ok=True)

    def setup_data(self,train_data,val_data):
        """Setup data loaders and initialize queue for training."""
        train_tensors = [train_data[k] for k in ['cell_state','pert_protein','perturbation']]
        train_dataset = TensorDataset(*train_tensors)
        val_tensors = [val_data[k] for k in ['cell_state','pert_protein','perturbation']]
        val_dataset = TensorDataset(*val_tensors)
        self.train_loader = DataLoader(train_dataset,batch_size=self.config.batch_size,shuffle=True,drop_last=True)
        self.val_loader = DataLoader(val_dataset,batch_size=self.config.batch_size,shuffle=False,drop_last=True)
        self.setup_queue()

    def setup_queue(self):
        """Initialize memory queue for contrastive learning."""
        # log_gpu_memory("Before queue init")
        print(f"Setting up queue: size={self.config.queue_size}, latent_dim={self.config.latent_dim}")
        self.protein_queue = torch.zeros(self.config.queue_size,self.config.latent_dim).to(self.device)
        print("Queue initialized")
        # log_gpu_memory("After queue creation")
        
        self.queue_ptr = 0
        self.queue_full = False
        self.protein_queue = F.normalize(self.protein_queue,dim=1)
        # log_gpu_memory("After queue normalize")

    def compute_queue_diversity(self):
        """Compute diversity metrics for queue in chunks."""
        if not self.queue_full:
            return
            
        print("[DEBUG] Computing Queue Diversity")
        chunk_size = 256
        n_chunks = (self.config.queue_size + chunk_size - 1) // chunk_size
        similarities = []
        
        with torch.no_grad():
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, self.config.queue_size)
                queue_chunk = self.protein_queue[start_idx:end_idx]
                
                chunk_sim = F.cosine_similarity(
                    queue_chunk.unsqueeze(1),
                    self.protein_queue.unsqueeze(0),
                    dim=2
                )
                similarities.append(chunk_sim)
                
            all_similarities = torch.cat(similarities, dim=0)
            
            diversity_metrics = {
                'queue_metrics/avg_similarity': all_similarities.mean().item(),
                'queue_metrics/std_dev': self.protein_queue.std(dim=0).mean().item()
            }
            wandb.log(diversity_metrics)

    def update_queue(self, new_proteins):
        """Update queue with new embeddings and track queue status correctly."""
        with torch.no_grad():
            batch_size = len(new_proteins)
            
            # Update pointer and check if we've filled the queue
            new_ptr = (self.queue_ptr + batch_size) % self.config.queue_size
            
            # Mark queue as full if we've made a complete pass
            if not self.queue_full:
                if new_ptr < self.queue_ptr:  # We wrapped around
                    self.queue_full = True
                elif new_ptr >= self.config.queue_size:  # We filled it exactly or went past
                    self.queue_full = True
            
            # Update queue at current position
            if self.queue_ptr + batch_size <= self.config.queue_size:
                # Simple case: batch fits before end of queue
                self.protein_queue[self.queue_ptr:self.queue_ptr + batch_size] = new_proteins
            else:
                # Batch wraps around end of queue
                first_part = self.config.queue_size - self.queue_ptr
                self.protein_queue[self.queue_ptr:] = new_proteins[:first_part]
                self.protein_queue[:batch_size - first_part] = new_proteins[first_part:]
            
            # Update pointer for next batch
            self.queue_ptr = new_ptr
            
            # Normalize queue
            self.protein_queue = F.normalize(self.protein_queue, dim=1)
            
            # Debug print
            print(f"Queue status - Full: {self.queue_full}, Ptr: {self.queue_ptr}, "
                f"Size: {self.config.queue_size}, Batch: {batch_size}")

            # Compute diversity metrics if queue is full
            if self.queue_full:
                print(f"Queue shape before similarity: {self.protein_queue.shape}")
                self.compute_queue_diversity()


    def get_negatives(self, protein_embeddings, k_hard=None):
        """
        Improved negative mining:
        - Filters for semi-hard negatives.
        - Adds momentum to queue updates.
        """
        if not self.queue_full:
            in_batch_negs = protein_embeddings[torch.randperm(len(protein_embeddings))]  
            return in_batch_negs, None, None  

        with torch.no_grad():
            # Compute cosine similarity between batch and queue
            sim = F.cosine_similarity(
                protein_embeddings.unsqueeze(1),
                self.protein_queue.unsqueeze(0),
                dim=2
            )

            # After normalization, cosine similarities will be between -1 and 1
            positive_threshold = 0.3   # Negatives should be more similar than this
            negative_threshold = 0.7   # But less similar than this
            mask = (sim > positive_threshold) & (sim < negative_threshold)

            # Select k hard negatives
            valid_indices = mask.nonzero()
            if len(valid_indices) > 0:
                perm = torch.randperm(len(valid_indices))[:k_hard]
                hard_indices = valid_indices[perm]
                hard_negs = self.protein_queue[hard_indices]
            else:
                # Fallback to random negatives
                hard_indices = torch.randperm(len(self.protein_queue))[:k_hard]
                hard_negs = self.protein_queue[hard_indices]

            # Define in-batch negatives
            in_batch_negs = protein_embeddings[torch.randperm(len(protein_embeddings))]  

            # More diverse noise generation
            noise_scale = torch.rand_like(protein_embeddings) *  0.05 
            gaussian_noise = torch.randn_like(protein_embeddings) * noise_scale
            noise_negs = protein_embeddings + gaussian_noise

        return in_batch_negs, hard_negs, noise_negs  



    def contrastive_loss(self, x, y, temperature=0.2, margin=0.3, in_batch_negs=None, queue=None, noise_negs=None):
        """
        Compute contrastive loss with proper scaling and temperature handling.
        Args:
            x, y: anchor and positive embeddings
            temperature: softmax temperature (default increased to 0.1 for stability)
            margin: margin for negative pairs
            in_batch_negs: in-batch negative samples
            queue: memory bank negatives
            noise_negs: noise-augmented negatives
        """
        # First normalize all vectors
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        if queue is not None:
            queue = F.normalize(queue.reshape(-1, x.shape[-1]), dim=1)
        if in_batch_negs is not None:
            in_batch_negs = F.normalize(in_batch_negs, dim=1)
        if noise_negs is not None:
            noise_negs = F.normalize(noise_negs, dim=1)
            
        # Apply scaling only once, here
        scale = min(np.sqrt(self.config.latent_dim) / 4, 8)  # Cap scaling
        x = x * scale
        y = y * scale
        if queue is not None:
            queue = queue * scale
        if in_batch_negs is not None:
            in_batch_negs = in_batch_negs * scale
        if noise_negs is not None:
            noise_negs = noise_negs * scale

        # Compute positive similarities
        sim_pos = torch.matmul(x, y.T)
        
        # Collect negative similarities
        neg_terms = []
        if in_batch_negs is not None:
            sim_in_batch = torch.matmul(x, in_batch_negs.T)
            neg_terms.append(sim_in_batch)
        
        if queue is not None and self.queue_full:
            sim_queue = torch.matmul(x, queue.T)
            neg_terms.append(sim_queue)
        
        if noise_negs is not None:
            sim_noise = torch.matmul(x, noise_negs.T)
            neg_terms.append(sim_noise)

        # Concatenate all similarities
        sim_matrix = torch.cat([sim_pos] + neg_terms, dim=1) if neg_terms else sim_pos

        # Apply temperature scaling
        logits = sim_matrix / temperature

        # Create labels for positive pairs (diagonal elements)
        labels = torch.arange(len(x), device=x.device)
        
        # Apply margin to negative pairs
        neg_mask = torch.ones_like(sim_matrix)
        neg_mask.scatter_(1, labels.unsqueeze(1), 0)
        logits = logits + margin * neg_mask  # Shift only negative logits

        # Compute InfoNCE loss
        loss = F.cross_entropy(logits, labels)

        # Compute metrics for monitoring
        with torch.no_grad():
            metrics = {
                'pos_sim': sim_pos.mean().item(),
                'in_batch_neg_sim': sim_in_batch.mean().item() if in_batch_negs is not None else 0,
                'queue_neg_sim': sim_queue.mean().item() if queue is not None and self.queue_full else 0,
                'noise_neg_sim': sim_noise.mean().item() if noise_negs is not None else 0,
                'n_random_negs': len(in_batch_negs) if in_batch_negs is not None else 0,
                'n_queue_negs': len(queue) if queue is not None and self.queue_full else 0,
                'n_noise_negs': len(noise_negs) if noise_negs is not None else 0,
                # Additional metrics for debugging
                'max_logit': logits.max().item(),
                'min_logit': logits.min().item(),
                'mean_logit': logits.mean().item(),
                'temperature': temperature
            }

        return loss, metrics



    
    def train_epoch(self,model,optimizer):
        """Execute one training epoch."""
        model.train()
        total_loss = 0
        for batch_idx,batch_data in enumerate(self.train_loader):
            batch_data = [b.to(self.device) for b in batch_data]
            cell_state,pert_protein,perturbation = batch_data
            optimizer.zero_grad()
            outputs = model({
                'cell_state':cell_state,
                'pert_protein':pert_protein,
                'perturbation':perturbation
            })
            loss = model.loss_function(outputs,batch_data)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.update_queue(pert_protein)
            total_loss += loss.item()
            wandb.log({'batch_loss':loss.item()})
        return total_loss/len(self.train_loader)

    def validate(self,model):
        """Execute validation."""
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_data in self.val_loader:
                batch_data = [b.to(self.device) for b in batch_data]
                cell_state,pert_protein,perturbation = batch_data
                outputs = model({
                    'cell_state':cell_state,
                    'pert_protein':pert_protein,
                    'perturbation':perturbation
                })
                loss = model.loss_function(outputs,batch_data)
                val_loss += loss.item()
        return val_loss/len(self.val_loader)

    def train(self,model,train_data,val_data):
        """Full training loop with validation and checkpointing."""
        self.setup_data(train_data,val_data)
        optimizer = Adam(model.parameters(),lr=self.config.learning_rate)
        best_val_loss = float('inf')
        for epoch in range(self.config.epochs):
            try:
                train_loss = self.train_epoch(model,optimizer)
                val_loss = self.validate(model)
                wandb.log({'train_loss':train_loss,'val_loss':val_loss,'epoch':epoch})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(model,optimizer,epoch,val_loss)
            except RuntimeError as e:
                print(f"Error in epoch {epoch}: {e}")
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                continue
        return best_val_loss

    def save_checkpoint(self,model,optimizer,epoch,loss):
        """Save model checkpoint including queue state."""
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':loss,
            'queue_state':{
                'queue':self.protein_queue,
                'ptr':self.queue_ptr,
                'full':self.queue_full
            }
        },f"{self.checkpoint_dir}/checkpoint_epoch_{epoch}.pt")
