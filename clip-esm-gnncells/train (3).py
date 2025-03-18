# train.py
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

def masked_sequence_loss(logits, targets, padding_mask):
    """
    Compute sequence loss only on non-padded tokens.
    
    Args:
        logits: Predicted logits [batch_size, seq_len, vocab_size]
        targets: Target tokens [batch_size, seq_len]
        padding_mask: Binary mask [batch_size, seq_len] (1 for real tokens, 0 for padding)
    """
    flat_logits = logits.view(-1, logits.size(-1))
    flat_targets = targets.view(-1)
    flat_mask = padding_mask.view(-1)
    
    # Only compute loss on non-padded tokens
    active_logits = flat_logits[flat_mask.bool()]
    active_targets = flat_targets[flat_mask.bool()]
    
    return F.cross_entropy(active_logits, active_targets)

def flow_matching_loss(v, target_v):
    """Compute MSE loss between predicted and target vector fields."""
    return F.mse_loss(v, target_v)

def path_length_regularization(v):
    """Compute path length regularization term."""
    return torch.mean(torch.norm(v, dim=-1) ** 2)

def jacobian_regularization(v, x):
    """Compute Jacobian regularization term."""
    jac = torch.autograd.functional.jacobian(lambda x: v, x)
    return torch.norm(jac, p='fro')

class TrainFlow:
    """Training manager for the protein flow model."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        self.setup_checkpointing()
        self.setup_esm()
        
    def setup_esm(self):
        """Initialize ESM model components for sequence prediction."""
        # Load ESM model
        self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_model = self.esm_model.to(self.device)
        
        # Extract and freeze ESM head
        self.esm_head = self.esm_model.lm_head
        for param in self.esm_head.parameters():
            param.requires_grad = False
        
        # ESM model constants    
        self.esm_vocab_size = len(self.alphabet)
        
    def setup_logging(self):
        """Initialize wandb logging."""
        self.logger = wandb.init(project="protein-flow", config=self.config)
        
    def setup_checkpointing(self):
        """Create checkpoint directory."""
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def setup_data(self, train_data, val_data):
        """
        Setup data loaders and initialize queue for training.
        
        Args:
            train_data: Dictionary containing training tensors
            val_data: Dictionary containing validation tensors
        """
        print("Train shapes:", {k: v.shape for k, v in train_data.items()})
        print("Val shapes:", {k: v.shape for k, v in val_data.items()})
        
        # Create training dataset
        train_tensors = [
            train_data['cell_state'], 
            train_data['pert_protein'], 
            train_data['pert_effect'],
            train_data['pert_esm_tokens'],
            train_data['topk_esm_tokens'],
            train_data['pert_esm_embeddings'],
            train_data['topk_esm_embeddings']
        ]
        train_dataset = TensorDataset(*train_tensors)
        
        # Create validation dataset
        val_tensors = [
            val_data['cell_state'], 
            val_data['pert_protein'],
            val_data['pert_esm_tokens'],
            val_data['topk_esm_tokens'],
            val_data['pert_esm_embeddings'],
            val_data['topk_esm_embeddings']
        ]
        val_dataset = TensorDataset(*val_tensors)

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=True
        )
        
        # Initialize queue
        self.setup_queue()
        
    def setup_queue(self):
        """Initialize memory queue for contrastive learning."""
        self.protein_queue = torch.zeros(
            self.config.queue_size,
            self.config.latent_dim
        ).to(self.device)
        self.queue_ptr = 0
        self.queue_full = False
        self.protein_queue = F.normalize(self.protein_queue, dim=1)
    
    def update_queue(self, new_proteins):
        """
        Update memory queue with new protein embeddings.
        
        Args:
            new_proteins: New protein embeddings to add to queue
        """
        batch_size = len(new_proteins)
        
        if not self.queue_full:
            space_left = self.config.queue_size - self.queue_ptr
            if space_left < batch_size:
                # Handle partial fill
                self.protein_queue[self.queue_ptr:] = new_proteins[:space_left].detach()
                remainder = batch_size - space_left
                self.protein_queue[:remainder] = new_proteins[space_left:].detach()
                self.queue_full = True
            else:
                self.protein_queue[self.queue_ptr:self.queue_ptr + batch_size] = new_proteins.detach()
            
            if self.queue_ptr + batch_size >= self.config.queue_size:
                self.queue_full = True
            
            print(f"Queue filling: {min(self.queue_ptr + batch_size, self.config.queue_size)}/{self.config.queue_size}")
        else:
            # Standard circular update once full
            end_idx = (self.queue_ptr + batch_size) % self.config.queue_size
            if end_idx > self.queue_ptr:
                self.protein_queue[self.queue_ptr:end_idx] = new_proteins.detach()
            else:
                # Handle wrap-around
                first_part = self.config.queue_size - self.queue_ptr
                self.protein_queue[self.queue_ptr:] = new_proteins[:first_part].detach()
                self.protein_queue[:end_idx] = new_proteins[first_part:].detach()
        
        self.queue_ptr = (self.queue_ptr + batch_size) % self.config.queue_size

    def get_negatives(self, protein_embeddings, k_hard=None):
        """
        Get both random and hard negative samples for contrastive learning.
        
        Args:
            protein_embeddings: Current batch protein embeddings
            k_hard: Number of hard negatives to sample
        """
        if not self.queue_full:
            return protein_embeddings[torch.randperm(len(protein_embeddings))], None
            
        with torch.no_grad():
            # Compute similarities for hard negative mining
            sim = F.cosine_similarity(
                protein_embeddings.unsqueeze(1),
                self.protein_queue.unsqueeze(0),
                dim=2
            )
            
            # Get hardest negatives (most similar)
            k = k_hard or len(protein_embeddings)
            hard_indices = torch.topk(sim, k=k).indices
            hard_negs = self.protein_queue[hard_indices]
            
            # Also include random in-batch negatives
            in_batch_negs = protein_embeddings[torch.randperm(len(protein_embeddings))]
            
        return in_batch_negs, hard_negs

    def contrastive_loss(self, x, y, temperature=0.1, queue=None, in_batch_negs=None):
        """
        Compute contrastive loss with weighted negatives.
        
        Args:
            x: Query embeddings
            y: Positive embeddings
            temperature: Temperature for scaling similarities
            queue: Queue of negative examples
            in_batch_negs: In-batch negative examples
        """
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        
        # Start with positives
        sim_pos = torch.matmul(x, y.T)
        
        # Collect negatives with weights
        neg_terms = []
        if in_batch_negs is not None:
            in_batch_negs = F.normalize(in_batch_negs, dim=-1)
            sim_in_batch = torch.matmul(x, in_batch_negs.T) * 0.3  # Local weight
            neg_terms.append(sim_in_batch)
            
        if queue is not None and self.queue_full:
            queue = F.normalize(queue, dim=-1)
            sim_queue = torch.matmul(x, queue.T) * 0.7  # Global weight
            neg_terms.append(sim_queue)
            
        # Combine all similarities
        if neg_terms:
            sim_matrix = torch.cat([sim_pos] + neg_terms, dim=1)
        else:
            sim_matrix = sim_pos
            
        sim_matrix = sim_matrix / temperature
        labels = torch.arange(len(x), device=x.device)
        
        return F.cross_entropy(sim_matrix, labels)
    

    def train_epoch(self, model, optimizer):
        """Execute one training epoch."""
        model.train()
        total_loss = 0
        
        for batch_idx, batch_data in enumerate(tqdm(self.train_loader)):
            # Unpack batch data with consistent naming
            cell_state, pert_protein, pert_effect, \
            pert_esm_tokens, topk_esm_tokens, \
            pert_esm_embeddings, topk_esm_embeddings = [
                d.to(self.device) for d in batch_data
            ]
            
            # Create padding masks
            pert_mask = (pert_esm_tokens != self.alphabet.padding_idx)
            topk_mask = (topk_esm_tokens != self.alphabet.padding_idx)
            
            # Get flows and regularization
            flows, reg_losses = model({
                'cell_state': cell_state,
                'pert_protein': pert_protein,
                'pert_effect': pert_effect,
                'topk_esm_embeddings': topk_esm_embeddings
            }, return_regularization=True)
            
            # Initialize losses
            loss_dict = {}
            total_batch_loss = 0
            
            # Flow matching and regularization losses
            for flow_name, (v, xt, t, ut) in flows.items():
                if 'esm' not in flow_name:  # Skip ESM projections
                    target_v = ut if 'protein' in flow_name else xt
                    flow_loss = flow_matching_loss(v, target_v)
                    reg_loss = reg_losses[flow_name]
                    
                    total_batch_loss += flow_loss + self.config.regularization_weight * reg_loss
                    loss_dict[f'flow_loss_{flow_name}'] = flow_loss.item()
                    loss_dict[f'reg_loss_{flow_name}'] = reg_loss.item()
            
            # Contrastive loss
            in_batch_negs, hard_negs = self.get_negatives(pert_protein)
            cont_loss = self.contrastive_loss(
                cell_state, pert_protein, 
                self.config.temperature,
                queue=hard_negs if self.queue_full else None,
                in_batch_negs=in_batch_negs
            )
            total_batch_loss += self.config.contrastive_weight * cont_loss
            loss_dict['contrastive_loss'] = cont_loss.item()
            
            # Perturbation protein sequence loss
            if 'pert_protein_esm' in flows:
                pert_seq_loss = masked_sequence_loss(
                    self.esm_head(flows['pert_protein_esm']), 
                    pert_esm_tokens,
                    pert_mask
                )
                total_batch_loss += self.config.sequence_weight * pert_seq_loss
                loss_dict['pert_sequence_loss'] = pert_seq_loss.item()
            
            # Top-k proteins sequence loss
            if 'topk_protein_esm' in flows:
                topk_seq_loss = masked_sequence_loss(
                    self.esm_head(flows['topk_protein_esm']).view(-1, self.config.k_proteins, self.esm_vocab_size),
                    topk_esm_tokens,
                    topk_mask
                )
                total_batch_loss += self.config.topk_weight * topk_seq_loss
                loss_dict['topk_sequence_loss'] = topk_seq_loss.item()
            
            # Optimization
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()
            
            # Update queue
            with torch.no_grad():
                self.update_queue(pert_protein)
                
            total_loss += total_batch_loss.item()
            wandb.log(loss_dict)
        
        return total_loss / len(self.train_loader)

    def validate(self, model):
        """Execute validation."""
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                # Unpack batch data with consistent naming
                cell_state, pert_protein, pert_esm_tokens, topk_esm_tokens, \
                pert_esm_embeddings, topk_esm_embeddings = [
                    d.to(self.device) for d in batch_data
                ]
                
                # Create padding masks
                pert_mask = (pert_esm_tokens != self.alphabet.padding_idx)
                topk_mask = (topk_esm_tokens != self.alphabet.padding_idx)
                
                # Get model outputs
                flows = model({
                    'cell_state': cell_state,
                    'pert_protein': pert_protein,
                    'topk_esm_embeddings': topk_esm_embeddings
                })
                
                total_batch_loss = 0
                
                # Flow matching loss
                for flow_name, (v, xt, t, ut) in flows.items():
                    if 'esm' not in flow_name:
                        target_v = xt
                        flow_loss = flow_matching_loss(v, target_v)
                        total_batch_loss += flow_loss
                
                # Contrastive loss
                cont_loss = self.contrastive_loss(
                    cell_state, pert_protein,
                    self.config.temperature,
                    queue=None,
                    in_batch_negs=pert_protein[torch.randperm(len(pert_protein))]
                )
                total_batch_loss += self.config.contrastive_weight * cont_loss
                
                # Sequence losses
                if 'pert_protein_esm' in flows:
                    pert_seq_loss = masked_sequence_loss(
                        self.esm_head(flows['pert_protein_esm']),
                        pert_esm_tokens,
                        pert_mask
                    )
                    total_batch_loss += self.config.sequence_weight * pert_seq_loss
                
                if 'topk_protein_esm' in flows:
                    topk_seq_loss = masked_sequence_loss(
                        self.esm_head(flows['topk_protein_esm']).view(-1, self.config.k_proteins, self.esm_vocab_size),
                        topk_esm_tokens,
                        topk_mask
                    )
                    total_batch_loss += self.config.topk_weight * topk_seq_loss
                
                val_loss += total_batch_loss.item()
                
        return val_loss / len(self.val_loader)

    def train(self, model, train_data, val_data):
        """
        Full training loop with validation and checkpointing.
        
        Args:
            model: TripleFlow model instance
            train_data: Training data dictionary
            val_data: Validation data dictionary
            
        Returns:
            best_val_loss: Best validation loss achieved
        """
        self.setup_data(train_data, val_data)
        optimizer = Adam(model.parameters(), lr=self.config.learning_rate)
        
        best_val_loss = float('inf')
        for epoch in range(self.config.epochs):
            try:
                train_loss = self.train_epoch(model, optimizer)
                val_loss = self.validate(model)
                
                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'epoch': epoch
                })
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(model, optimizer, epoch, val_loss)
                    
            except RuntimeError as e:
                print(f"Error in epoch {epoch}: {e}")
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                continue
                
        return best_val_loss

    def save_checkpoint(self, model, optimizer, epoch, loss):
        """Save model checkpoint including queue state."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'queue_state': {
                'queue': self.protein_queue,
                'ptr': self.queue_ptr,
                'full': self.queue_full
            }
        }, f"{self.checkpoint_dir}/checkpoint_epoch_{epoch}.pt")
        
    def load_checkpoint(self, model, optimizer, path):
        """Load model checkpoint and restore queue state."""
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'queue_state' in checkpoint:
            self.protein_queue = checkpoint['queue_state']['queue']
            self.queue_ptr = checkpoint['queue_state']['ptr']
            self.queue_full = checkpoint['queue_state']['full']
            
        return checkpoint['epoch'], checkpoint['loss']
    


    def inference_step(self, model, cell_states):
        """
        Generate proteins and top-k predictions from cell states.
        
        Args:
            model: Trained TripleFlow model
            cell_states: Input cell states [batch_size, latent_dim]
            
        Returns:
            Dictionary containing:
            - pert_protein_logits: Perturbation protein logits
            - pert_protein_state: Final protein state
            - topk_protein_logits: Top-k proteins logits
            - topk_protein_state: Final top-k state
            - flows: Raw flow outputs for analysis
        """
        model.eval()
        with torch.no_grad():
            # Move input to device
            cell_states = cell_states.to(self.device)
            
            # Get flow predictions
            flows = model({
                'cell_state': cell_states
            })
            
            outputs = {
                'flows': flows  # Store raw flows for analysis
            }
            
            # Get perturbation protein predictions
            if 'pert_protein_esm' in flows:
                pert_logits = self.esm_head(flows['pert_protein_esm'])
                outputs['pert_protein_logits'] = pert_logits
                outputs['pert_protein_state'] = flows['pert_protein_esm']
            
            # Get top-k protein predictions
            if 'topk_protein_esm' in flows:
                topk_logits = self.esm_head(flows['topk_protein_esm'])
                outputs['topk_protein_logits'] = topk_logits.view(-1, self.config.k_proteins, self.esm_vocab_size)
                outputs['topk_protein_state'] = flows['topk_protein_esm']
            
            return outputs
            
    def generate_sequences(self, logits, return_confidence=False):
        """
        Convert ESM logits to amino acid sequences.
        
        Args:
            logits: ESM logits [batch_size, seq_len, vocab_size] or [batch_size, k, seq_len, vocab_size]
            return_confidence: Whether to return prediction confidences
            
        Returns:
            sequences: List of amino acid sequences
            confidences: (optional) Prediction confidences
        """
        # Get predictions
        if logits.dim() == 3:
            # Single sequence prediction
            probs = F.softmax(logits, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)
        else:
            # Top-k sequences prediction
            probs = F.softmax(logits, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)
        
        # Convert to sequences
        sequences = []
        for pred in predictions:
            if pred.dim() == 2:  # Top-k case
                seq_list = []
                for p in pred:
                    seq = ''.join([self.alphabet.get_tok(t.item()) for t in p 
                                 if t.item() != self.alphabet.padding_idx])
                    seq_list.append(seq)
                sequences.append(seq_list)
            else:  # Single sequence case
                seq = ''.join([self.alphabet.get_tok(t.item()) for t in pred 
                             if t.item() != self.alphabet.padding_idx])
                sequences.append(seq)
        
        if return_confidence:
            return sequences, confidences
        return sequences

    def batch_inference(self, model, cell_states, batch_size=32):
        """
        Run inference on large number of cell states in batches.
        
        Args:
            model: Trained TripleFlow model
            cell_states: Input cell states [num_cells, latent_dim]
            batch_size: Batch size for processing
            
        Returns:
            Dictionary containing all predictions and confidences
        """
        all_outputs = {
            'pert_sequences': [],
            'pert_confidences': [],
            'topk_sequences': [],
            'topk_confidences': []
        }
        
        # Process in batches
        for i in range(0, len(cell_states), batch_size):
            batch = cell_states[i:i + batch_size]
            
            # Get predictions
            outputs = self.inference_step(model, batch)
            
            # Get perturbation protein sequences
            if 'pert_protein_logits' in outputs:
                sequences, confidences = self.generate_sequences(
                    outputs['pert_protein_logits'], 
                    return_confidence=True
                )
                all_outputs['pert_sequences'].extend(sequences)
                all_outputs['pert_confidences'].extend(confidences.cpu().numpy())
            
            # Get top-k protein sequences
            if 'topk_protein_logits' in outputs:
                sequences, confidences = self.generate_sequences(
                    outputs['topk_protein_logits'],
                    return_confidence=True
                )
                all_outputs['topk_sequences'].extend(sequences)
                all_outputs['topk_confidences'].extend(confidences.cpu().numpy())
        
        return all_outputs