import torch
import torch.nn.functional as F
import wandb
from types import SimpleNamespace
from models.flows import TripleFlow
from classifier_flow import TrainFlow, TransitionClassifier
import os
from collections import defaultdict
import numpy as np

class TrainFlowClassifier:
    def __init__(self,config):
        self.config = config
        self.train_flow = TrainFlow(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_directories()
        self.setup_models()
        self.setup_queue()
        wandb.init(project="protein-flow-classifier",config=vars(config))

    def analyze_embeddings(self, data, name="train", chunk_size=256):
        print(f"\nAnalyzing {name} embeddings:")
        
        metrics = defaultdict(float)
        n_chunks = (len(data['cell_state']) + chunk_size - 1) // chunk_size
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(data['cell_state']))
            
            scale = np.sqrt(self.config.latent_dim)
            cell_chunk = F.normalize(data['cell_state'][start_idx:end_idx], dim=1).to(self.device) * scale
            prot_chunk = F.normalize(data['pert_protein'][start_idx:end_idx], dim=1).to(self.device) * scale
            
            with torch.no_grad():
                # Now norms should be ~1
                metrics['cell_norm'] += torch.norm(cell_chunk, dim=1).mean().item()
                metrics['prot_norm'] += torch.norm(prot_chunk, dim=1).mean().item()
                
                # Rest of metrics same as before...
                metrics['cell_std'] += cell_chunk.std(dim=0).mean().item()
                metrics['prot_std'] += prot_chunk.std(dim=0).mean().item()
                
                cell_sims = F.cosine_similarity(cell_chunk.unsqueeze(1), 
                                                cell_chunk.unsqueeze(0), dim=2)
                prot_sims = F.cosine_similarity(prot_chunk.unsqueeze(1), 
                                                prot_chunk.unsqueeze(0), dim=2)
                
                metrics['cell_self_sim'] += cell_sims.mean().item()
                metrics['prot_self_sim'] += prot_sims.mean().item()
        
        metrics = {k: v/n_chunks for k,v in metrics.items()}
        
        print("\nEmbedding Quality Metrics:")
        print(f"Cell Embeddings:")
        print(f"- Average Norm (should be ~1): {metrics['cell_norm']:.4f}")
        print(f"- Feature Std: {metrics['cell_std']:.4f}")
        print(f"- Self Similarity: {metrics['cell_self_sim']:.4f}")
        
        print(f"\nProtein Embeddings:")
        print(f"- Average Norm (should be ~1): {metrics['prot_norm']:.4f}")
        print(f"- Feature Std: {metrics['prot_std']:.4f}")
        print(f"- Self Similarity: {metrics['prot_self_sim']:.4f}")
        
        return metrics
        
    def setup_directories(self):
        self.dirs = ['checkpoints','flow_checkpoints','classifier_checkpoints']
        for d in self.dirs:
            os.makedirs(d,exist_ok=True)

    def setup_queue(self):
        """Initialize memory queue for contrastive learning."""
        self.protein_queue = torch.zeros(self.config.queue_size,self.config.latent_dim).to(self.device)
        self.queue_ptr = 0
        self.queue_full = False
        self.protein_queue = F.normalize(self.protein_queue,dim=1)

            
    def setup_models(self):
        self.flow_model = TripleFlow(self.config).to(self.device)
        self.classifier = TransitionClassifier(self.config).to(self.device)
        self.flow_optimizer = torch.optim.Adam(self.flow_model.parameters(),lr=self.config.learning_rate)
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(),lr=self.config.learning_rate)

        flow_params = sum(p.numel() for p in self.flow_model.parameters())
        class_params = sum(p.numel() for p in self.classifier.parameters())
        print(f"Flow model parameters: {flow_params/1e6:.2f}M")
        print(f"Classifier parameters: {class_params/1e6:.2f}M")


    def prepare_classifier_batch(self,flow_outputs,batch_data):
        classifier_inputs = {
            'cell_state':batch_data['cell_state'],
            'protein':batch_data['pert_protein']
        }
        if self.config.use_trajectory:
            classifier_inputs['trajectory'] = flow_outputs['trajectory']
        return classifier_inputs

    def train_step(self,batch_data, is_training=True):

        def log_gpu_memory(tag=""):
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9  # Convert to GB
                cached = torch.cuda.memory_reserved() / 1e9  # Convert to GB
                
                free_mem, total_mem = torch.cuda.mem_get_info()  # Get total and free GPU memory
                free_mem, total_mem = free_mem / 1e9, total_mem / 1e9  # Convert to GB
                used_mem = total_mem - free_mem  # Actual used GPU memory

                print(f"\nGPU Memory [{tag}]:")
                print(f"- Allocated: {allocated:.2f} GB (actively used by tensors)")
                print(f"- Cached: {cached:.2f} GB (reserved by PyTorch)")
                print(f"- Total GPU Memory: {total_mem:.2f} GB")
                print(f"- Free GPU Memory: {free_mem:.2f} GB")
                print(f"- Used GPU Memory: {used_mem:.2f} GB (overall usage)")
                print(f"- Usage Percentage: {(used_mem / total_mem) * 100:.2f}%\n")

        # log_gpu_memory("Start train_step")

        batch_data = [b.to(self.device) for b in batch_data]
        cell_state, pert_protein, perturbation = batch_data
        batch_dict = {
            'cell_state': cell_state.to(self.device),
            'pert_protein': pert_protein.to(self.device),
            'perturbation': perturbation.to(self.device)
        }


        # log_gpu_memory("After data to GPU")

        flow_outputs = self.flow_model(batch_dict,return_trajectories=True)
        # log_gpu_memory("After flow model")

        # Update train_step to log reg_losses
        flow_loss, reg_losses = self.flow_model.loss_function(flow_outputs, batch_data)
        # log_gpu_memory("After flow loss")
        

        classifier_inputs = self.prepare_classifier_batch(flow_outputs, batch_dict)
        # log_gpu_memory("After classifier prep")

        classifier_outputs = self.classifier(classifier_inputs)
        # log_gpu_memory("After classifier out")

        in_batch_negs, hard_negs, noise_negs = self.train_flow.get_negatives(batch_dict['pert_protein'], k_hard=self.config.batch_size)
        # log_gpu_memory("After negatives")

        contrastive_loss, cont_metrics = self.train_flow.contrastive_loss(
            batch_dict['cell_state'],
            batch_dict['pert_protein'],
            self.config.temperature,
            queue=hard_negs,
            in_batch_negs=in_batch_negs,
            noise_negs=noise_negs
        )

        # log_gpu_memory("After contrastive loss")
        # Before any metrics
        # log_gpu_memory("Before metrics")

        losses = {
            'flow_loss': flow_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            **{f'reg_{k}': v.item() for k,v in reg_losses.items()}
        }
        # log_gpu_memory("After loss metrics")

        losses.update({
            'embedding_metrics/pos_similarity': cont_metrics['pos_sim'],
            'embedding_metrics/neg_similarity': cont_metrics['queue_neg_sim'],
            'embedding_metrics/random_neg_similarity': cont_metrics['in_batch_neg_sim'],
            'embedding_metrics/noise_neg_similarity': cont_metrics['noise_neg_sim'],
            'embedding_metrics/n_random_negs': cont_metrics['n_random_negs'],
            'embedding_metrics/n_queue_negs': cont_metrics['n_queue_negs'],
            'embedding_metrics/n_noise_negs': cont_metrics['n_noise_negs'],
            # Additional debugging metrics
            'embedding_metrics/max_logit': cont_metrics['max_logit'],
            'embedding_metrics/min_logit': cont_metrics['min_logit'],
            'embedding_metrics/mean_logit': cont_metrics['mean_logit'],
            'embedding_metrics/temperature': cont_metrics['temperature']
        })

        # log_gpu_memory("After embedding metrics")

        if is_training:
            total_loss = flow_loss + self.config.contrastive_weight * contrastive_loss
            # log_gpu_memory("Before backward")

            self.flow_optimizer.zero_grad()
            self.classifier_optimizer.zero_grad()
            # log_gpu_memory("After zero_grad")

            total_loss.backward()
            # log_gpu_memory("After backward")

            self.flow_optimizer.step()
            self.classifier_optimizer.step()
            # log_gpu_memory("After optimizer steps")

            self.train_flow.update_queue(batch_dict['pert_protein'])
            # log_gpu_memory("After queue update")

        return losses

    def train(self, train_data, val_data, n_epochs):

        def log_gpu_memory(tag=""):
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9  # Convert to GB
                cached = torch.cuda.memory_reserved() / 1e9  # Convert to GB
                
                free_mem, total_mem = torch.cuda.mem_get_info()  # Get total and free GPU memory
                free_mem, total_mem = free_mem / 1e9, total_mem / 1e9  # Convert to GB
                used_mem = total_mem - free_mem  # Actual used GPU memory

                print(f"\nGPU Memory [{tag}]:")
                print(f"- Allocated: {allocated:.2f} GB (actively used by tensors)")
                print(f"- Cached: {cached:.2f} GB (reserved by PyTorch)")
                print(f"- Total GPU Memory: {total_mem:.2f} GB")
                print(f"- Free GPU Memory: {free_mem:.2f} GB")
                print(f"- Used GPU Memory: {used_mem:.2f} GB (overall usage)")
                print(f"- Usage Percentage: {(used_mem / total_mem) * 100:.2f}%\n")

        # train_flow = TrainFlow(self.config)
        # train_flow.setup_data(train_data, val_data)
        self.train_flow.setup_data(train_data,val_data)  # Use existing instance
        best_loss = float('inf')

        # Analyze embeddings before training
        train_metrics = self.analyze_embeddings(train_data, "train")
        val_metrics = self.analyze_embeddings(val_data, "val")
        wandb.log({
            'embedding_quality/train': train_metrics,
            'embedding_quality/val': val_metrics
        })
        
        for epoch in range(n_epochs):
            self.flow_model.train()
            self.classifier.train()
            train_losses = []
            
            for batch in self.train_flow.train_loader:
                losses = self.train_step(batch, is_training=True)
                train_losses.append(losses)
                
            val_losses = self.validate()
            avg_train = {k:sum(l[k] for l in train_losses)/len(train_losses) for k in train_losses[0]}
            

            # log_gpu_memory("Before TRAIN wandb log")
            wandb.log({
                'epoch': epoch,
                **{f'train_{k}':v for k,v in avg_train.items()},
                **{f'val_{k}':v for k,v in val_losses.items()}
            })
            # log_gpu_memory("After TRAIN wandb log")
            
            val_total = val_losses['flow_loss'] + self.config.contrastive_weight * val_losses['contrastive_loss']
            if val_total < best_loss:
                best_loss = val_total
                self.save_checkpoint(epoch, val_total)
                
        return best_loss
    

    def validate(self):
        self.flow_model.eval()
        self.classifier.eval()
        val_losses = defaultdict(float)
        n_batches = 0
        
        with torch.no_grad():
            for batch in self.train_flow.val_loader:
                losses = self.train_step(batch, is_training=False)
                for k,v in losses.items():
                    val_losses[k] += v
                n_batches += 1
                    
        return {k: v/n_batches for k,v in val_losses.items()}

    def save_checkpoint(self,epoch,loss):
        checkpoint = {
            'epoch':epoch,
            'loss':loss,
            'flow_state':self.flow_model.state_dict(),
            'flow_optimizer':self.flow_optimizer.state_dict(),
            'classifier_state':self.classifier.state_dict(),
            'classifier_optimizer':self.classifier_optimizer.state_dict()
        }
        torch.save(checkpoint,f"checkpoints/joint_checkpoint_{epoch}.pt")

def main():
    config = SimpleNamespace(
        batch_size=8,
        epochs=20,
        learning_rate=1e-4,
        contrastive_weight=0.5,
        queue_size=4096,
        temperature=0.2,
        latent_dim=1024,
        hidden_dim=1024,
        n_heads=8,
        n_layers=3,
        dropout=0.1,
        use_trajectory=True,
        flow_type='exact_ot',
        use_time_embedding=True,
        time_embed_dim=128,
        sigma=0.05,
        use_path_length_reg=True, 
        use_jacobian_reg=True,
        use_feature_mixing=True,
        regularization_weight=0.05,  # For path length & jacobian regularization
        use_layernorm=True    
        )

    
    # Print keys from results
    results = torch.load('results/encoder_results.pt')
    train_data = {
        'cell_state':torch.cat([emb['cell'] for emb in results['train']['embeddings']]),
        'pert_protein':torch.cat([emb['protein'] for emb in results['train']['embeddings']]),
        'perturbation':torch.cat([emb['perturbation'] for emb in results['train']['embeddings']]),
    }
    val_data = {
        'cell_state':torch.cat([emb['cell'] for emb in results['val']['embeddings']]),
        'pert_protein':torch.cat([emb['protein'] for emb in results['val']['embeddings']]),
        'perturbation':torch.cat([emb['perturbation'] for emb in results['val']['embeddings']]),
    }

    trainer = TrainFlowClassifier(config)
    try:
        best_loss = trainer.train(train_data,val_data,config.epochs)
        print(f"Training completed with best loss: {best_loss}")
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e
    finally:
        wandb.finish()

if __name__=="__main__":
    main()
