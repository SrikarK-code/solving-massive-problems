# train_flow.py
import torch
import wandb
from train import TrainFlow
from models.flows import TripleFlow
from types import SimpleNamespace
import os

def main():
    """Main training script with configuration and data loading."""
    
    # Create directories for outputs
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('flow_train_logs', exist_ok=True)
    os.makedirs('flow_checkpoints', exist_ok=True)
    
    # Configure model and training parameters
    config = SimpleNamespace(
        # Training parameters
        batch_size=32,
        epochs=20,
        learning_rate=1e-4,
        queue_size=4096,
        temperature=0.2,
        
        # Loss weights
        contrastive_weight=2.0,
        regularization_weight=0.05,
        sequence_weight=1.0,      # For perturbation protein sequence
        topk_weight=1.0,         # For top-k sequence prediction
        
        # Model dimensions
        latent_dim=1024,
        hidden_dim=1024,
        esm_dim=1280,           # ESM embedding dimension
        n_layers=3,
        dropout=0.1,
        n_heads=8,
        
        # Flow parameters
        flow_type='exact_ot',
        use_time_embedding=True,
        time_embed_dim=128,
        sigma=0.05,
        use_path_length_reg=True,
        use_jacobian_reg=True,
        use_feature_mixing=True,
        
        # Sequence modeling
        esm_model_name="esm2_t33_650M_UR50D",
        use_sequence_loss=True,
        k_proteins=10,           # Number of top proteins to predict
        
        # Architecture
        use_layernorm=True      # For projections
    )
    
    # Initialize model and trainer
    model = TripleFlow(config).to('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = TrainFlow(config)
    
    # Load preprocessed data
    results = torch.load('saved_results/encoder_results.pt')
    
    # Prepare training data with consistent naming
    train_data = {
        'cell_state': torch.cat([emb['cell'] for emb in results['train']['embeddings']]),
        'pert_protein': torch.cat([emb['protein'] for emb in results['train']['embeddings']]),
        'pert_effect': torch.cat([emb['perturbation'] for emb in results['train']['embeddings']]),
        'pert_esm_tokens': torch.cat([emb['esm_tokens'] for emb in results['train']['embeddings']]),
        'topk_esm_tokens': torch.cat([emb['top5_tokens'] for emb in results['train']['embeddings']]),
        'pert_esm_embeddings': torch.cat([emb['esm_embeddings'] for emb in results['train']['embeddings']]),
        'topk_esm_embeddings': torch.cat([emb['top5_embeddings'] for emb in results['train']['embeddings']])
    }
    print("Training data shapes:", {k: v.shape for k, v in train_data.items()})
    
    # Prepare validation data
    val_data = {
        'cell_state': torch.cat([emb['cell'] for emb in results['val']['embeddings']]),
        'pert_protein': torch.cat([emb['protein'] for emb in results['val']['embeddings']]),
        'pert_esm_tokens': torch.cat([emb['esm_tokens'] for emb in results['val']['embeddings']]),
        'topk_esm_tokens': torch.cat([emb['top5_tokens'] for emb in results['val']['embeddings']]),
        'pert_esm_embeddings': torch.cat([emb['esm_embeddings'] for emb in results['val']['embeddings']]),
        'topk_esm_embeddings': torch.cat([emb['top5_embeddings'] for emb in results['val']['embeddings']])
    }
    print("Validation data shapes:", {k: v.shape for k, v in val_data.items()})
    
    try:
        # Configure wandb logging
        wandb.init(
            project="protein-flow",
            config={
                **vars(config),
                'train_size': len(train_data['cell_state']),
                'val_size': len(val_data['cell_state']),
                'esm_dim': train_data['pert_esm_embeddings'].shape[-1]
            }
        )
        
        # Train model
        best_val_loss = trainer.train(model, train_data, val_data)
        print(f"Training completed with best validation loss: {best_val_loss}")
        
        # Save final model state
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'best_val_loss': best_val_loss
        }, 'flow_checkpoints/final_model.pt')
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        # Save interrupted state
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'interrupted': True
        }, 'flow_checkpoints/interrupted_model.pt')
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise e
        
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()