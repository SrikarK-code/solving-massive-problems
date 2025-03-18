import torch
from trainer import ContrastiveTrainer
from config import config
import wandb


def main():
    # Load real dataset (assumes embeddings are precomputed)
    results = torch.load('results/encoder_results.pt')

    train_data = {
        'cell_state': torch.cat([emb['cell'] for emb in results['train']['embeddings']]),
        'pert_protein': torch.cat([emb['protein'] for emb in results['train']['embeddings']]),
        'perturbation': torch.cat([emb['perturbation'] for emb in results['train']['embeddings']]),
    }

    val_data = {
        'cell_state': torch.cat([emb['cell'] for emb in results['val']['embeddings']]),
        'pert_protein': torch.cat([emb['protein'] for emb in results['val']['embeddings']]),
        'perturbation': torch.cat([emb['perturbation'] for emb in results['val']['embeddings']]),
    }

    trainer = ContrastiveTrainer(config)
    
    try:
        best_loss = trainer.train(train_data, val_data, config.epochs)
        print(f"Training completed with best loss: {best_loss}")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()
