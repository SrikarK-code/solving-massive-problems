# Protein Flow Model Documentation

## Table of Contents
1. [Naming Conventions](#naming-conventions)
2. [Data Flow & Tensor Shapes](#data-flow--tensor-shapes)
3. [Loss Computation Pipeline](#loss-computation-pipeline)
4. [Model Architecture](#model-architecture)

## Naming Conventions

### Data Keys
```python
# Core Embeddings
'cell_state'             # Cell state embeddings [batch_size, latent_dim]
'pert_protein'           # Perturbation protein embeddings [batch_size, latent_dim]
'pert_effect'           # Perturbation effect embeddings [batch_size, latent_dim]

# ESM Related
'pert_esm_tokens'        # ESM tokens for perturbation [batch_size, seq_len]
'pert_esm_embeddings'    # ESM embeddings for perturbation [batch_size, seq_len, esm_dim]
'topk_esm_tokens'        # ESM tokens for top-k proteins [batch_size, k, seq_len]
'topk_esm_embeddings'    # ESM embeddings for top-k proteins [batch_size, k, seq_len, esm_dim]
```

### Flow Names
```python
# Core Flows
'cell_to_protein_flow'   # Cell → Protein transformation
'cell_to_pert_flow'      # Cell → Perturbation transformation
'pert_to_protein_flow'   # Perturbation → Protein transformation
'cell_to_topk_flow'      # Cell → Top-k proteins transformation

# ESM Projections
'pert_protein_esm'       # Protein ESM space projection
'topk_protein_esm'       # Top-k proteins ESM space projection
```

### Loss Names
```python
# Flow Losses
'flow_loss_cell_to_protein'  # Flow matching loss for main prediction
'flow_loss_cell_to_pert'     # Flow matching loss for perturbation
'flow_loss_cell_to_topk'     # Flow matching loss for top-k

# Regularization Losses
'reg_loss_cell_to_protein'   # Regularization for main flow
'reg_loss_cell_to_pert'      # Regularization for pert flow
'reg_loss_cell_to_topk'      # Regularization for top-k flow

# Sequence Losses
'pert_sequence_loss'         # Sequence loss for perturbation protein
'topk_sequence_loss'         # Sequence loss for top-k proteins

# Other Losses
'contrastive_loss'           # Contrastive learning loss
```

## Data Flow & Tensor Shapes

### Input Pipeline
```
Data Encoder (data_encoder.py)
↓
[batch_size, latent_dim] Cell State
↓
TripleFlow Model
↓
1. Flow Transformations
2. ESM Projections
3. Sequence Generation
```

### Tensor Shapes Through Pipeline
```python
# Input Shapes
cell_state:          [batch_size, latent_dim]        # e.g., [32, 1024]
pert_protein:        [batch_size, latent_dim]        # e.g., [32, 1024]
pert_effect:         [batch_size, latent_dim]        # e.g., [32, 1024]

# ESM Related Shapes
pert_esm_tokens:     [batch_size, seq_len]           # e.g., [32, 512]
topk_esm_tokens:     [batch_size, k, seq_len]        # e.g., [32, 10, 512]
pert_esm_embeddings: [batch_size, seq_len, esm_dim]  # e.g., [32, 512, 1280]
topk_esm_embeddings: [batch_size, k, seq_len, esm_dim] # e.g., [32, 10, 512, 1280]

# Flow Outputs
flow_outputs:        (v, xt, t, ut)
- v:  [batch_size, latent_dim]  # Vector field
- xt: [batch_size, latent_dim]  # Current point
- t:  [batch_size, 1]           # Time parameter
- ut: [batch_size, latent_dim]  # Target velocity

# ESM Projections
pert_protein_esm:    [batch_size, esm_dim]           # e.g., [32, 1280]
topk_protein_esm:    [batch_size, k, esm_dim]        # e.g., [32, 10, 1280]
```

## Loss Computation Pipeline

### 1. Flow Matching Losses
```python
# For each flow:
1. Compute vector field: v = flow_network(xt)
2. Compare with target: loss = MSE(v, target_v)
3. Add regularization: loss += reg_weight * (path_length + jacobian)
```

### 2. Contrastive Loss
```python
1. Get negatives:
   - In-batch negatives (random permutation)
   - Queue negatives (memory bank)
   - Hard negatives (similarity-based)

2. Compute similarities:
   - Positive pairs: sim(cell_state, pert_protein)
   - Negative pairs: sim(cell_state, negatives)

3. Weight and combine:
   - Local weight: 0.3 for in-batch
   - Global weight: 0.7 for queue
```

### 3. Sequence Losses
```python
1. Generate sequences:
   - Project to ESM space
   - Apply ESM head
   - Get logits

2. Mask padding tokens:
   - Create padding mask
   - Only compute loss on real tokens

3. Compute cross entropy:
   - For perturbation protein
   - For each of the top-k proteins
```

### Loss Weights (Default)
```python
config = {
    'contrastive_weight': 2.0,    # Weight for contrastive loss
    'regularization_weight': 0.05, # Weight for flow regularization
    'sequence_weight': 1.0,        # Weight for sequence loss
    'topk_weight': 1.0            # Weight for top-k sequence loss
}
```

## Model Architecture

### TripleFlow Components
```
1. Flow Networks
   ├── cell_to_protein (main prediction)
   ├── cell_to_pert (perturbation prediction)
   ├── pert_to_protein (perturbation flow)
   └── cell_to_topk (top-k prediction)

2. Feature Mixing
   ├── Custom projection layer
   └── Skip connections

3. ESM Projections
   ├── Linear + LayerNorm for protein
   └── Linear + LayerNorm for top-k
```

