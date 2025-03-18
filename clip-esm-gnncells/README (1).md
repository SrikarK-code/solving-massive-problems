# TripleFlow: Optimal Transport-Based Flow Model for Protein Perturbation Prediction

## Table of Contents
1. [Overview](#overview)
2. [Mathematical Framework](#mathematical-framework)
3. [Architecture](#architecture)
4. [Implementation Details](#implementation-details)
5. [Training Pipeline](#training-pipeline)
6. [Inference Pipeline](#inference-pipeline)
7. [Usage Guide](#usage-guide)
8. [Configuration](#configuration)
9. [Appendix](#appendix)

## Overview

TripleFlow is a flow-based deep learning model that integrates three biological modalities:
- Cell states (via GNN encodings)
- Perturbation effects (via ESM embeddings)
- Protein information (via ESM embeddings)

### Key Features
1. Multi-task prediction capabilities:
   - Perturbation protein sequences
   - Top-k affected proteins
   - Vector field learning
2. Multiple learning objectives:
   - Flow matching loss
   - Contrastive learning
   - Sequence prediction
   - Regularization
3. Efficient processing:
   - Batch inference
   - Memory queue
   - Masked computations

## Mathematical Framework

### 1. Conditional Flow Matching (CFM)
The core objective is learning vector fields that transport probability distributions:

```math
dx/dt = v_θ(x,t)
x(0) ~ p₀ (source)
x(1) ~ p₁ (target)
```

The CFM loss is defined as:
```math
L(θ) = E_{t~U(0,1), x₀~p₀, x₁~p₁}[‖v_θ(x_t,t) - u_t(x_t|x₀,x₁)‖²]
```
where:
- x_t = tx₁ + (1-t)x₀ + σε, ε ~ N(0,I)
- u_t is the target vector field
- θ represents model parameters

### 2. Optimal Transport (OT)
For distributions μ and ν:
```math
W₂(μ,ν) = inf_{π ∈ Π(μ,ν)} ∫∫ ‖x-y‖² dπ(x,y)
```
where Π(μ,ν) represents joint distributions with marginals μ and ν.

### 3. Loss Components

#### Flow Matching Loss
```math
L_flow = E[‖v_θ(x_t,t) - u_t(x_t|x₀,x₁)‖²]
```

#### Contrastive Loss
```math
L_cont = -log(exp(sim(h₁,h₂)/τ) / Σᵢexp(sim(h₁,hᵢ)/τ))
```

#### Sequence Loss (Masked)
```math
L_seq = -Σᵢ mᵢ log p(yᵢ|xᵢ)
```
where mᵢ is the mask for non-padding tokens.

#### Regularization
```math
L_reg = λ₁E[‖v_θ‖²] + λ₂E[‖∇ₓv_θ‖²]
```

### 4. Combined Objective
```math
L_total = L_flow + α L_cont + β L_seq + γ L_reg
```
where α, β, and γ are weighting coefficients.




## Architecture

### 1. Core Components

```
TripleFlow
├── Flow Networks
│   ├── cell_to_protein_flow
│   ├── cell_to_pert_flow
│   ├── pert_to_protein_flow
│   └── cell_to_topk_flow
├── ESM Projections
│   ├── pert_esm_proj (Linear + LayerNorm)
│   └── topk_esm_proj (Linear + LayerNorm)
└── Feature Mixing
    └── create_projection_layer (Custom MLP)
```

### 2. Data Structures

#### Input Embeddings
```python
{
    'cell_state': tensor[batch_size, latent_dim],        # 1024
    'pert_protein': tensor[batch_size, latent_dim],      # 1024
    'pert_effect': tensor[batch_size, latent_dim],       # 1024
    'pert_esm_tokens': tensor[batch_size, seq_len],      # Vocab indices
    'pert_esm_embeddings': tensor[batch_size, seq_len, esm_dim],  # 1280
    'topk_esm_tokens': tensor[batch_size, k, seq_len],
    'topk_esm_embeddings': tensor[batch_size, k, seq_len, esm_dim]
}
```

#### Flow Outputs
```python
{
    'cell_to_protein_flow': (v, xt, t, ut),  # Vector field components
    'cell_to_pert_flow': (v, xt, t, ut),
    'pert_to_protein_flow': (v, xt, t, ut),
    'cell_to_topk_flow': (v, xt, t, ut),
    'pert_protein_esm': tensor[batch_size, esm_dim],
    'topk_protein_esm': tensor[batch_size, k, esm_dim]
}
```

### 3. Projection Layers

#### Custom Projection (Complex Transformations)
```python
create_projection_layer(
    d_in: int,
    d_out: int,
    dropout: float = 0.1,
    use_layernorm: bool = True
) -> nn.Sequential:
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
```

#### ESM Projection (Simple Transformations)
```python
esm_proj = nn.Sequential(
    nn.Linear(latent_dim, esm_dim),
    nn.LayerNorm(esm_dim)
)
```

### 4. Flow Implementation

```python
class OTFlow(nn.Module):
    """Base class for all flow transformations."""
    def __init__(self, config):
        self.net = self._build_network()
        self.time_encoder = self._build_time_encoder()
    
    def forward(self, source, target, return_regularization=False):
        # 1. Sample time points
        t = torch.rand(len(source), 1)
        
        # 2. Compute intermediate points
        xt = t * target + (1-t) * source + self.sigma * torch.randn_like(source)
        
        # 3. Compute target velocities
        ut = target - source
        
        # 4. Predict vector field
        v = self.net(torch.cat([xt, self.time_encoder(t)], dim=-1))
        
        if return_regularization:
            reg = self._compute_regularization(v, xt)
            return v, xt, t, ut, reg
        return v, xt, t, ut
```

### 5. Memory Management

#### Queue Structure
```python
class MemoryQueue:
    def __init__(self, size, dim):
        self.queue = torch.zeros(size, dim)  # [queue_size, latent_dim]
        self.ptr = 0
        self.full = False
        
    def update(self, embeddings):
        batch_size = len(embeddings)
        
        # Circular update
        if self.ptr + batch_size > len(self.queue):
            # Handle wrap-around
            first_part = len(self.queue) - self.ptr
            self.queue[self.ptr:] = embeddings[:first_part]
            self.queue[:batch_size-first_part] = embeddings[first_part:]
        else:
            self.queue[self.ptr:self.ptr+batch_size] = embeddings
            
        self.ptr = (self.ptr + batch_size) % len(self.queue)
        if not self.full and self.ptr == 0:
            self.full = True
```




## Implementation Details

### 1. Data Processing Pipeline

```python
# Starting from raw data:
Raw Data
  ↓
Data Encoder (data_encoder.py)
  ├── Cell State Processing
  │   ├── GNN encoding
  │   ├── Pseudotime integration
  │   └── Graph metrics computation
  │
  ├── Protein Processing
  │   ├── ESM sequence tokenization
  │   ├── Token-level embeddings
  │   └── Padding handling
  │
  └── Top-k Processing
      ├── Multiple sequence handling
      ├── Batch padding
      └── ESM embedding generation

# Output format for each dataset:
{
    'cell': cell_emb,               # [n_cells, latent_dim]
    'protein': protein_emb,         # [n_cells, latent_dim]
    'perturbation': pert_emb,       # [n_cells, latent_dim]
    'esm_tokens': tokens,           # [n_cells, seq_len]
    'esm_embeddings': embeddings,   # [n_cells, seq_len, esm_dim]
    'top5_tokens': topk_tokens,     # [n_cells, k, seq_len]
    'top5_embeddings': topk_emb     # [n_cells, k, seq_len, esm_dim]
}
```

### 2. Model Components Interaction

```python
# Flow component interactions:
Cell State ─────┬─────> Perturbation ─────> Protein
                │                            ↑
                └────────────────────────────┘

# Data flow through components:
input_cell_state
  ↓
cell_to_protein_flow
  ├── Direct path: cell → protein
  └── Indirect path: cell → pert → protein
  ↓
ESM Projections
  ├── pert_protein_esm
  └── topk_protein_esm
  ↓
Sequence Generation
```

## Training Pipeline

### 1. Training Loop Structure

```python
def train_epoch(model, optimizer):
    for batch in dataloader:
        # 1. Forward Pass
        cell_state, pert_protein, pert_effect, \
        pert_esm_tokens, topk_esm_tokens, \
        pert_esm_embeddings, topk_esm_embeddings = batch

        # 2. Get Flow Predictions
        flows, reg_losses = model({
            'cell_state': cell_state,
            'pert_protein': pert_protein,
            'pert_effect': pert_effect,
            'topk_esm_embeddings': topk_esm_embeddings
        })

        # 3. Compute Losses
        loss = 0.0
        
        # Flow matching losses
        for flow_name, (v, xt, t, ut) in flows.items():
            if 'esm' not in flow_name:
                loss += flow_matching_loss(v, target_v)
                loss += config.reg_weight * reg_losses[flow_name]
        
        # Contrastive loss
        loss += config.contrastive_weight * contrastive_loss(
            cell_state, pert_protein, queue
        )
        
        # Sequence losses
        if 'pert_protein_esm' in flows:
            loss += config.sequence_weight * masked_sequence_loss(
                esm_head(flows['pert_protein_esm']),
                pert_esm_tokens,
                padding_mask
            )
            
        if 'topk_protein_esm' in flows:
            loss += config.topk_weight * masked_sequence_loss(
                esm_head(flows['topk_protein_esm']),
                topk_esm_tokens,
                topk_padding_mask
            )

        # 4. Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 5. Update Queue
        update_queue(pert_protein)
```

### 2. Inference Pipeline

```python
# Two-level inference system:

# Level 1: Raw Predictions (TripleFlow)
def generate(cell_state):
    """Generate raw predictions from cell state."""
    with torch.no_grad():
        # Get flows
        flows = model({'cell_state': cell_state})
        
        # Extract states
        pert_state = flows['cell_to_protein_flow'][1]
        topk_state = flows['cell_to_topk_flow'][1]
        
        # Project to ESM space
        pert_esm = model.pert_esm_proj(pert_state)
        topk_esm = model.topk_esm_proj(topk_state)
        
        return pert_state, pert_esm, topk_state, topk_esm

# Level 2: Full Pipeline (TrainFlow)
def inference_step(model, cell_states):
    """Full prediction pipeline including sequences."""
    outputs = {}
    
    # Get raw predictions
    flows = model({'cell_state': cell_states})
    
    # Generate sequences for perturbation protein
    if 'pert_protein_esm' in flows:
        logits = esm_head(flows['pert_protein_esm'])
        outputs['pert_protein_logits'] = logits
        outputs['pert_protein_state'] = flows['pert_protein_esm']
    
    # Generate sequences for top-k proteins
    if 'topk_protein_esm' in flows:
        logits = esm_head(flows['topk_protein_esm'])
        outputs['topk_protein_logits'] = \
            logits.view(-1, config.k_proteins, esm_vocab_size)
        outputs['topk_protein_state'] = flows['topk_protein_esm']
    
    return outputs

# Batch processing for large datasets
def batch_inference(model, cell_states, batch_size=32):
    """Process large datasets in batches."""
    all_outputs = {
        'pert_sequences': [],
        'pert_confidences': [],
        'topk_sequences': [],
        'topk_confidences': []
    }
    
    for i in range(0, len(cell_states), batch_size):
        batch = cell_states[i:i + batch_size]
        outputs = inference_step(model, batch)
        
        # Process perturbation predictions
        if 'pert_protein_logits' in outputs:
            seqs, conf = generate_sequences(
                outputs['pert_protein_logits']
            )
            all_outputs['pert_sequences'].extend(seqs)
            all_outputs['pert_confidences'].extend(conf)
        
        # Process top-k predictions
        if 'topk_protein_logits' in outputs:
            seqs, conf = generate_sequences(
                outputs['topk_protein_logits']
            )
            all_outputs['topk_sequences'].extend(seqs)
            all_outputs['topk_confidences'].extend(conf)
    
    return all_outputs
```
