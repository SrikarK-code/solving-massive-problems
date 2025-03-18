import torch
import torch.nn as nn
import torch.nn.functional as F

class SimilarityTransformer(nn.Module):
    def __init__(self, dim, num_heads=8, ff_dim=512, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        # Cross-attention between modalities
        attn_out, _ = self.cross_attn(
            query=x.unsqueeze(1), 
            key=y.unsqueeze(1), 
            value=y.unsqueeze(1)
        )
        x = self.norm1(x + self.dropout(attn_out.squeeze(1)))
        
        # Self-attention within modality
        self_attn_out, _ = self.self_attn(
            query=x.unsqueeze(1),
            key=x.unsqueeze(1),
            value=x.unsqueeze(1)
        )
        x = self.norm2(x + self.dropout(self_attn_out.squeeze(1)))
        
        # Feedforward
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x
