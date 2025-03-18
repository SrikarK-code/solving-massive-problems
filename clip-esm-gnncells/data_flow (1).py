# data_flow.py
import torch
from typing import Dict, List
from types import SimpleNamespace
from models.flows import TripleFlow, ExactOTFlow, SchrodingerBridgeFlow

class DataFlow:
    def __init__(self):
        self.config = SimpleNamespace(
            use_time_embedding=True,
            time_embed_dim=128,
            latent_dim=1024,  # Matching encoder
            hidden_dim=1024,
            n_layers=3,
            dropout=0.1,
            sigma=0.1,
            use_path_length_reg=True,
            use_jacobian_reg=True,
            use_feature_mixing=True
        )
        
    def compute_flows(self, embeddings: Dict) -> Dict:
        flow_results = {'exact': {}, 'sb': {}}
        
        # Test both flow types
        exact_config = self.config
        exact_config.flow_type = 'exact_ot'
        exact_flow = TripleFlow(exact_config)
        
        sb_config = self.config
        sb_config.flow_type = 'sb'
        sb_flow = TripleFlow(sb_config)
        
        # Compute flows for each dataset's embeddings
        for flow_name, curr_flow in [('exact', exact_flow), ('sb', sb_flow)]:
            flows = curr_flow(embeddings)
            metrics = self.compute_flow_metrics(flows)
            flow_results[flow_name] = {
                'flows': flows,
                'metrics': metrics
            }
            
        return flow_results
    
    def compute_flow_metrics(self, flows: Dict) -> Dict:
        metrics = {}
        for flow_name, (v, xt, t) in flows.items():
            metrics[flow_name] = {
                'vector_field': {
                    'mean': v.mean().item(),
                    'std': v.std().item(),
                    'min': v.min().item(),
                    'max': v.max().item(),
                    'norm': torch.norm(v, dim=1).mean().item()
                },
                'sample_points': {
                    'mean': xt.mean().item(),
                    'std': xt.std().item()
                },
                'time': {
                    'mean': t.mean().item(),
                    'std': t.std().item()
                }
            }
        return metrics
