import re
import dask.dataframe as dd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Recap
import networkx as nx
import dgl
from dgl.nn import GraphConv, GATv2Conv, GINConv, EdgeConv
from torch_geometric.nn import TransformerConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, mean_squared_error
from hdbscan import HDBSCAN
import logging

logging.basicConfig(level=logging.INFO)

class CompleteBigSMILESParser:
    def __init__(self):
        self.stochastic_object_pattern = re.compile(r'\{(.*?)\}')
        self.terminal_descriptor_pattern = re.compile(r'\{\[(.*?)\](.*?)\[(.*?)\]\}')
        self.aa_type_descriptor = re.compile(r'\[\$(\d*)\]')
        self.ab_type_descriptor_left = re.compile(r'\[<(\d*)\]')
        self.ab_type_descriptor_right = re.compile(r'\[>(\d*)\]')
        self.star_descriptor = re.compile(r'\[\*\]')
        self.empty_descriptor = re.compile(r'\[\]')
        self.any_bonding_descriptor = re.compile(r'\[\$\d*\]|\[<\d*\]|\[>\d*\]|\[\*\]|\[\]')
        self.end_group_pattern = re.compile(r';(.*?)(?:\[|$)')
        self.repeat_unit_pattern = re.compile(r'(?:\[.*?\])?(.*?)(?:\[.*?\])?(?:,|\]|;|$)')
        self.ladder_bond_descriptor = re.compile(r'\[([$<>])(\d*)\[([$<>])(\d*)\](\d*)\]')
        self.fragment_placeholder = re.compile(r'\[#([^]]*)\]')
        self.fragment_definition = re.compile(r'\.{#([^=]*)=(.*?)}')
        self.nested_stochastic_pattern = re.compile(r'\{[^{}]*(\{.*?\})[^{}]*\}')
        self.bond_symbol_pattern = re.compile(r'(-|=|#|:|\$|/|\\)')
        self.atomic_symbol_pattern = re.compile(r'\[([^]]*)\]')
        self.ring_closure_pattern = re.compile(r'(\d+)')
        self.stereochemistry_pattern = re.compile(r'(@{1,2})')
        self.motif_with_bonding_pattern = re.compile(r'(\[[$<>*]\d*\])([^[]+?)(\[[$<>*]\d*\])')
        self.junction_point_pattern = re.compile(r'(\[[$<>]\d*\])([^[]+?)(\[[$<>]\d*\])')
        self.simplified_vinyl_pattern = re.compile(r'\{[^\[\]]+?,[^\[\]]+?\}')
        self.simplified_amino_acid_pattern = re.compile(r'\{[<>].*?[<>]\}')
        self.charge_pattern = re.compile(r'\[([^]]+?)([+-]\d*)\]')
        self.isotope_pattern = re.compile(r'\[(\d+)([^]]+?)\]')
        self.tacticity_pattern = re.compile(r'\[C@{1,2}H\]')
        self.cis_trans_pattern = re.compile(r'[/\\]C=C[/\\]')
        self.ionic_compound_pattern = re.compile(r'(\[[^]]+?\][^.]*?\.[^[]*?\[[^]]+?\])')

    def smiles_to_bigsmiles(self, smiles):
        """Convert SMILES to BigSMILES using Recap decomposition for repeat units."""
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if not mol:
                raise ValueError("Invalid SMILES")
            recap = Recap.RecapDecompose(mol)
            repeat_units = [Chem.MolToSmiles(node.mol) for node in recap.GetLeaves()]
            if not repeat_units:
                return f"{{{smiles}}}"  # Fallback to basic stochastic object
            bigsmiles = "{" + ",".join(f"[<]{ru}[>]" for ru in repeat_units) + "}"
            end_groups = self.end_group_pattern.findall(smiles)
            if end_groups:
                bigsmiles += ";" + ";".join(end_groups)
            return bigsmiles
        except Exception as e:
            logging.warning(f"SMILES to BigSMILES conversion failed for {smiles}: {str(e)}")
            return f"{{{smiles}}}"

    def parse(self, bigsmiles):
        try:
            fragments = {m.group(1): m.group(2) for m in self.fragment_definition.finditer(bigsmiles)}
            for placeholder, definition in fragments.items():
                bigsmiles = bigsmiles.replace(f"[#{placeholder}]", definition)
            
            while self.nested_stochastic_pattern.search(bigsmiles):
                nested = self.nested_stochastic_pattern.search(bigsmiles).group(1)
                flattened = self._flatten_nested(nested)
                bigsmiles = bigsmiles.replace(nested, flattened)
            
            motifs = self._extract_motifs(bigsmiles)
            end_groups = self.end_group_pattern.findall(bigsmiles)
            bonding_descriptors = self.any_bonding_descriptor.findall(bigsmiles)
            motif_graphs = []
            for m in motifs:
                g = self._motif_to_graph(m, bonding_descriptors, bigsmiles)
                if g:
                    motif_graphs.append(g)
            if not motif_graphs:
                raise ValueError("No valid motifs parsed")
            return self._build_hierarchical_graph(motif_graphs, end_groups, bonding_descriptors, bigsmiles)
        except Exception as e:
            logging.error(f"Parsing failed for {bigsmiles}: {str(e)}")
            return None

    def _flatten_nested(self, nested):
        motifs = self.repeat_unit_pattern.findall(nested)
        return ','.join(m for m in motifs if m.strip())

    def _extract_motifs(self, bigsmiles):
        stochastic_content = self.stochastic_object_pattern.search(bigsmiles)
        if not stochastic_content:
            return [bigsmiles]
        content = stochastic_content.group(1)
        return [m.group(1).strip() for m in self.repeat_unit_pattern.finditer(content) if m.group(1).strip()]

    def _motif_to_graph(self, motif, bonding_descriptors, bigsmiles):
        mol = Chem.MolFromSmiles(motif, sanitize=False)
        if not mol:
            return None
        g = nx.Graph()
        atom_features = []
        junction_atoms = self._identify_junction_atoms(motif, bonding_descriptors, bigsmiles)
        
        for atom in mol.GetAtoms():
            charge_match = self.charge_pattern.search(motif)
            isotope_match = self.isotope_pattern.search(motif)
            charge = int(charge_match.group(2)) if charge_match else 0
            isotope = int(isotope_match.group(1)) if isotope_match else 0
            stereo = 1 if (self.stereochemistry_pattern.search(motif) or self.tacticity_pattern.search(motif)) else 0
            feat = torch.tensor([
                atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
                atom.GetNumImplicitHs(), atom.GetIsAromatic(), atom.GetChiralTag(),
                atom.GetHybridization().real, atom.IsInRing(), 
                1 if atom.GetIdx() in junction_atoms else 0,
                atom.GetTotalValence(), charge, isotope, stereo
            ], dtype=torch.float32)
            g.add_node(atom.GetIdx(), feat=feat)
            atom_features.append(feat)
        for bond in mol.GetBonds():
            cis_trans = 1 if self.cis_trans_pattern.search(motif) else 0
            edge_feat = torch.tensor([
                bond.GetBondTypeAsDouble(), bond.GetStereo().real, bond.IsInRing(),
                bond.GetBeginAtomIdx() in junction_atoms or bond.GetEndAtomIdx() in junction_atoms,
                cis_trans
            ], dtype=torch.float32)
            g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), feat=edge_feat)
        dgl_g = dgl.from_networkx(g, node_attrs=['feat'], edge_attrs=['feat'])
        dgl_g.ndata['feat'] = torch.stack(atom_features)
        return dgl_g

    def _identify_junction_atoms(self, motif, bonding_descriptors, bigsmiles):
        mol = Chem.MolFromSmiles(motif, sanitize=False)
        if not mol:
            return set()
        junction_atoms = set()
        motif_start = bigsmiles.find(motif)
        for desc in bonding_descriptors:
            desc_pos = bigsmiles.find(desc, motif_start)
            if desc_pos != -1 and desc_pos < motif_start + len(motif):
                # Precise positional mapping using atom index approximation
                relative_pos = (desc_pos - motif_start) / len(motif)
                atom_idx = min(int(relative_pos * mol.GetNumAtoms()), mol.GetNumAtoms() - 1)
                junction_atoms.add(atom_idx)
        return junction_atoms

    def _build_hierarchical_graph(self, motif_graphs, end_groups, bonding_descriptors, bigsmiles):
        chain_graph = dgl.DGLGraph()
        chain_features = []
        topology = self._infer_topology(motif_graphs, bonding_descriptors, bigsmiles)
        
        for i, mg in enumerate(motif_graphs):
            chain_graph.add_nodes(1)
            motif_mean = torch.mean(mg.ndata['feat'], dim=0)
            pos_encoding = torch.tensor([np.sin(i / 10000 ** (2 * j / 512)) for j in range(512)], dtype=torch.float32)
            chain_features.append(torch.cat([motif_mean, pos_encoding[:13]]))
            chain_graph.ndata['motif'] = [mg] * chain_graph.num_nodes()
            
            for src, dst, bond_type, prob in topology:
                if src == i and dst < len(motif_graphs):
                    chain_graph.add_edges(src, dst, {'type': torch.tensor([bond_type]), 'prob': torch.tensor([prob])})
                    chain_graph.add_edges(dst, src, {'type': torch.tensor([bond_type]), 'prob': torch.tensor([prob])})
        
        chain_graph.ndata['feat'] = torch.stack(chain_features)
        chain_graph.ndata['end_group'] = torch.tensor([1 if i < len(end_groups) else 0 for i in range(len(motif_graphs))], dtype=torch.float32)
        return chain_graph

    def _infer_topology(self, motif_graphs, bonding_descriptors, bigsmiles):
        topology = []
        ladder_matches = self.ladder_bond_descriptor.findall(bigsmiles)
        descriptor_map = {i: [] for i in range(len(motif_graphs))}
        motif_starts = [bigsmiles.find(m) for m in self._extract_motifs(bigsmiles)]
        
        for desc in bonding_descriptors:
            desc_pos = bigsmiles.find(desc)
            if desc_pos != -1:
                for i, start in enumerate(motif_starts):
                    if start <= desc_pos < (start + len(self._extract_motifs(bigsmiles)[i]) if i < len(motif_starts) else float('inf')):
                        descriptor_map[i].append(desc)
                        break
        
        for i in range(len(motif_graphs)):
            descriptors = descriptor_map.get(i, [])
            for desc in descriptors:
                if '<' in desc or '>' in desc:
                    if i + 1 < len(motif_graphs):
                        topology.append((i, i + 1, 1.0, 0.95))  # AB-type linear
                elif '$' in desc:
                    for j in range(len(motif_graphs)):
                        if i != j and '$' in descriptor_map.get(j, []):
                            topology.append((i, j, 1.0, 0.90))  # AA-type branching
        for outer, outer_id, inner, inner_id, group_id in ladder_matches:
            topology.append((int(group_id) - 1, int(group_id), 1.0, 0.98))  # Ladder connection
        return topology

class GraphSAINTSampler:
    def __init__(self, graph, batch_size=1000):
        self.graph = graph
        self.batch_size = batch_size
        self.edge_probs = torch.softmax(graph.edata['prob'], dim=0)

    def sample_subgraph(self):
        nodes = torch.multinomial(torch.ones(self.graph.num_nodes()), self.batch_size, replacement=False)
        subgraph = dgl.node_subgraph(self.graph, nodes)
        edge_mask = torch.bernoulli(self.edge_probs[:subgraph.num_edges()])
        subgraph = dgl.edge_subgraph(subgraph, edge_mask.bool())
        return subgraph

class HierarchicalGNN(nn.Module):
    def __init__(self, atom_dim=13, motif_dim=256, chain_dim=512, prop_dim=29):
        super().__init__()
        # Edge-Aware Experts
        self.atom_conv = GraphConv(atom_dim, motif_dim)
        self.motif_gat = GATv2Conv(motif_dim, motif_dim // 4, num_heads=4, feat_drop=0.1)
        self.chain_gin = GINConv(nn.Linear(motif_dim, chain_dim))
        self.edge_conv = EdgeConv(chain_dim, chain_dim)  # Edge-focused expert
        self.attn_transformer = TransformerConv(chain_dim, chain_dim // 2, heads=4, edge_dim=5)
        
        # Deep Bidirectional Attention
        self.cross_attn_layers = nn.ModuleList([nn.MultiheadAttention(chain_dim, num_heads=4) for _ in range(3)])
        
        # Virtual Node
        self.virtual_node = nn.Parameter(torch.randn(1, chain_dim))
        
        # Gating for Mixture of Experts
        self.gate = nn.Linear(chain_dim * 4, 4)  # Four experts
        
        # Property Prediction Head
        self.prop_head = nn.Sequential(
            nn.Linear(chain_dim * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, prop_dim)
        )

    def forward(self, graph):
        motif_feats = []
        edge_feats_list = []
        for mg in graph.ndata['motif']:
            atom_h = F.relu(self.atom_conv(mg, mg.ndata['feat'], edge_weight=mg.edata['feat']))
            motif_h = self.motif_gat(mg, atom_h).flatten(1)
            motif_feats.append(motif_h.mean(dim=0))
            edge_feats_list.append(mg.edata['feat'].mean(dim=0))
        motif_feats = torch.stack(motif_feats)
        edge_feats = torch.stack(edge_feats_list)
        
        # Expert Pathways with Edge Integration
        chain_h_gin = self.chain_gin(graph, motif_feats)
        chain_h_edge = self.edge_conv(graph, chain_h_gin)
        chain_h_transformer = self.attn_transformer(graph, chain_h_gin, edge_attr=graph.edata['feat'])
        chain_h = chain_h_transformer.flatten(1) + self.virtual_node.expand(graph.num_nodes(), -1)
        
        # Deep Bidirectional Attention with Edge Context
        for attn_layer in self.cross_attn_layers:
            motif_query = motif_feats.unsqueeze(0)
            chain_key_value = chain_h.unsqueeze(0)
            attn_output, _ = attn_layer(motif_query, chain_key_value, chain_key_value)
            chain_h = chain_h + attn_output.squeeze(0)
        
        # Gating over Experts
        gate_input = torch.cat([chain_h_gin, chain_h_edge, chain_h_transformer.flatten(1), chain_h], dim=1)
        gate_weights = F.softmax(self.gate(gate_input), dim=1)
        chain_h = (gate_weights[:, 0:1] * chain_h_gin + 
                   gate_weights[:, 1:2] * chain_h_edge + 
                   gate_weights[:, 2:3] * chain_h_transformer.flatten(1) + 
                   gate_weights[:, 3:4] * chain_h)
        
        graph_emb = torch.cat([chain_h.mean(dim=0, keepdim=True), chain_h.max(dim=0)[0].unsqueeze(0)], dim=1)
        prop_pred = self.prop_head(graph_emb)
        return chain_h, prop_pred

class ContrastiveCHGAE(nn.Module):
    def __init__(self, num_ensembles=3):
        super().__init__()
        self.encoders = nn.ModuleList([HierarchicalGNN() for _ in range(num_ensembles)])
        self.projection = nn.Sequential(
            nn.Linear(512 * num_ensembles, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def nt_xent_loss(self, emb1, emb2, prop1, prop2):
        proj1 = self.projection(emb1)
        proj2 = self.projection(emb2)
        sim_matrix = F.cosine_similarity(proj1.unsqueeze(1), proj2.unsqueeze(0), dim=-1) / 0.07
        prop_diff = torch.norm(prop1 - prop2, p=2, dim=-1)
        weights = torch.exp(-prop_diff / prop_diff.mean())
        labels = torch.arange(len(emb1)).to(emb1.device)
        return F.cross_entropy(sim_matrix * weights, labels)

    def forward(self, graph):
        embeddings = []
        prop_preds = []
        for encoder in self.encoders:
            emb, prop = encoder(graph)
            embeddings.append(emb)
            prop_preds.append(prop)
        combined_emb = torch.cat(embeddings, dim=-1)
        avg_prop = torch.mean(torch.stack(prop_preds), dim=0)
        return combined_emb, avg_prop

def load_polyone_data(data_path="polyOne_*.parquet"):
    ddf = dd.read_parquet(data_path, engine="pyarrow")
    df = ddf.compute()
    smiles = df['smiles'].values
    prop_cols = [col for col in df.columns if col != 'smiles']
    props = df[prop_cols].values
    scaler = StandardScaler()
    props = scaler.fit_transform(props)
    return smiles, props, scaler

def cluster_split(smiles, props):
    hdbscan = HDBSCAN(min_cluster_size=200, min_samples=5, cluster_selection_epsilon=0.3)
    clusters = hdbscan.fit_predict(props)
    unique_clusters = np.unique(clusters[clusters >= 0])
    train_idx, val_idx = [], []
    for cluster in unique_clusters:
        idx = np.where(clusters == cluster)[0]
        np.random.shuffle(idx)
        split = int(0.8 * len(idx))
        train_idx.extend(idx[:split])
        val_idx.extend(idx[split:])
    return smiles[train_idx], props[train_idx], smiles[val_idx], props[val_idx]

def preprocess_dataset(smiles, batch_size=1000):
    parser = CompleteBigSMILESParser()
    bigsmiles_list = [parser.smiles_to_bigsmiles(s) for s in smiles]
    graphs = []
    for i in range(0, len(bigsmiles_list), batch_size):
        batch_bs = bigsmiles_list[i:i+batch_size]
        batch_graphs = [parser.parse(bs) for bs in batch_bs]
        graphs.extend([g for g in batch_graphs if g is not None])
    return graphs

def chemistry_aware_augmentation(graph):
    g = graph.clone()
    edge_mask = torch.ones(g.num_edges(), dtype=torch.bool)
    for i, edge in enumerate(g.edges()):
        edge_feat = g.edata['feat'][i]
        if edge_feat[-1] == 0 and edge_feat[0] == 1.0 and torch.rand(1) > 0.25:  # Preserve cis/trans and non-single bonds
            edge_mask[i] = False
    g.remove_edges(torch.where(~edge_mask)[0])
    return g

def substructure_masking(graph):
    g = graph.clone()
    junction_mask = g.ndata['feat'][:, 8] == 0  # Don't mask junction atoms
    node_mask = torch.rand(g.num_nodes()) > 0.2
    node_mask = node_mask & junction_mask
    g.ndata['feat'][~node_mask] = 0
    return g

def train_model(train_graphs, train_props, val_graphs, val_props, epochs=100, batch_size=1000):
    model = ContrastiveCHGAE(num_ensembles=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    full_graph = dgl.batch(train_graphs)
    sampler = GraphSAINTSampler(full_graph, batch_size)
    best_val_loss = float('inf')
    
    for phase, epoch_range in [("pretrain", range(30)), ("finetune", range(30, epochs))]:
        for epoch in epoch_range:
            model.train()
            total_loss = 0
            prop_mse = 0
            num_batches = max(1, len(train_graphs) // batch_size)
            for _ in range(num_batches):
                batch_graph = sampler.sample_subgraph()
                emb, pred_props = model(batch_graph)
                aug_graph = chemistry_aware_augmentation(batch_graph)
                mask_graph = substructure_masking(batch_graph)
                aug_emb, _ = model(aug_graph)
                mask_emb, _ = model(mask_graph)
                batch_props = torch.tensor(train_props[:len(emb)], dtype=torch.float32)
                
                contrastive_loss = model.nt_xent_loss(emb, aug_emb, pred_props, batch_props)
                mask_loss = F.mse_loss(mask_emb, emb)
                prop_loss = F.mse_loss(pred_props, batch_props) if phase == "finetune" else 0
                total_loss_batch = contrastive_loss + 0.1 * mask_loss + (0.2 * prop_loss if phase == "finetune" else 0)
                
                optimizer.zero_grad()
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += total_loss_batch.item()
                if phase == "finetune":
                    prop_mse += prop_loss.item()
            
            scheduler.step()
            val_loss, val_mse = evaluate_model(model, val_graphs, val_props)
            logging.info(f"Epoch {epoch+1}/{epochs}, Phase: {phase}, Train Loss: {total_loss/num_batches:.4f}, "
                         f"Train MSE: {prop_mse/num_batches if phase == 'finetune' else 0:.4f}, "
                         f"Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "chg_ae_best.pth")
    
    model.load_state_dict(torch.load("chg_ae_best.pth"))
    return model

def evaluate_model(model, graphs, props):
    model.eval()
    total_loss = 0
    prop_mse = 0
    with torch.no_grad():
        for i in range(0, len(graphs), 100):
            batch_graphs = graphs[i:i+100]
            batch_props = torch.tensor(props[i:i+100], dtype=torch.float32)
            batch_g = dgl.batch(batch_graphs)
            emb, pred_props = model(batch_g)
            aug_g = chemistry_aware_augmentation(batch_g)
            aug_emb, _ = model(aug_g)
            loss = model.nt_xent_loss(emb, aug_emb, pred_props, batch_props)
            prop_loss = F.mse_loss(pred_props, batch_props)
            total_loss += (loss + 0.1 * prop_loss).item()
            prop_mse += prop_loss.item()
    num_batches = (len(graphs) + 99) // 100
    return total_loss / num_batches, prop_mse / num_batches

def compute_embeddings(model, graphs, batch_size=1000):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(graphs), batch_size):
            batch_g = dgl.batch(graphs[i:i+batch_size])
            emb, _ = model(batch_g)
            embeddings.append(emb)
    return torch.cat(embeddings, dim=0)

def validate_embeddings(embeddings, props):
    clusters = HDBSCAN(min_cluster_size=200, min_samples=5, cluster_selection_epsilon=0.3).fit_predict(embeddings.numpy())
    valid_idx = clusters >= 0
    silhouette = silhouette_score(props[valid_idx], clusters[valid_idx])
    return silhouette

def main():
    smiles, props, scaler = load_polyone_data()
    train_smiles, train_props, val_smiles, val_props = cluster_split(smiles, props)
    train_graphs = preprocess_dataset(train_smiles)
    val_graphs = preprocess_dataset(val_smiles)
    model = train_model(train_graphs, train_props, val_graphs, val_props)
    train_emb = compute_embeddings(model, train_graphs)
    val_emb = compute_embeddings(model, val_graphs)
    train_silhouette = validate_embeddings(train_emb, train_props)
    val_silhouette = validate_embeddings(val_emb, val_props)
    _, train_mse = evaluate_model(model, train_graphs, train_props)
    _, val_mse = evaluate_model(model, val_graphs, val_props)
    logging.info(f"Train Silhouette: {train_silhouette:.4f}, Val Silhouette: {val_silhouette:.4f}, "
                 f"Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")
    torch.save(model.state_dict(), "chg_ae_final.pth")
    np.save("train_embeddings.npy", train_emb.numpy())
    np.save("val_embeddings.npy", val_emb.numpy())

if __name__ == "__main__":
    torch.jit.enable_onednn_fusion(True)
    main()
