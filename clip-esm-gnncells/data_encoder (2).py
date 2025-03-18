import torch
import esm
from typing import Dict, Tuple, List
import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import networkx as nx
from torch import nn
import torch.nn.functional as F
from types import SimpleNamespace
from torch_scatter import scatter_mean
import requests
import glob as glob

missing_genes_filled = {
    'APOBEC3C': 'MNPQIRNPMKAMYPGTFYFQFKNLWEANDRNETWLCFTVEGIKRRSVVSWKTGVFRNQVDSETHCHAERCFLSWFCDDILSPNTKYQVTWYTSWSPCPDCAGEVAEFLARHSNVNLTIFTARLYYFQYPCYQEGLRSLSQEGVAVEIMDYEDFKYCWENFVYNDNEPFKPWKGLKTNFRLLKRRLRESLQ',
    'APOBEC3D': 'MNPQIRNPMERMYRDTFYDNFENEPILYGRSYTWLCYEVKIKRGRSNLLWDTGVFRGPVLPKRQSNHRQEVYFRFENHAEMCFLSWFCGNRLPANRRFQITWFVSWNPCLPCVVKVTKFLAEHPNVTLTISAARLYYYRDRDWRWVLLRLHKAGARVKIMDYEDFAYCWENFVCNEGQPFMPWYKFDDNYASLHRTLKEILRNPMEAMYPHIFYFHFKNLLKACGRNESWLCFTMEVTKHHSAVFRKRGVFRNQVDPETHCHAERCFLSWFCDDILSPNTNYEVTWYTSWSPCPECAGEVAEFLARHSNVNLTIFTARLCYFWDTDYQEGLCSLSQEGASVKIMGYKDFVSCWKNFVYSDDEPFKPWKGLQTNFRLLKRRLREILQ',
    'SmoM2': 'MQRRRFLAQAAGAAGAGLATVGMPAIAQSAPAVRWRMSTSWPKSLDTIYGSAEDLCKRVAQLTDGKFEIRAFPGGELVPAAQNMDAVSNGTVECNHVLSTAYIGKNTALTFDTGLSFGLSARQHNAWVHSGGGLKMLRALYKKYNIVNHVCGNVGVQMGGWYRKEIKSLADLKGLNMRIGGIGGMVLSKLGAVPQQIPPGDIYPALEKGTIDAAEWIGPYDDEKLGFNKVAPFYYSPGWFEGSASITSMVHDKAWEALPPAYQAAFEAAAGEQTMRMLANYDARNPLALRKLIAGGAKVSFFPKEVMDAAYNASQELWVELSAKNPDFAAIYPDWKKFQVDQVGWFRVAESPLDNYTFAAVAKAQAK',
    'SV40': 'MDKVLNREESLQLMDLLGLERSAWGNIPLMRKAYLKKCKEFHPDKGGDEEKMKKMNTLYKKMEDGVKYAHQPDFGGFWDATEIPTYGTDEWEQWWNAFNEENLFCSEEMPSSDDEATADSQHSTPPKKKRKVEDPKDFPSELLSFLSHAVFSNRTLACFAIYTTKEKAALLYKKIMEKYSVTFISRHNSYNHNILFFLTPHRHRVSAINNYAQKLCTFSFLICKGVNKEYLMYSALTRDPFSVIEESLPGGLKEHDFNPEEAEETKQVSWKLVTEYAMETKCDDVLLLLGMYLEFQYSFEMCLKCIKKEQPSHYKYHEKHYANAAIFADSKNQKTICQQAVDTVLAKKRVDSLQLTREQMLTNRFNDLLDRMDIMFGSTGSADIEEWMAGVAWLHCLLPKMDSVVYDFLKCMVYNIPKKRYWLFKGPIDSGKTTLAAALLELCGGKALNVNLPLDRLNFELGVAIDQFLVVFEDVKGTGGESRDLPSGQGINNLDNLRDYLDGSVKVNLEKKHLNKRTQIFPPGIVTMNEYSVPKTLQARFVKQIDFRPKDYLKHCLERSEFLLEKRIIQSGIALLLMLIWYRPVAEFAQSIQSRIVEWKERLDKEFSLSVYQKMKFNVAMGIGVLDWLRNSDDDDEDSQENADKNEDGGEKNMEDSGHETGIDSQSQGSFQAPQSSQSVHDHNQPYHICRGFTCFKKPPTPPPEPET',
    'IKBKA': 'MERPPGLRPGAGGPWEMRERLGTGGFGNVCLYQHRELDLKIAIKSCRLELSTKNRERWCHEIQIMKKLNHANVVKACDVPEELNILIHDVPLLAMEYCSGGDLRKLLNKPENCCGLKESQILSLLSDIGSGIRYLHENKIIHRDLKPENIVLQDVGGKIIHKIIDLGYAKDVDQGSLCTSFVGTLQYLAPELFENKPYTATVDYWSFGTMVFECIAGYRPFLHHLQPFTWHEKIKKKDPKCIFACEEMSGEVRFSSHLPQPNSLCSLVVEPMENWLQLMLNWDPQQRGGPVDLTLKQPRCFVLMDHILNLKIVHILNMTSAKIISFLLPPDESLHSLQSRIERETGINTGSQELLSETGISLDPRKPASQCVLDGVRGCDSYMVYLFDKSKTVYEGPFASRSLSDCVNYIVQDSKIQLPIIQLRKVWAEAVHYVSGLKEDYSRLFQGQRAAMLSLLRYNANLTKMKNTLISASQQLKAKLEFFHKSIQLDLERYSEQMTYGISSEKMLKAWKEMEEKAIHYAEVGVIGYLEDQIMSLHAEIMELQKSPYGRRQGDLMESLEQRAIDLYKQLKHRPSDHSYSDSTEMVKIIVHTVQSQDRVLKELFGHLSKLLGCKQKIIDLLPKVEVALSNIKEADNTVMFMQGKRQKEIWHLLKIACTQSSARSLVGSSLEGAVTPQTSAWLPPTSAEHDHSLSCVVTPQDGETSAQMIEENLNCLGHLSTIIHEANEEQGNSMMNLDWSWLTE',
    'BCL-XL': 'MSQSNRELVVDFLSYKLSQKGYSWSQFSDVEENRTEAPEGTESEMETPSAINGNPSWHLADSPAVNGATGHSSSLDAREVIPMAAVKQALREAGDEFELRYRRAFSDLTSQLHITPGTAYQSFEQVVNELFRDGVNWGRIVAFFSFGGALCVESVDKEMQVLVSRIAAWMATYLNDHLEPWIQENGGWDTFVELYGNNAAAESRKGQERFNRWFLTGMTVAGVVLLGSLFSRK',
}

def get_uniprot_sequence(gene, organism="9606"):
    """Get protein sequence from UniProt."""
    url = f"https://rest.uniprot.org/uniprotkb/search?query=gene_exact:{gene}+AND+organism_id:{organism}&format=fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta_parts = response.text.split('>')
        if len(fasta_parts) > 1:
            fasta_content = fasta_parts[1].strip()
            if '\n' in fasta_content:
                header, sequence = fasta_content.split('\n', 1)
                sequence = sequence.replace('\n', '')
                if len(sequence) > 10000:
                    return None
                return sequence
    return None
        
class DataEncoder(nn.Module):  # Change to inherit from nn.Module
    def __init__(self, base_path: str = "v1_data"):
        super().__init__()  # Add parent class initialization
        self.base_path = base_path
        self.train_datasets = [1, 3, 4, 5, 6, 7, 8]
        self.val_datasets = [2, 9]
        self.setup_esm()
        self.config = SimpleNamespace(
            esm_dim=1280,
            latent_dim=1024,
            dropout=0.1,
            n_heads=8,
            use_layernorm=True,
            top_k=10,
            n_layers=3,
            use_time_encoding=True,
            time_encoding_dim=128,
            gnn_type='pignn',
            n_neighbors=32
        )
        
        self.gene_to_sequence = {}
        
    def forward(self, x):
        # Required for DDP, but not used in this case
        return x

    def setup_esm(self):
        """Initialize ESM model for tokenization and embeddings"""
        # device = next(self.parameters()).device
        self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        
        device = torch.cuda.current_device()

        self.esm_model = self.esm_model.to(device)
        self.esm_model.eval()

    def load_dataset(self, dataset_num: int) -> Tuple[sc.AnnData, pd.DataFrame]:
        """Load h5ad and csv files for a dataset."""
        h5ad_file = f"dataset_{dataset_num}_single_gene_updated.h5ad"
        csv_file = f"dataset_{dataset_num}_single_gene_updated_cellwise_data.csv"
        
        adata = sc.read_h5ad(os.path.join(self.base_path, h5ad_file))
        cell_data = pd.read_csv(os.path.join(self.base_path, csv_file))
        
        return adata, cell_data

    def compute_graph_metrics(self, adata: sc.AnnData) -> Dict:
        """Compute graph metrics from AnnData object."""
        if 'connectivities' not in adata.obsp:
            sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_diffmap')
        
        connectivity_matrix = adata.obsp['connectivities']
        if not isinstance(connectivity_matrix, sparse.csr_matrix):
            connectivity_matrix = sparse.csr_matrix(connectivity_matrix)
        
        try:
            G = nx.from_scipy_sparse_array(connectivity_matrix)
        except AttributeError:
            G = nx.Graph(connectivity_matrix)
            
        metrics = {
            'edges': G.number_of_edges(),
            'avg_degree': G.number_of_edges()*2/G.number_of_nodes(),
            'components': nx.number_connected_components(G)
        }
        return metrics

    def encode_cell_state(self, adata: sc.AnnData) -> torch.Tensor:
        """Encode cell state from AnnData object."""
        device = next(self.parameters()).device
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        if isinstance(adata.X, np.ndarray) or isinstance(adata.X, memoryview):
            X = sparse.csr_matrix(adata.X)
        else:
            X = adata.X

        self.config.gene_dim = adata.n_vars

        # Move tensors to correct device
        expr = torch.tensor(X.toarray(), device=device).float()
        dpt = torch.tensor(adata.obsm['X_diffmap'], device=device).float()

        adj = adata.obsp['connectivities']
        edge_indices = torch.tensor(adj.nonzero(), device=device).long()
        edge_idx = edge_indices.T if edge_indices.shape[0] != 2 else edge_indices

        batch_idx = torch.zeros(len(adata), dtype=torch.long, device=device)

        from models.encoders.cell_encoder import CellStateEncoder
        encoder = CellStateEncoder(self.config).to(device)
        if world_size > 1:
            encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[rank])
        cell_emb = encoder(expr, dpt, edge_idx, batch_idx)

        return cell_emb
    
    def get_esm_tokens_and_embeddings(self, sequence: str):
        data = [("protein", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        device = next(self.parameters()).device
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_embeddings = results["representations"][33]

        return batch_tokens[0], token_embeddings[0]
     
    def process_batch(self, genes, max_len, is_top5=False, batch_idx=0, checkpoint_dir="checkpoints"):
        """
        Process a batch of genes with DDP support and memory management
        """
        device = next(self.parameters()).device
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        with torch.no_grad():  # Prevent gradient computation
            tokens_list = []
            embeddings_list = []
            batch_max_len = 0

            try:
                for gene_idx, gene in enumerate(genes):
                    if gene_idx % 10 == 0:
                        torch.cuda.empty_cache()
#                         if hasattr(torch.cuda, 'reset_peak_memory_stats'):
#                             torch.cuda.reset_peak_memory_stats()

                    if is_top5:
                        # Process top5 genes...
                        gene_tokens = []
                        gene_embeddings = []
                        gene_list = [g.strip().strip("'\"") for g in gene.strip('[]').split(',')]

                        for g in gene_list:
                            seq = self.gene_to_sequence.get(g)
                            if seq:
                                tokens, embeddings = self.get_esm_tokens_and_embeddings(seq)
                                tokens = tokens.to(device)
                                embeddings = embeddings.to(device)

                                batch_max_len = max(batch_max_len, len(tokens))

                                # Pad sequences
                                padded_tokens = F.pad(tokens, (0, max_len - len(tokens)), 
                                                    value=self.alphabet.padding_idx)
                                padded_embeddings = F.pad(embeddings, (0, 0, 0, max_len - len(embeddings)))

                                gene_tokens.append(padded_tokens)
                                gene_embeddings.append(padded_embeddings)
                            else:
                                # Create padding directly on GPU
                                padding_token = torch.full((max_len,), self.alphabet.padding_idx, 
                                                        device=device)
                                padding_embedding = torch.zeros((max_len, self.config.esm_dim), 
                                                            device=device)
                                gene_tokens.append(padding_token)
                                gene_embeddings.append(padding_embedding)

                        # Stack on current GPU
                        stacked_tokens = torch.stack(gene_tokens)
                        stacked_embeddings = torch.stack(gene_embeddings)
                        
                        # Move to CPU to save memory if needed
                        if gene_idx % 50 == 0:
                            stacked_tokens = stacked_tokens.cpu()
                            stacked_embeddings = stacked_embeddings.cpu()
                        
                        tokens_list.append(stacked_tokens)
                        embeddings_list.append(stacked_embeddings)

                    else:
                        # Process single gene...
                        seq = self.gene_to_sequence.get(gene)
                        if seq:
                            tokens, embeddings = self.get_esm_tokens_and_embeddings(seq)
                            tokens = tokens.to(device)
                            embeddings = embeddings.to(device)
                            
                            batch_max_len = max(batch_max_len, len(tokens))

                            padded_tokens = F.pad(tokens, (0, max_len - len(tokens)), 
                                                value=self.alphabet.padding_idx)
                            padded_embeddings = F.pad(embeddings, (0, 0, 0, max_len - len(embeddings)))

                            # Move to CPU periodically
                            if gene_idx % 50 == 0:
                                padded_tokens = padded_tokens.cpu()
                                padded_embeddings = padded_embeddings.cpu()

                            tokens_list.append(padded_tokens)
                            embeddings_list.append(padded_embeddings)
                        else:
                            padding_token = torch.full((max_len,), self.alphabet.padding_idx, 
                                                    device=device)
                            padding_embedding = torch.zeros((max_len, self.config.esm_dim), 
                                                        device=device)
                            tokens_list.append(padding_token)
                            embeddings_list.append(padding_embedding)

                    # Save checkpoint every 50 genes
                    if gene_idx > 0 and gene_idx % 50 == 0:
                        torch.distributed.barrier()  # Synchronize before saving
                        checkpoint = {
                            'tokens_list': tokens_list,
                            'embeddings_list': embeddings_list,
                            'batch_idx': batch_idx,
                            'gene_idx': gene_idx,
                            'batch_max_len': batch_max_len
                        }
                        checkpoint_path = f'{checkpoint_dir}/checkpoint_batch_{batch_idx}_gene_{gene_idx}_rank_{rank}.pt'
                        torch.save(checkpoint, checkpoint_path)
                        if rank == 0:  # Only rank 0 prints progress
                            print(f"Saved checkpoint for batch {batch_idx}, gene {gene_idx}")

            except RuntimeError as e:
                print(f"RuntimeError in batch {batch_idx}: {str(e)}")
                torch.cuda.empty_cache()
                raise

            finally:
                torch.cuda.empty_cache()
                
            # Synchronize before returning results
            torch.distributed.barrier()
            return tokens_list, embeddings_list, batch_max_len

    def encode_protein_perturbation(self, cell_data: pd.DataFrame, batch_size=8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        torch.cuda.empty_cache()

        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Split data across ranks
        total_samples = len(cell_data)
        samples_per_rank = total_samples // world_size
        start_idx = rank * samples_per_rank
        end_idx = start_idx + samples_per_rank if rank != world_size - 1 else total_samples
        local_cell_data = cell_data.iloc[start_idx:end_idx]

        # Collect gene sequences efficiently
        all_genes = set()
        for genes in local_cell_data['top5_genes'].str.strip('[]').str.split(','):
            all_genes.update(g.strip().strip("'\"") for g in genes)
        all_genes.update(local_cell_data['perturbation_gene'])
        unique_genes = list(all_genes)

        # Gather unique genes from all ranks
        try:
            all_unique_genes = [None] * world_size
            torch.distributed.all_gather_object(all_unique_genes, unique_genes)
            all_unique_genes = list(set().union(*all_unique_genes))
            gene_to_idx = {gene: idx for idx, gene in enumerate(all_unique_genes)}

            if rank == 0:
                print(f"\nTotal unique genes (including top5): {len(all_unique_genes)}")
        except RuntimeError as e:
            print(f"Error during gathering unique genes on rank {rank}: {e}")
            # Save emergency checkpoint
            torch.save({
                'unique_genes': unique_genes,
            }, f'emergency_unique_genes_rank_{rank}.pt')
            raise

        if rank == 0:
            print(f"\nTotal unique genes (including top5): {len(all_unique_genes)}")

        # Process gene sequences
        self.gene_to_sequence = {}
        missing_count = 0
        try:
            for i, gene in enumerate(all_unique_genes[rank::world_size]):  # Distribute gene processing
                if rank == 0 and i % 100 == 0:
                    print(f"Processing gene {i}/{len(all_unique_genes)}")

                if gene in missing_genes_filled:
                    self.gene_to_sequence[gene] = missing_genes_filled[gene]
                else:
                    seq = get_uniprot_sequence(gene)
                    if seq:
                        self.gene_to_sequence[gene] = seq
                    else:
                        missing_count += 1
                        self.gene_to_sequence[gene] = None
        except Exception as e:
            print(f"Error during gene processing on rank {rank}: {e}")
            # Save emergency checkpoint
            torch.save({
                'gene_to_sequence_partial': self.gene_to_sequence,
                'all_unique_genes': all_unique_genes,
            }, f'emergency_gene_processing_rank_{rank}.pt')
            raise


        # Gather gene sequences from all ranks
        try:
            all_sequences = [None] * world_size
            torch.distributed.all_gather_object(all_sequences, self.gene_to_sequence)
            for sequences in all_sequences:
                self.gene_to_sequence.update(sequences)

            torch.distributed.barrier()  # Synchronize after sequence processing
        except RuntimeError as e:
            print(f"Error during gathering gene sequences on rank {rank}: {e}")
            # Save emergency checkpoint
            torch.save({
                'gene_to_sequence': self.gene_to_sequence,
            }, f'emergency_gene_sequences_rank_{rank}.pt')
            raise
    
        if rank == 0:
            print(f"Number of genes with missing sequences: {missing_count} out of {len(all_unique_genes)}\n")

        # Calculate max lengths (do this on all ranks to ensure consistency)
        def get_seq_length(seq):
            if seq:
                tokens, _ = self.get_esm_tokens_and_embeddings(seq)
                return len(tokens)
            return 0

        pert_max_len = max(get_seq_length(self.gene_to_sequence[gene]) or 0 
                        for gene in local_cell_data['perturbation_gene'])

        top5_max_len = 0
        for genes_str in local_cell_data['top5_genes']:
            gene_list = [g.strip().strip("'\"") for g in genes_str.strip('[]').split(',')]
            for gene in gene_list:
                seq_len = get_seq_length(self.gene_to_sequence.get(gene)) or 0
                top5_max_len = max(top5_max_len, seq_len)

        # Synchronize max lengths across ranks
        pert_max_len = torch.tensor(pert_max_len, device=device)
        top5_max_len = torch.tensor(top5_max_len, device=device)
        torch.distributed.all_reduce(pert_max_len, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(top5_max_len, op=torch.distributed.ReduceOp.MAX)
        pert_max_len = pert_max_len.item()
        top5_max_len = top5_max_len.item()

        if rank == 0:
            print(f"Max sequence lengths - Perturbation: {pert_max_len}, Top5: {top5_max_len}")

        # Process in batches with memory management and checkpointing
        pert_tokens_all = []
        pert_embeddings_all = []
        top_tokens_all = []
        top_embeddings_all = []

        sub_batch_size = 1  # Process one sequence at a time

        # Check for existing checkpoints
        latest_checkpoint = None
        if os.path.exists(checkpoint_dir):
            checkpoints = sorted(glob.glob(f'{checkpoint_dir}/checkpoint_batch_*_emergency_rank_{rank}.pt'))
            if checkpoints:
                latest_checkpoint = torch.load(checkpoints[-1])
                if rank == 0:
                    print(f"Resuming from checkpoint at batch {latest_checkpoint['batch_idx']}")

        start_batch = latest_checkpoint['batch_idx'] if latest_checkpoint else 0

        for i in range(start_batch, len(local_cell_data), batch_size):
            if rank == 0:
                print(f"Processing batch {i//batch_size + 1}/{len(local_cell_data)//batch_size + 1}")

            try:
                # Process perturbation genes
                batch_pert = local_cell_data['perturbation_gene'][i:i+batch_size]
                for j in range(0, len(batch_pert), sub_batch_size):
                    sub_batch = batch_pert[j:j+sub_batch_size]
                    pert_tokens, pert_embeddings, batch_len = self.process_batch(
                        sub_batch, pert_max_len, is_top5=False, batch_idx=i+j)

                    pert_tokens_all.extend(pert_tokens)
                    pert_embeddings_all.extend(pert_embeddings)
                    torch.cuda.empty_cache()

                # Process top5 genes
                batch_top5 = local_cell_data['top5_genes'][i:i+batch_size]
                for j in range(0, len(batch_top5), sub_batch_size):
                    sub_batch = batch_top5[j:j+sub_batch_size]
                    top_tokens, top_embeddings, batch_len = self.process_batch(
                        sub_batch, top5_max_len, is_top5=True, batch_idx=i+j)

                    top_tokens_all.extend(top_tokens)
                    top_embeddings_all.extend(top_embeddings)
                    torch.cuda.empty_cache()

                # Save checkpoint after each successful batch
                checkpoint = {
                    'batch_idx': i + batch_size,
                    'pert_tokens': pert_tokens_all,
                    'pert_embeddings': pert_embeddings_all,
                    'top_tokens': top_tokens_all,
                    'top_embeddings': top_embeddings_all
                }
                torch.save(checkpoint, f'{checkpoint_dir}/checkpoint_batch_{i}_rank_{rank}.pt')

            except RuntimeError as e:
                print(f"Error processing batch {i} on rank {rank}: {str(e)}")
                checkpoint = {
                    'batch_idx': i,
                    'pert_tokens': pert_tokens_all,
                    'pert_embeddings': pert_embeddings_all,
                    'top_tokens': top_tokens_all,
                    'top_embeddings': top_embeddings_all
                }
                torch.save(checkpoint, 
                        f'{checkpoint_dir}/checkpoint_batch_{i}_emergency_rank_{rank}.pt')
                raise

        torch.distributed.barrier()  # Synchronize before stacking results

        # Stack local results
        try:
            padded_pert_tokens = torch.stack(pert_tokens_all)
            padded_pert_embeddings = torch.stack(pert_embeddings_all)
            padded_top_tokens = torch.stack(top_tokens_all)
            padded_top_embeddings = torch.stack(top_embeddings_all)
        except Exception as e:
            print(f"Error stacking results on rank {rank}:", e)
            print("Lengths:", len(pert_tokens_all), len(pert_embeddings_all), 
                len(top_tokens_all), len(top_embeddings_all))
            raise

        # Process local gene IDs and values
        pert_gene_ids = torch.tensor([gene_to_idx[g] for g in local_cell_data['perturbation_gene']])
        top_gene_ids = torch.tensor([[gene_to_idx[g.strip().strip("'\"")]  
                                    for g in genes.strip('[]').split(',')] 
                                for genes in local_cell_data['top5_genes']])
        top_values = torch.tensor([
            list(map(float, vals.replace('[', '').replace(']', '').split(',')))
            for vals in local_cell_data['top5_values']
        ])

        # Initialize encoder
        from models.encoders.combined_pert_prot_encoder import CombinedEncoder
        encoder = CombinedEncoder(self.config).to(device)
        if world_size > 1:
            encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[rank])

        protein_embs = []
        pert_embs = []
        chunk_size = batch_size * 2

        for i in range(0, len(padded_pert_embeddings), chunk_size):
            chunk_inputs = [
                pert_gene_ids[i:i+chunk_size].to(device),
                padded_pert_embeddings[i:i+chunk_size].to(device).mean(dim=1),
                top_gene_ids[i:i+chunk_size].to(device),
                top_values[i:i+chunk_size].to(device),
                padded_top_embeddings[i:i+chunk_size].to(device).mean(dim=2)
            ]

            protein_emb, pert_emb = encoder(*chunk_inputs)

            protein_embs.append(protein_emb.cpu())
            pert_embs.append(pert_emb.cpu())

            torch.cuda.empty_cache()
            if i % (chunk_size * 10) == 0:
                torch.save({
                    'protein_embs': protein_embs,
                    'pert_embs': pert_embs,
                    'chunk_idx': i
                }, f'{checkpoint_dir}/encoder_checkpoint_{i}_rank_{rank}.pt')

        torch.distributed.barrier()  # Synchronize before combining results

        # Combine local results
        protein_emb = torch.cat(protein_embs)
        pert_emb = torch.cat(pert_embs)

        # Gather results from all ranks
        if world_size > 1:
            all_protein_embs = [torch.zeros_like(protein_emb) for _ in range(world_size)]
            all_pert_embs = [torch.zeros_like(pert_emb) for _ in range(world_size)]
            all_pert_tokens = [torch.zeros_like(padded_pert_tokens) for _ in range(world_size)]
            all_pert_embeddings = [torch.zeros_like(padded_pert_embeddings) for _ in range(world_size)]
            all_top_tokens = [torch.zeros_like(padded_top_tokens) for _ in range(world_size)]
            all_top_embeddings = [torch.zeros_like(padded_top_embeddings) for _ in range(world_size)]

            try:
                torch.distributed.all_gather(all_protein_embs, protein_emb)
                torch.distributed.all_gather(all_pert_embs, pert_emb)
                torch.distributed.all_gather(all_pert_tokens, padded_pert_tokens)
                torch.distributed.all_gather(all_pert_embeddings, padded_pert_embeddings)
                torch.distributed.all_gather(all_top_tokens, padded_top_tokens)
                torch.distributed.all_gather(all_top_embeddings, padded_top_embeddings)
            except RuntimeError as e:
                print(f"Error during tensor gathering on rank {rank}: {e}")
                # Save emergency checkpoint
                torch.save({
                    'protein_emb': protein_emb,
                    'pert_emb': pert_emb,
                    'padded_pert_tokens': padded_pert_tokens,
                    'padded_pert_embeddings': padded_pert_embeddings,
                    'top_tokens': padded_top_tokens,
                    'top_embeddings': padded_top_embeddings,
                }, f'emergency_gather_checkpoint_rank_{rank}.pt')
                raise

                
            if rank == 0:
                protein_emb = torch.cat(all_protein_embs)
                pert_emb = torch.cat(all_pert_embs)
                padded_pert_tokens = torch.cat(all_pert_tokens)
                padded_pert_embeddings = torch.cat(all_pert_embeddings)
                padded_top_tokens = torch.cat(all_top_tokens)
                padded_top_embeddings = torch.cat(all_top_embeddings)

        if rank == 0:
            print("Encoding completed successfully")

        return protein_emb, pert_emb, padded_pert_tokens, padded_pert_embeddings, padded_top_tokens, padded_top_embeddings
    
    def compute_embedding_metrics(self, embeddings: torch.Tensor) -> Dict:
        device = next(self.parameters()).device
        embeddings = embeddings.to(device)
        metrics = {
            'mean': embeddings.mean().item(),
            'std': embeddings.std().item(),
            'min': embeddings.min().item(),
            'max': embeddings.max().item(),
            'norm_mean': torch.norm(embeddings, dim=1).mean().item(),
            'cos_sim_mean': F.cosine_similarity(embeddings.unsqueeze(1), 
                                             embeddings.unsqueeze(0), dim=2).mean().item()
        }
        return metrics
    
    def process_all_data(self) -> Dict:
        device = next(self.parameters()).device
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        train_data = {'adata': [], 'cell_data': [], 'embeddings': []}
        val_data = {'adata': [], 'cell_data': [], 'embeddings': []}

        # Split datasets across processes
        train_datasets = self.train_datasets[rank::world_size]
        val_datasets = self.val_datasets[rank::world_size]
        
        for dataset_num in train_datasets:
            print(f"Rank {rank} processing training dataset {dataset_num}")
            adata, cell_data = self.load_dataset(dataset_num)
            cell_emb = self.encode_cell_state(adata)
            protein_emb, pert_emb, esm_tokens, esm_embeddings, top5_tokens, top5_embeddings = \
                self.encode_protein_perturbation(cell_data, batch_size = 8)
            
            metrics = {
                'graph': self.compute_graph_metrics(adata),
                'cell_emb': self.compute_embedding_metrics(cell_emb),
                'protein_emb': self.compute_embedding_metrics(protein_emb),
                'pert_emb': self.compute_embedding_metrics(pert_emb)
            }
            
            train_data['adata'].append(adata)
            train_data['cell_data'].append(cell_data)
            train_data['embeddings'].append({
                'cell': cell_emb,
                'protein': protein_emb,
                'perturbation': pert_emb,
                'esm_tokens': esm_tokens,          # Perturbation gene tokens
                'esm_embeddings': esm_embeddings,  # Perturbation gene embeddings
                'top5_tokens': top5_tokens,        # Top5 perturbed genes tokens
                'top5_embeddings': top5_embeddings,# Top5 perturbed genes embeddings
                'metrics': metrics
            })
            
        for dataset_num in val_datasets:
            print(f"Rank {rank} processing validation dataset {dataset_num}")
            adata, cell_data = self.load_dataset(dataset_num)
            cell_emb = self.encode_cell_state(adata)
            protein_emb, pert_emb, esm_tokens, esm_embeddings, top5_tokens, top5_embeddings = \
                self.encode_protein_perturbation(cell_data)
            
            metrics = {
                'graph': self.compute_graph_metrics(adata),
                'cell_emb': self.compute_embedding_metrics(cell_emb),
                'protein_emb': self.compute_embedding_metrics(protein_emb),
                'pert_emb': self.compute_embedding_metrics(pert_emb)
            }
            
            val_data['adata'].append(adata)
            val_data['cell_data'].append(cell_data)
            val_data['embeddings'].append({
                'cell': cell_emb,
                'protein': protein_emb,
                'perturbation': pert_emb,
                'esm_tokens': esm_tokens,
                'esm_embeddings': esm_embeddings,
                'top5_tokens': top5_tokens,
                'top5_embeddings': top5_embeddings,
                'metrics': metrics
            })
            
        # Gather data from all processes
        try:
            all_train_data = [None] * world_size
            all_val_data = [None] * world_size

            torch.distributed.all_gather_object(all_train_data, train_data)
            torch.distributed.all_gather_object(all_val_data, val_data)

        except RuntimeError as e:
            print(f"Error during data gathering on rank {rank}: {e}")
            # Save emergency checkpoint
            torch.save({
                'train_data': train_data,
                'val_data': val_data,
            }, f'emergency_gather_data_rank_{rank}.pt')
            raise

        if rank == 0:
            # Combine results on rank 0
            combined_train = {'adata': [], 'cell_data': [], 'embeddings': []}
            combined_val = {'adata': [], 'cell_data': [], 'embeddings': []}

            for proc_data in all_train_data:
                combined_train['adata'].extend(proc_data['adata'])
                combined_train['cell_data'].extend(proc_data['cell_data'])
                combined_train['embeddings'].extend(proc_data['embeddings'])

            for proc_data in all_val_data:
                combined_val['adata'].extend(proc_data['adata'])
                combined_val['cell_data'].extend(proc_data['cell_data'])
                combined_val['embeddings'].extend(proc_data['embeddings'])

            return {'train': combined_train, 'val': combined_val}
        return None
                 
# if __name__ == "__main__":
#     # Test the DataEncoder class
#     encoder = DataEncoder()
#     print("Available methods:", [method for method in dir(encoder) if not method.startswith('_')])
#     try:
#         encoder.process_all_data()
#         print("process_all_data exists and runs")
#     except Exception as e:
#         print(f"Error testing process_all_data: {e}")

# if __name__ == "__main__":
#     encoder = DataEncoder()
#     print("Testing process_batch method...")
#     try:
#         # Use actual gene names for the test
#         test_genes = ["APOBEC3C", "APOBEC3D"]  # Example gene names from missing_genes_filled
#         encoder.process_batch(test_genes, max_len=1134)  # Use an appropriate max_len for padding
#         print("process_batch is defined and works with actual gene names.")
#     except Exception as e:
#         print(f"Error: {e}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    
    def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def cleanup():
        dist.destroy_process_group()
        
    def run(rank, world_size):
        setup(rank, world_size)
        encoder = DataEncoder().to(rank)
        encoder = DDP(encoder, device_ids=[rank])
        try:
            test_genes = ["APOBEC3C", "APOBEC3D"]
            encoder.process_batch(test_genes, max_len=1134)
        finally:
            cleanup()

    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

