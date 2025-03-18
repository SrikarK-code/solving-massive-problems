import torch
import torch.nn.functional as F


def gmc_loss(cell_state, protein, queue, dataset_queue, dataset_labels, temperature):
    batch_size = cell_state.shape[0]
    eps = 1e-8
    temp = temperature.clamp(min=0.5, max=1.0)  # Wider temperature range
    
    # ---- Cross-Modal Contrast ----
    sim_cp = F.cosine_similarity(cell_state, protein, dim=-1)
    pos_weight = torch.exp(sim_cp / temp)
    
    # Hard negative mining from queue
    if queue is not None:
        queue_sims = torch.mm(cell_state, queue.t())
        topk_vals, _ = torch.topk(queue_sims, k=min(5, queue.size(0)), dim=1)
        neg_weight = torch.exp(topk_vals / temp).sum(dim=1)
    else:
        neg_weight = torch.zeros(batch_size, device=cell_state.device)

    # Add similarity distribution monitoring
    with torch.no_grad():
        print(f"CP Similarities: Min={sim_cp.min().item():.2f} Mean={sim_cp.mean().item():.2f} Max={sim_cp.max().item():.2f}")
        if queue is not None:
            print(f"Queue Similarities: Min={queue_sims.min().item():.2f} Mean={queue_sims.mean().item():.2f} Max={queue_sims.max().item():.2f}")
    
    # Stable CP loss calculation
    cp_denominator = pos_weight + neg_weight + eps
    loss_cp = -torch.log(pos_weight / cp_denominator)
    
    # ---- Intra-Modal Regularization ----
    with torch.no_grad():  # Prevent intra-modal gradients from dominating
        cell_sim = torch.mm(cell_state, cell_state.t()).fill_diagonal_(0)
        protein_sim = torch.mm(protein, protein.t()).fill_diagonal_(0)
        
    # Gentle intra-modal regularization
    loss_cell = 0.01 * torch.log(1 + torch.exp(cell_sim.mean() / temp))
    loss_protein = 0.01 * torch.log(1 + torch.exp(protein_sim.mean() / temp))

    # ---- Joint Modality Alignment ----
    joint_emb = torch.cat([cell_state, protein], dim=0)
    joint_sim = torch.mm(joint_emb, joint_emb.t())
    pos_mask = torch.eye(2*batch_size, device=joint_emb.device).bool()
    
    pos_pairs = joint_sim.masked_select(pos_mask)
    neg_pairs = joint_sim.masked_fill(pos_mask, float('-inf'))
    
    loss_joint = -pos_pairs + torch.logsumexp(neg_pairs / temp, dim=1)
    
    # ---- Final Stabilized Loss ----
    final_loss = (
        loss_cp.mean() +
        loss_cell.mean() +
        loss_protein.mean() +
        0.5 * loss_joint.mean()
    )

    # ---- Debugging and Validation ----
    valid_mask = (final_loss > 0) & (final_loss < 1e3)
    if valid_mask.all():
        print(f"VALID loss...: "
              f"CP: {loss_cp.mean().item():.3f}, "
              f"Cell: {loss_cell.mean().item():.3f}, "
              f"Protein: {loss_protein.mean().item():.3f}, "
              f"Joint: {loss_joint.mean().item():.3f}")
    if not valid_mask.all():
        print(f"Invalid loss detected! Components: "
              f"CP: {loss_cp.mean().item():.3f}, "
              f"Cell: {loss_cell.mean().item():.3f}, "
              f"Protein: {loss_protein.mean().item():.3f}, "
              f"Joint: {loss_joint.mean().item():.3f}")
        raise ValueError("Numerical instability detected")

    return final_loss.mean()
    
