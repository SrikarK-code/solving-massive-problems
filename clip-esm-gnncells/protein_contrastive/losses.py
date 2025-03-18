import torch
import torch.nn.functional as F
import geomloss

import torch
import torch.nn.functional as F

def nt_xent_loss(x, y, temperature=0.07):
    x, y = F.normalize(x, dim=1), F.normalize(y, dim=1)
    logits = torch.exp(torch.matmul(x, y.T) / temperature)
    loss = -torch.log(logits / torch.sum(logits, dim=1, keepdim=True)).mean()
    
    print(f"[DEBUG] NT-Xent Loss: {loss.item():.6f}")
    return loss

def barlow_twins_loss(x, y, lambda_param=0.0051):
    x, y = F.normalize(x, dim=1), F.normalize(y, dim=1)
    c = torch.mm(x.T, y) / x.shape[0]
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = (c - torch.diag(torch.diagonal(c))).pow_(2).sum()
    
    print(f"[DEBUG] Barlow Twins Loss: On-diag={on_diag.item():.6f}, Off-diag={off_diag.item():.6f}")
    return on_diag + lambda_param * off_diag

def vicreg_loss(x, y, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
    x, y = F.normalize(x, dim=1), F.normalize(y, dim=1)
    sim_loss = F.mse_loss(x, y)
    std_loss = torch.mean(F.relu(1 - torch.std(x, dim=0))) + torch.mean(F.relu(1 - torch.std(y, dim=0)))
    cov_loss = torch.mean((x.T @ x - torch.eye(x.shape[1], device=x.device))**2) + \
               torch.mean((y.T @ y - torch.eye(y.shape[1], device=y.device))**2)
    
    print(f"[DEBUG] VICReg Loss: Sim={sim_loss.item():.6f}, Std={std_loss.item():.6f}, Cov={cov_loss.item():.6f}")
    return sim_coeff * sim_loss + std_coeff * std_loss + cov_coeff * cov_loss


def wasserstein_loss(x, y, reg=0.01):
    """Computes Wasserstein loss with regularization for stability."""
    x, y = F.normalize(x, dim=1), F.normalize(y, dim=1)

    cost_matrix = torch.cdist(x, y, p=2)  # L2 distance
    wasserstein_dist = cost_matrix.mean()

    reg_term = reg * (x ** 2).sum()  # Regularization to prevent collapse

    print(f"[DEBUG] Wasserstein Loss: {wasserstein_dist.item()}")

    return wasserstein_dist + reg_term



def hyperbolic_loss(x, y):
    """Computes hyperbolic loss using projected embeddings."""
    def hyperbolic_distance(u, v):
        return torch.acosh(torch.clamp(1 + 2 * (u - v).pow(2).sum(-1) / 
               ((1 - u.pow(2).sum(-1)) * (1 - v.pow(2).sum(-1))), min=1.0 + 1e-6))
    
    x_hyp = project_hyperbolic(x)
    y_hyp = project_hyperbolic(y)

    dist = hyperbolic_distance(x_hyp.unsqueeze(1), y_hyp.unsqueeze(0))

    print(f"[DEBUG] Hyperbolic Distance - Mean: {dist.mean().item()}, Max: {dist.max().item()}")

    return dist.mean()


def off_diagonal(x):
    return x - torch.diag(torch.diag(x))

def project_hyperbolic(x, c=1.0):
    norm = torch.norm(x, dim=-1, keepdim=True)
    return x / (1 + norm)  # Maps to Poincare ball
