import torch
import numpy as np


def sim(z_i, z_j):
    """Normalized dot product (cosine similarity) between two 1-D vectors."""
    norm_dot_product = torch.dot(z_i, z_j) / (
        torch.linalg.norm(z_i) * torch.linalg.norm(z_j)
    )
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version)."""
    N = out_left.shape[0]
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    total_loss = 0
    for k in range(N):
        z_k, z_k_N = out[k], out[k + N]
        # l(k, k+N): positive similarity vs all negatives (excluding self index k)
        sims_k = torch.stack([sim(z_k, out[j]) for j in range(2 * N) if j != k]) / tau
        l1 = torch.logsumexp(sims_k, dim=0) - sim(z_k, z_k_N) / tau
        # l(k+N, k): symmetric term, excluding self index k+N
        sims_kn = torch.stack([sim(z_k_N, out[j]) for j in range(2 * N) if j != k + N]) / tau
        l2 = torch.logsumexp(sims_kn, dim=0) - sim(z_k_N, z_k) / tau
        total_loss += l1 + l2
    total_loss = total_loss / (2 * N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between each (out_left[k], out_right[k]) pair."""
    left = out_left / torch.linalg.norm(out_left, dim=1, keepdim=True)
    right = out_right / torch.linalg.norm(out_right, dim=1, keepdim=True)
    pos_pairs = (left * right).sum(dim=1, keepdim=True)
    return pos_pairs


def compute_sim_matrix(out):
    """2N x 2N matrix of cosine similarities between all rows of out."""
    norm = out / torch.linalg.norm(out, dim=1, keepdim=True)
    sim_matrix = norm @ norm.T
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device="cuda"):
    """Compute the contrastive loss L over a batch (vectorized)."""
    N = out_left.shape[0]
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]

    # Step 1: denominator for every augmented sample (sum over all j != i)
    exponential = torch.exp(sim_matrix / tau)  # [2N, 2N]
    mask = (torch.ones_like(exponential) - torch.eye(2 * N, device=sim_matrix.device)).bool()
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2N, 2N-1]
    denom = exponential.sum(dim=1)  # [2N]

    # Step 2/3: positive-pair similarities (k <-> k+N) for every augmented sample
    pos = torch.cat(
        [sim_positive_pairs(out_left, out_right), sim_positive_pairs(out_right, out_left)],
        dim=0,
    ).squeeze() / tau  # [2N]

    # Step 4: average InfoNCE loss
    loss = -torch.mean(pos - torch.log(denom))
    return loss


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
