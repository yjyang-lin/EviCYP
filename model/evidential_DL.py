import torch
import torch.nn.functional as F

def _to_one_hot(y, num_classes: int):
    """
    Accepts: y as shape (B,) integers OR (B, C) one-hot / soft labels.
    Returns: one-hot float tensor on same device.
    """
    if y.dim() == 1 or (y.dim() == 2 and y.size(-1) == 1):
        y = y.view(-1).long()
        y_oh = F.one_hot(y, num_classes=num_classes).float()
        return y_oh
    elif y.dim() == 2 and y.size(-1) == num_classes:
        return y.float()
    else:
        raise ValueError(f'Unsupported label shape: {tuple(y.shape)}')

def dirichlet_kl(alpha, beta=None):
    """
    KL(Dir(alpha) || Dir(beta)). If beta is None, use uniform Dirichlet with all-ones concentration.
    alpha, beta: (B, C)
    """
    if beta is None:
        beta = torch.ones_like(alpha, device=alpha.device)
    sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    sum_beta  = torch.sum(beta,  dim=-1, keepdim=True)

    lnB_alpha = torch.lgamma(sum_alpha) - torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)
    lnB_beta  = torch.lgamma(sum_beta)  - torch.sum(torch.lgamma(beta),  dim=-1, keepdim=True)

    # E[log p] terms
    digamma_alpha = torch.digamma(alpha)
    digamma_sum_alpha = torch.digamma(sum_alpha)

    kl = (lnB_alpha - lnB_beta) + torch.sum((alpha - beta) * (digamma_alpha - digamma_sum_alpha), dim=-1, keepdim=True)
    return kl  # (B, 1)

def dirichlet_evidence_loss(y, alpha, lam: float = 0.1, reduction: str = 'mean'):
    """
    Evidential classification loss
    with mean square error term + annealed KL regularizer to a uniform Dirichlet.

    y: labels as (B,) ints or (B, C) one-hot
    alpha: Dirichlet parameters (B, C) = evidence + 1, evidence >= 0
    lam:  KL weight
    reduction: 'mean' | 'sum' | 'none'
    """
    K = alpha.size(-1)
    y_oh = _to_one_hot(y, K).to(alpha.device)

    S = torch.sum(alpha, dim=-1, keepdim=True)              # (B,1)
    p = alpha / S                                           # (B,C) expected class probs
    # Squared error + variance term
    A = torch.sum((y_oh - p) ** 2, dim=-1, keepdim=True)    # (B,1)
    B = torch.sum(p * (1 - p) / (S + 1), dim=-1, keepdim=True)
    sce = A + B                                             # (B,1)

    # KL regularizer; following common practice use alpha_hat to avoid punishing correct evidence
    alpha_hat = y_oh + (1 - y_oh) * alpha
    kl = dirichlet_kl(alpha_hat)                            # (B,1)

    loss = sce + lam * kl                                   # (B,1)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.squeeze(-1)


def alpha_from_evidence_logits(logits):
    """
    Convert raw logits (B,C) to Dirichlet parameters alpha via evidence = softplus(logits), alpha = evidence + 1.
    """
    evidence = F.softplus(logits)                           # >= 0
    alpha = evidence + 1.0
    return alpha

def alpha_to_probs(alpha):
    """
    Expected probabilities under Dirichlet: E[p] = alpha / sum(alpha).
    """
    return alpha / torch.sum(alpha, dim=-1, keepdim=True)

def alpha_to_log_probs(alpha, eps: float = 1e-8):
    probs = alpha_to_probs(alpha).clamp_min(eps)
    return torch.log(probs)

def dirichlet_uncertainty(alpha):
    """
    Return (belief, uncertainty) where belief b_k = e_k / S, uncertainty u = K / S.
    """
    K = alpha.size(-1)
    S = torch.sum(alpha, dim=-1, keepdim=True)
    evidence = alpha - 1.0
    belief = evidence / S
    uncertainty = K / S
    return belief, uncertainty
