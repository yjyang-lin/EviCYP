import os
import sys
import argparse
import logging
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
sys.modules['numpy._core'] = np.core
sys.modules['numpy._core.numeric'] = np.core.numeric


from evidential_DL import (
    dirichlet_evidence_loss,
    alpha_from_evidence_logits,
    alpha_to_log_probs,
    dirichlet_uncertainty,
)

device = torch.device("cuda:1")

drugcol = 'SMILES'
targetcol = 'Protein'
labelcol = 'Y'

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def pad_or_trim_with_mask(x_np: np.ndarray, max_len: int):
    L, D = x_np.shape
    if L > max_len:
        x_np = x_np[:max_len, :]
        L = max_len
    out = np.zeros((max_len, D), dtype=x_np.dtype)
    out[:L] = x_np
    mask = np.zeros((max_len,), dtype=bool)
    mask[:L] = True
    return torch.from_numpy(out).float(), torch.from_numpy(mask)

class CLSimpleDataset(Dataset):
    def __init__(self, drugs: pd.Series, targets: pd.Series, labels: pd.Series,
                 drug_features: dict, target_features: dict, target_masks: dict):
        self.drugs = drugs.tolist()
        self.targets = targets.tolist()
        self.labels = torch.tensor(labels.values if isinstance(labels, pd.Series) else labels, dtype=torch.float32)
        self.drug_features = drug_features
        self.target_features = target_features
        self.target_masks = target_masks

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i: int):
        drug_name = self.drugs[i]
        target_name = self.targets[i]
        drug = self.drug_features[drug_name]
        if not torch.is_tensor(drug):
            drug = torch.tensor(drug, dtype=torch.float32)
        target = self.target_features[target_name]
        mask = self.target_masks[target_name]
        label = self.labels[i]
        return drug, target, mask, label, drug_name, target_name

def my_collate_fn(batch):
    d_emb, t_emb, t_msk, labels, drug_names, target_names = zip(*batch)
    drugs = torch.stack([x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32) for x in d_emb], dim=0)
    targets = torch.stack(t_emb, dim=0)
    masks = torch.stack(t_msk, dim=0)
    labels = torch.stack(labels, dim=0)
    return drugs, targets, masks, labels, list(drug_names), list(target_names)

def load_val_from_datadir(datadir, drug, target, batch_size, target_max_len=550, num_workers=0):
    val_csv = os.path.join(datadir, 'val.csv')
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Expected validation CSV at {val_csv} but not found.")
    # load pickles
    with open(f'{datadir}/{drug}.pkl', 'rb') as f:
        drugdict = pickle.load(f)
    with open(f'{datadir}/{target}.pkl', 'rb') as f:
        targetdict = pickle.load(f)

    df = pd.read_csv(val_csv)

    all_drugs = df[drugcol].unique()
    all_targets = df[targetcol].unique()

    drug_features_map = {}
    for d in all_drugs:
        feat = drugdict[d]
        drug_features_map[d] = torch.tensor(feat, dtype=torch.float32)

    target_features_map, target_masks_map = {}, {}
    for t in all_targets:
        feat = targetdict[t]["features"]
        feat_pad, mask = pad_or_trim_with_mask(feat, target_max_len)
        target_features_map[t] = feat_pad
        target_masks_map[t] = mask

    dataset = CLSimpleDataset(df[drugcol], df[targetcol], df[labelcol],
                              drug_features_map, target_features_map, target_masks_map)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn,
                        pin_memory=True, num_workers=num_workers)
    return loader


def evaluate(model, dataloader, verbose: bool, device, savefolder=None):
    """
    Evaluate model on dataloader, compute metrics.
    Returns classification metrics dictionary. Requires model.predict() to return
    (log_probs, d_q, t_q, alpha, d_perp, t_perp) for evidential uncertainty.
    """
    model.eval()
    prob_list, pred_list, gold_list = [], [], []

    # Per-sample storage
    smiles_all = []
    protein_all = []
    label_all = []
    prob_all = []
    pred_all = []
    uncertainty_all = []
    belief_all = []
    alpha_all = []

    # Batch-level perplexities
    d_perp_vals = []
    t_perp_vals = []

    with torch.no_grad():
        for batch in dataloader:
            # unpack batch
            dr, tr, msk, labels, drug_names, target_names = batch
            dr = dr.to(device)
            tr = tr.to(device)
            msk = msk.to(device)
            labels = labels.to(device)

            # expect (log_probs, d_q, t_q, alpha, d_perp, t_perp)
            log_probs, d_q, t_q, alpha, d_perp, t_perp = model.predict(dr, tr, msk)

            # probabilities / preds
            prob = torch.exp(log_probs)[:, 1]      # probability of class 1
            pred = torch.argmax(log_probs, dim=-1) # predicted class index

            # accumulate tensors for global metrics
            prob_list.append(prob.cpu())
            pred_list.append(pred.cpu())
            gold_list.append(labels.cpu())

            # collect perplexities (may be scalars or 0-d tensors)
            try:
                d_perp_vals.append(float(d_perp.item()) if torch.is_tensor(d_perp) else float(d_perp))
            except Exception:
                # fallback: mean if vector-like
                d_perp_vals.append(float(torch.mean(d_perp).cpu().item()) if torch.is_tensor(d_perp) else float(d_perp))
            try:
                t_perp_vals.append(float(t_perp.item()) if torch.is_tensor(t_perp) else float(t_perp))
            except Exception:
                t_perp_vals.append(float(torch.mean(t_perp).cpu().item()) if torch.is_tensor(t_perp) else float(t_perp))

            # per-sample info
            prob_all.extend(prob.cpu().numpy().tolist())
            pred_all.extend(pred.cpu().numpy().tolist())
            label_all.extend(labels.cpu().numpy().tolist())
            smiles_all.extend(drug_names)
            protein_all.extend(target_names)

            # evidential outputs -> compute belief & uncertainty
            belief, uncertainty = dirichlet_uncertainty(alpha)
            belief_all.extend(belief.cpu().numpy().tolist())                        # (B, C)
            uncertainty_all.extend(uncertainty.cpu().numpy().reshape(-1).tolist())  # (B,)
            alpha_all.extend(alpha.cpu().numpy().tolist())

        # concat accumulated lists
        prob_arr = torch.cat(prob_list).numpy()
        pred_arr = torch.cat(pred_list).numpy()
        gold_arr = torch.cat(gold_list).numpy()

    # Save representations and per-sample CSV if requested
    if savefolder is not None:
        os.makedirs(savefolder, exist_ok=True)

        df = pd.DataFrame({
            'SMILES': smiles_all,
            'Protein': protein_all,
            'Label': label_all,
            'PredProb_class1': prob_all,
            'PredLabel': pred_all,
            'Uncertainty': uncertainty_all,
            'Belief': belief_all,
            'Alpha': alpha_all,
        })
        df.to_csv(os.path.join(savefolder, 'val_predictions_with_uncertainty.csv'), index=False)

    if verbose:
        if len(d_perp_vals) > 0 or len(t_perp_vals) > 0:
            mean_d_perp = float(np.mean(d_perp_vals)) if len(d_perp_vals) > 0 else float('nan')
            mean_t_perp = float(np.mean(t_perp_vals)) if len(t_perp_vals) > 0 else float('nan')
            logger.info(f"Evaluate codebook perplexity - Drug: {mean_d_perp:.4f} | Target: {mean_t_perp:.4f}")
        else:
            logger.info("Evaluate codebook perplexity - no perplexity values collected.")

    # Compute classification metrics
    acc = metrics.accuracy_score(gold_arr, pred_arr)
    confusion = metrics.confusion_matrix(gold_arr, pred_arr)
    cfd = {'TP': confusion[1, 1], 'TN': confusion[0, 0], 'FP': confusion[0, 1], 'FN': confusion[1, 0]}
    precision = cfd['TP'] / (cfd['TP'] + cfd['FP']) if (cfd['TP'] + cfd['FP']) != 0 else 0.0
    recall = cfd['TP'] / (cfd['TP'] + cfd['FN']) if (cfd['TP'] + cfd['FN']) != 0 else 0.0
    specificity = cfd['TN'] / (cfd['TN'] + cfd['FP']) if (cfd['TN'] + cfd['FP']) != 0 else 0.0
    f1_score = 2 * precision * recall / (recall + precision) if (recall + precision) != 0 else 0.0
    fpr, tpr, _ = metrics.roc_curve(gold_arr, prob_arr)
    auroc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(gold_arr, prob_arr)
    mcc = metrics.matthews_corrcoef(gold_arr, pred_arr)

    metrics_dict = {
        "Accuracy": acc, "Precision": precision, "Recall": recall,
        "Specificity": specificity, "F1 Score": f1_score, 
        "AUROC": auroc, "AUPR": aupr, "MCC": mcc
    }

    if verbose:
        logger.info('; '.join([f'{k}: {v}' for k, v in cfd.items()]))
        for metric, value in metrics_dict.items():
            logger.info(f"{metric}: {value}")

    return metrics_dict


class HardVectorQuantizer(nn.Module):
    """
    EMA-based hard vector quantizer (keeps class name and return signature).
    Returns: (quantized_out, loss, encoding_indices, perplexity)
    """
    def __init__(self, embed_dim, codebook_size, code_dim,
                 commitment_cost: float = 0.1,
                 decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        assert embed_dim % code_dim == 0, \
            f"embed_dim={embed_dim} must be divisible by code_dim={code_dim}"
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.num_slices = embed_dim // code_dim

        # codebook: (K, code_dim)
        self.codebook = nn.Embedding(codebook_size, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        # EMA buffers (registered so they move with module/device)
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size, dtype=torch.float32))
        self.register_buffer('ema_embed_avg', self.codebook.weight.data.clone())

    def forward(self, z_e: torch.Tensor):
        """
        z_e: (B, embed_dim)
        returns:
            quantized_out: (B, embed_dim)  # straight-through quantized vectors
            loss: scalar tensor (commitment loss)
            encoding_indices: (B, num_slices) long tensor of indices
            perplexity: scalar float
        """
        B = z_e.shape[0]
        # (N, code_dim) where N = B * num_slices
        z_e_reshaped = z_e.view(-1, self.code_dim)

        # distances: (N, K)
        distances = torch.cdist(z_e_reshaped, self.codebook.weight, p=2)

        # hard assignment (nearest)
        encoding_indices = torch.argmin(distances, dim=1)  # (N,)

        # one-hot encodings: (N, K) - ensure dtype & device consistent with z_e_reshaped
        encodings = F.one_hot(encoding_indices, num_classes=self.codebook_size).to(z_e_reshaped.dtype)

        # avg probs & perplexity (hard assignment)
        avg_probs = torch.mean(encodings, dim=0)  # (K,)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # quantized vectors: (N, code_dim)
        quantized = torch.matmul(encodings, self.codebook.weight)

        # straight-through estimator
        quantized_st = z_e_reshaped + (quantized - z_e_reshaped).detach()

        # commitment loss (codebook updated via EMA, so use quantized.detach())
        commitment_loss = F.mse_loss(z_e_reshaped, quantized.detach()) * self.commitment_cost
        loss = commitment_loss

        # EMA updates (only during training)
        if self.training:
            with torch.no_grad():
                batch_cluster_size = torch.sum(encodings, dim=0)  # (K,)
                batch_embed_sum = torch.matmul(encodings.t(), z_e_reshaped)  # (K, code_dim)

                # EMA update
                self.ema_cluster_size.mul_(self.decay).add_(batch_cluster_size * (1.0 - self.decay))
                self.ema_embed_avg.mul_(self.decay).add_(batch_embed_sum * (1.0 - self.decay))

                # normalize with laplace smoothing (VQ-VAE(EMA) style)
                n = torch.sum(self.ema_cluster_size)
                # avoid zero division
                cluster_size = ((self.ema_cluster_size + self.eps) / (n + self.codebook_size * self.eps)) * n
                new_weight = self.ema_embed_avg / cluster_size.unsqueeze(1)

                # copy to codebook weights (in-place)
                self.codebook.weight.data.copy_(new_weight)

        # reshape outputs
        quantized_out = quantized_st.view(B, -1)
        encoding_indices = encoding_indices.view(B, self.num_slices).long()

        return quantized_out, loss, encoding_indices, perplexity


# ====== Light-Attention encoder (kernel_size=9)======
class TargetLAEncoder(nn.Module):
    """
    Input:  t_emb as (B, L, D), mask as (B, L) bool
    Internally:
      - transpose to (B, D, L)
      - feature conv1d (kernel=9), attention conv1d (kernel=9)
      - masked softmax over length
      - weighted-sum + global max-pool, concat -> linear proj to out_dim
    Output:
      (B, out_dim)  # out_dim == 1024
    """
    def __init__(self, in_channels: int, out_dim: int = 1024, hidden_dim: int = 768, kernel_size: int = 9, conv_dropout: float = 0.25, mlp_dropout: float = 0.25):
        super().__init__()
        padding = kernel_size // 2
        self.feature_conv = nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=padding)
        self.feature_bn = nn.BatchNorm1d(hidden_dim) 
        self.attn_conv = nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=padding)
        self.dropout = nn.Dropout(conv_dropout)

        self.proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(),
            nn.Dropout(mlp_dropout)
        )

    def forward(self, t_emb, mask):
        # t_emb: (B, L, D) -> (B, D, L)
        x = t_emb.transpose(1, 2)
        feat = self.feature_bn(self.feature_conv(x))      # (B, D, L)
        feat = self.dropout(feat)
        attn = self.attn_conv(x)                       # (B, D, L)

        # mask: (B, L) -> broadcast to (B, 1, L)
        if mask is not None:
            attn = attn.masked_fill(mask[:, None, :] == False, -1e9)

        weights = F.softmax(attn, dim=-1)             # (B, D, L)
        weighted = torch.sum(feat * weights, dim=-1)  # (B, D)

        # global max pooling with mask: set padded positions to -inf before max
        if mask is not None:
            neg_inf = torch.finfo(feat.dtype).min
            feat_masked = feat.masked_fill(mask[:, None, :] == False, neg_inf)
            gmax, _ = torch.max(feat_masked, dim=-1)  # (B, D)
        else:
            gmax, _ = torch.max(feat, dim=-1)         # (B, D)

        combined = torch.cat([weighted, gmax], dim=-1) # (B, 2D)
        out = self.proj(combined)                      # (B, out_dim)
        return out


class CYPClassifier(nn.Module):
    def __init__(self, drug_embed_dim, target_embed_dim,
                 drug_codebook_size, target_codebook_size,
                 code_dim, n_class):
        super(CYPClassifier, self).__init__()

        self.n_class = n_class
        self.drug_encoder = nn.Sequential(
                            nn.Linear(drug_embed_dim, 1024),
                            nn.BatchNorm1d(1024),
                            nn.LeakyReLU(),
                            nn.Dropout(0.25)
                        )
        # LA encoder for target (variable-length -> fixed vector of 1024)
        self.target_encoder = TargetLAEncoder(in_channels=target_embed_dim, hidden_dim=768 ,out_dim=1024, kernel_size=9)

        self.drug_vq = HardVectorQuantizer(embed_dim=1024, codebook_size=drug_codebook_size, code_dim=code_dim, commitment_cost=1.0)
        self.target_vq = HardVectorQuantizer(embed_dim=1024, codebook_size=target_codebook_size, code_dim=code_dim, commitment_cost=1.0)


        # Output EVIDENCE (non-negative); alpha = evidence + 1
        self.evidence_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, n_class) # logits for evidence; will pass through softplus inside alpha_from_evidence_logits
        )

    def _fusion(self, d_q, t_q):
        return torch.cat([d_q, t_q], dim=-1)  # (B, 2048)

    def forward(self, d_emb, t_emb, t_mask, y, evd_lambda: float = 0.1):
        # d_emb: (B, drug_embed_dim)
        # t_emb: (B, L, D), t_mask: (B, L)
        d_z = self.drug_encoder(d_emb)
        t_z = self.target_encoder(t_emb, t_mask)  # -> (B, 1024)

        d_q, d_vq_loss, _, d_perplexity = self.drug_vq(d_z)
        t_q, t_vq_loss, _, t_perplexity = self.target_vq(t_z)

        logits = self.evidence_head(self._fusion(d_q, t_q))         # (B, C)
        alpha = alpha_from_evidence_logits(logits)                   # (B, C), alpha = softplus(logits) + 1
        class_loss = dirichlet_evidence_loss(y, alpha, lam=evd_lambda, reduction='mean')

        loss = class_loss + d_vq_loss + t_vq_loss
        return loss, class_loss, d_vq_loss, t_vq_loss, d_perplexity, t_perplexity

    def predict(self, d_emb, t_emb, t_mask):
        d_z = self.drug_encoder(d_emb)
        t_z = self.target_encoder(t_emb, t_mask)
        d_q, _, _, d_perplexity = self.drug_vq(d_z)
        t_q, _, _, t_perplexity = self.target_vq(t_z)

        logits = self.evidence_head(self._fusion(d_q, t_q))  # (B, C)
        alpha = alpha_from_evidence_logits(logits)
        log_probs = alpha_to_log_probs(alpha)                # (B, C), log of expected probabilities
        return log_probs, d_q, t_q, alpha, d_perplexity, t_perplexity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data_val')
    parser.add_argument('--drug', type=str, default='drug_bmfm')
    parser.add_argument('--target', type=str, default='target_esmc')
    parser.add_argument('--random', type=int, default=0)
    BATCH_SIZE = 128
    parser.add_argument('--drug_n_embed', type=int, default=128)
    parser.add_argument('--target_n_embed', type=int, default=128)
    parser.add_argument('--code_dim', type=int, default=4)
    parser.add_argument('--model_path', type=str, required=True, help='model.pth file path')
    parser.add_argument('--target_max_len', type=int, default=550)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    # replicate original running_folder logic (save to results/... same as train script)
    random_seed = args.random
    dataset = os.path.basename(args.datadir)
    running_folder = os.path.join(os.path.dirname(__file__), 'results', dataset, f'{args.drug}-{args.target}', str(args.random))
    os.makedirs(running_folder, exist_ok=True)

    # write logs to running_folder/log.txt 
    logging.basicConfig(filename=os.path.join(running_folder, 'log.txt'), level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.info(f"Random seed: {random_seed}")
    logger.info(f"Running on {device}")
    logger.info(f"Dataset: {args.datadir}")
    logger.info(f"Batch size (fixed): {BATCH_SIZE}")
    logger.info("Use evidential: True (forced)")

    set_seed(random_seed)

    feature_len = {
        'target_esmc': 1152,
        'drug_bmfm': 1236,
    }

    # load validation set from datadir/val.csv
    val_loader = load_val_from_datadir(args.datadir, args.drug, args.target, BATCH_SIZE,
                                       target_max_len=args.target_max_len, num_workers=args.num_workers)

    CODE_DIM = args.code_dim
    model = CYPClassifier(
        drug_embed_dim=feature_len[args.drug],
        target_embed_dim=feature_len[args.target],
        drug_codebook_size=args.drug_n_embed,
        target_codebook_size=args.target_n_embed,
        code_dim=CODE_DIM, n_class=2
    ).to(device)


    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    logger.info("Model weights loaded.")

    # run inference and save outputs to running_folder
    metrics = evaluate(model, val_loader, device=device, savefolder=running_folder, verbose=True)
    logger.info("Inference finished.")
    logger.info(str(metrics))
    print("Inference finished. Results saved to:", running_folder)
