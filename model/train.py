import numpy as np
import pandas as pd
import torch
import pickle
import random
import math
from tqdm import tqdm
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import logging
import os
import argparse
import sys
sys.modules['numpy._core'] = np.core
sys.modules['numpy._core.numeric'] = np.core.numeric


from evidential_DL import (
    dirichlet_evidence_loss,
    alpha_from_evidence_logits,
    alpha_to_log_probs,
    dirichlet_uncertainty,
)

drugcol = 'SMILES'
targetcol = 'Protein'
labelcol = 'Y'

device = torch.device("cuda:0")

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

# ====== pad/trim with mask ======
def pad_or_trim_with_mask(x_np: np.ndarray, max_len: int):
    """
    x_np: (L, D) numpy
    returns:
      feat:  torch.float32 (max_len, D)  # zero-padded if needed
      mask:  torch.bool (max_len,)       # True for valid positions
    """
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
        self.drug_features = drug_features            # {drug_name: 1D torch/numpy}
        self.target_features = target_features        # {target_name: 2D torch (Lmax,D)}
        self.target_masks = target_masks              # {target_name: 1D torch.bool (Lmax,)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i: int):
        drug_name = self.drugs[i]
        target_name = self.targets[i]
        drug = self.drug_features[drug_name]
        if not torch.is_tensor(drug):
            drug = torch.tensor(drug, dtype=torch.float32)
        target = self.target_features[target_name]   # (Lmax, D) torch.float32
        mask = self.target_masks[target_name]        # (Lmax,)  torch.bool
        label = self.labels[i]
        return drug, target, mask, label, drug_name, target_name

def my_collate_fn(batch):
    d_emb, t_emb, t_msk, labels, drug_names, target_names = zip(*batch)
    drugs = torch.stack([x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32) for x in d_emb], dim=0)  # (B, drug_dim)
    targets = torch.stack(t_emb, dim=0)  # (B, Lmax, D)
    masks = torch.stack(t_msk, dim=0)    # (B, Lmax)
    labels = torch.stack(labels, dim=0)
    # names remain python lists of strings
    return drugs, targets, masks, labels, list(drug_names), list(target_names)

def load_data(datadir, drug, target, batch_size, target_max_len=550):
    train = pd.read_csv(f'{datadir}/train.csv')
    val = pd.read_csv(f'{datadir}/val.csv')
    test = pd.read_csv(f'{datadir}/test.csv')

    with open(f'{datadir}/{drug}.pkl', 'rb') as f:
        drugdict = pickle.load(f)    # {smiles: 1D embedding (np.ndarray)}

    with open(f'{datadir}/{target}.pkl', 'rb') as f:
        targetdict = pickle.load(f)  # {name: {"features": (L,D), "length":..., "sequence":...}}

    all_drugs = pd.concat([train[drugcol], val[drugcol], test[drugcol]]).unique()
    all_targets = pd.concat([train[targetcol], val[targetcol], test[targetcol]]).unique()

    drug_features_map = {}
    for d in all_drugs:
        feat = drugdict[d]  # 1D numpy array
        drug_features_map[d] = torch.tensor(feat, dtype=torch.float32)

    target_features_map, target_masks_map = {}, {}
    for t in all_targets:
        feat = targetdict[t]["features"]  # (L, D), numpy
        feat_pad, mask = pad_or_trim_with_mask(feat, target_max_len)
        target_features_map[t] = feat_pad
        target_masks_map[t] = mask

    train_dataset = CLSimpleDataset(
        train[drugcol], train[targetcol], train[labelcol],
        drug_features_map, target_features_map, target_masks_map
    )
    val_dataset = CLSimpleDataset(
        val[drugcol], val[targetcol], val[labelcol],
        drug_features_map, target_features_map, target_masks_map
    )
    test_dataset = CLSimpleDataset(
        test[drugcol], test[targetcol], test[labelcol],
        drug_features_map, target_features_map, target_masks_map
    )

    train_generator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=my_collate_fn, pin_memory=True)
    val_generator   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn, pin_memory=True)
    test_generator  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn, pin_memory=True)
    
    return train_generator, val_generator, test_generator



def evaluate(model, dataloader, verbose: bool, device, savefolder=None):
    """
    Evaluate model on dataloader, compute metrics.
    Returns classification metrics dictionary. Requires model.predict() to return
    (log_probs, d_q, t_q, alpha, d_perp, t_perp) for evidential uncertainty.
    """
    model.eval()
    prob_list, pred_list, gold_list = [], [], []
    d_q_list, t_q_list = [], []

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
            d_q_list.append(d_q.cpu())
            t_q_list.append(t_q.cpu())

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
        d_q_arr = torch.cat(d_q_list).numpy()
        t_q_arr = torch.cat(t_q_list).numpy()

    # Save representations and per-sample CSV if requested
    if savefolder is not None:
        os.makedirs(savefolder, exist_ok=True)
        np.save(os.path.join(savefolder, 'd_q_list.npy'), d_q_arr)
        np.save(os.path.join(savefolder, 't_q_list.npy'), t_q_arr)

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
        df.to_csv(os.path.join(savefolder, 'test_predictions_with_uncertainty.csv'), index=False)

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


# ====== Light-Attention encoder (kernel_size=9) ======
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
        d_z = self.drug_encoder(d_emb)  # 通过drug_encoder
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
    parser.add_argument('--datadir', type=str, default='data_random')
    parser.add_argument('--drug', type=str, default='drug_bmfm')
    parser.add_argument('--target', type=str, default='target_esmc')
    parser.add_argument('--random', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epoch', type=int, default=20)
    parser.add_argument('--print_every_n_epoch', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=2e-5)
    parser.add_argument('--use_metric', type=str, default='MCC') # MCC for random split, AUROC for scaffold split
    parser.add_argument('--drug_n_embed', type=int, default=128)
    parser.add_argument('--target_n_embed', type=int, default=128)
    parser.add_argument('--evd_lambda', type=float, default=0.5, help='Base lambda for KL term in evidential loss')
    parser.add_argument('--evd_anneal_epochs', type=int, default=10, help='Linear warm-up epochs for evidential KL weight')
    parser.add_argument('--target_max_len', type=int, default=520)
    args = parser.parse_args()

    random_seed = args.random
    dataset = os.path.basename(args.datadir)
    running_folder = os.path.join(os.path.dirname(__file__), 'results', dataset, f'{args.drug}-{args.target}', str(args.random))
    os.makedirs(running_folder, exist_ok=True)

    logging.basicConfig(filename=os.path.join(running_folder, 'log.txt'), level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.info(f"Random seed: {random_seed}")
    logger.info(f"Running on {device}")
    logger.info(f"Dataset: {args.datadir}")
    logger.info(f"Batch size: {args.batch_size}")

    set_seed(random_seed)

    feature_len = {
        'target_esmc': 1152,
        'drug_bmfm': 1236,
    }

    train_generator, val_generator, test_generator = load_data(args.datadir, args.drug, args.target, args.batch_size, target_max_len=args.target_max_len)

    CODE_DIM = 4 
    model = CYPClassifier(
        drug_embed_dim=feature_len[args.drug],
        target_embed_dim=feature_len[args.target],
        drug_codebook_size=args.drug_n_embed,
        target_codebook_size=args.target_n_embed,
        code_dim=CODE_DIM, n_class=2
    ).to(device)

    logger.info(f"Model: {model}")
    init_metric = evaluate(model, val_generator, verbose=True, device=device)

    n_epoch = args.n_epoch
    print_every_n_epoch = args.print_every_n_epoch
    weight_decay = args.weight_decay
    use_metric = args.use_metric

    logger.info(f"Number of epochs: {n_epoch}")
    logger.info(f"Weight decay: {weight_decay}")
    logger.info(f"Use metric: {use_metric}")

    best_model = None
    best_epoch = 0
    best_metric = -1

    max_lr = 1e-3
    min_lr = 1e-4

    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    ramp_epochs = min(20, n_epoch)
    decay_epochs = max(n_epoch - ramp_epochs, 1)

    def lr_lambda(epoch: int) -> float:
        t = epoch / n_epoch  # 0..1
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * t))
        return lr / max_lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    

    def evd_lambda_at_epoch(epo: int) -> float:
        if args.evd_anneal_epochs <= 0:
            return args.evd_lambda
        ramp = min(1.0, max(0.0, float(epo) / float(args.evd_anneal_epochs)))
        return args.evd_lambda * ramp

    for epo in tqdm(range(1, n_epoch + 1)):
        model.train()
        epoch_loss = 0.0
        epoch_class_loss = 0.0
        epoch_d_vq_loss = 0.0
        epoch_t_vq_loss = 0.0
        epoch_d_perp = 0.0
        epoch_t_perp = 0.0
        batch_count = 0
        lam_eff = evd_lambda_at_epoch(epo)

        for dr, tr, msk, label, _, _ in train_generator:
            dr = dr.to(device, non_blocking=True)
            tr = tr.to(device, non_blocking=True)
            msk = msk.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            loss, class_loss, d_vq_loss, t_vq_loss, d_perp, t_perp = model(dr, tr, msk, label, evd_lambda=lam_eff)

            epoch_loss += float(loss.item())
            epoch_class_loss += float(class_loss.item())
            epoch_d_vq_loss += float(d_vq_loss.item())
            epoch_t_vq_loss += float(t_vq_loss.item())

            epoch_d_perp += float(d_perp.item() if torch.is_tensor(d_perp) else d_perp)
            epoch_t_perp += float(t_perp.item() if torch.is_tensor(t_perp) else t_perp)
            batch_count += 1

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        scheduler.step()

        if batch_count > 0:
            avg_d_perp = epoch_d_perp / batch_count
            avg_t_perp = epoch_t_perp / batch_count
        else:
            avg_d_perp = avg_t_perp = float('nan')

        verbose = (epo - 1) % print_every_n_epoch == 0
        if verbose:
            logger.info("=" * 5 + f' Epoch {epo} | total {epoch_loss:.4f} | class {epoch_class_loss:.4f} | dVQ {epoch_d_vq_loss:.4f} | tVQ {epoch_t_vq_loss:.4f} | lam_eff {lam_eff:.4f} ' + "=" * 5)
            logger.info(f"Epoch {epo} codebook perplexity - Drug: {avg_d_perp:.4f} | Target: {avg_t_perp:.4f}")
        val_metrics = evaluate(model, val_generator, verbose=verbose, device=device)

        current_metric = val_metrics[use_metric]

        if current_metric > best_metric:
            best_epoch = epo
            best_metric = current_metric
            best_model = model.state_dict()

    logger.info('=' * 10 + 'Testing' + '=' * 10)
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best metric: {best_metric}")

    model.load_state_dict(best_model)
    test_metrics = evaluate(model, test_generator, verbose=True, device=device, savefolder=running_folder)
    torch.save(model.state_dict(), os.path.join(running_folder, 'model.pth'))
