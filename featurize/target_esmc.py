from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import torch
import pandas as pd
import logging
import pickle
from tqdm import tqdm
import os
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model once (cpu)
client = ESMC.from_pretrained("esmc_600m").to("cpu")

MAX_LEN = 520  # Maximum sequence length for processing

def get_emb(seq: str):
    """
    Return torch.Tensor with shape = (L, embed_dim), where L = len(seq) (<= MAX_LEN)
    """
    seq = seq[:MAX_LEN]  # Truncate sequence to maximum length
    protein = ESMProtein(sequence=seq)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    emb = logits_output.embeddings[0][1:-1]  # Remove BOS/EOS tokens
    return emb  # torch.Tensor (L, embed_dim)

def read_data(datadir, sep=','):
    """
    Read data from datadir/protein.csv which must contain at least two columns: name, sequence
    """
    path = os.path.join(datadir, "protein.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"protein.csv not found in {datadir}")
    protein_df = pd.read_csv(path, sep=sep)
    if 'name' not in protein_df.columns or 'sequence' not in protein_df.columns:
        raise ValueError("protein.csv must contain columns 'name' and 'sequence'")
    entities = protein_df['name'].unique()
    logger.info(f'{len(entities)} unique entities found')
    name_to_seq = dict(zip(protein_df['name'], protein_df['sequence']))
    return entities, name_to_seq

def featurize_data(entities, name_to_seq):
    """
    Return dict: name -> {
        'sequence': seq,
        'length': L,
        'features': np.array with shape (L, embed_dim)
    }
    """
    out = {}
    example_embed_dim = None

    for name in tqdm(entities):
        seq = str(name_to_seq[name]).strip().upper()
        seq_trunc = seq[:MAX_LEN]
        L = len(seq_trunc)

        # Get embedding
        emb = get_emb(seq_trunc)  # torch.Tensor (L, embed_dim)
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)

        embed_np = emb.cpu().detach().numpy().astype(np.float32)
        # Truncate to L if model returns longer sequence
        if embed_np.shape[0] > L:
            embed_np = embed_np[:L, :]
        embed_dim = embed_np.shape[1]
        if example_embed_dim is None:
            example_embed_dim = embed_dim

        out[name] = {
            'sequence': seq_trunc,
            'length': L,
            'features': embed_np
        }

    logger.info(f"Featurization done. embed_dim={example_embed_dim}")
    return out


if __name__ == '__main__':
    feature = os.path.basename(__file__)[:-3]  # Extract feature name from script filename
    datadir = "./data_random"
    logger.info(f"Reading data from {datadir}")

    entities, name_to_seq = read_data(datadir)
    out = featurize_data(entities, name_to_seq)

    out_path = f'{datadir}/{feature}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Saved features for {len(out)} entities to {out_path}")