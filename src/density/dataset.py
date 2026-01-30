import os
import os.path as osp
from pathlib import Path
import sys

os.environ['PYTHONPATH'] = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, os.environ['PYTHONPATH'])

import pickle

import torch

from kbc.utils import QuerDAG
from kbc.utils import preload_env

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data" / "FB15k-237"
MODELS_PATH = REPO_ROOT / "models"


def load_entity_embeddings(
    dataset_path=f"{DATASET_PATH}",
    model_path=f"{MODELS_PATH}/FB15k-237-model-rank-1000-epoch-100-1602508358.pt",
    split="valid",
    chain_type=QuerDAG.TYPE1_1.value,
):
    """Load entity embeddings tensor from KBC environment."""
    dataset = osp.basename(dataset_path)
    data_hard_path = osp.join(dataset_path, f"{dataset}_{split}_hard.pkl")
    data_complete_path = osp.join(dataset_path, f"{dataset}_{split}_complete.pkl")

    data_hard = pickle.load(open(data_hard_path, "rb"))
    data_complete = pickle.load(open(data_complete_path, "rb"))

    # Instantiate singleton KBC object
    preload_env(model_path, data_hard, chain_type, mode="hard")
    env = preload_env(model_path, data_complete, chain_type, mode="complete")

    entity_embedding_weights = env.kbc.model.embeddings[0].weight
    return entity_embedding_weights


def _create_random_split(entity_embedding_weights, train_ratio, val_ratio, seed=987):
    """Create random train/val/test split indices."""
    torch.manual_seed(seed)
    print(f"Set torch seed to {seed}.")

    n_entities = entity_embedding_weights.shape[0]
    indices = torch.randperm(n_entities)

    train_size = int(n_entities * train_ratio)
    val_size = int(n_entities * val_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    splits = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }
    return splits


def save_embedding_splits(
    entity_embedding_weights,
    dataset_path=f"{DATASET_PATH}",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,  # Sums to 1.0
    seed=987,
    filename="embedding_splits.pt",
):
    """Save splits file once; if it exists, do nothing. Returns splits."""
    splits_path = osp.join(dataset_path, filename)
    splits = _create_random_split(entity_embedding_weights, train_ratio, val_ratio, seed)
    torch.save(splits, splits_path)
    return splits


def load_embedding_splits(dataset_path=f"{DATASET_PATH}", filename="embedding_splits.pt"):
    """Load previously saved splits from disk. Saves them if not existing."""
    splits_path = osp.join(dataset_path, filename)
    if not osp.exists(splits_path):
        splits = save_embedding_splits(
            load_entity_embeddings(dataset_path=dataset_path),
            dataset_path=dataset_path,
            filename=filename,
        )
        return splits

    return torch.load(splits_path)

def create_random_vectors(mean=0.0, std=1.0, unit_norm=True, dim=100, num_vectors=1000):
    """Create random vectors of norm 2."""
    vectors = std * torch.randn(num_vectors, dim)
    if unit_norm:
        return vectors / vectors.norm(dim=1, keepdim=True)
    else:
        return mean * (vectors / vectors.norm(dim=1, keepdim=True))

def assert_all_chain_types_match(dataset_path, model_path, split="valid"):
    """Verify all chain types produce identical entity embeddings."""
    embeddings = {}
    for chain_type in QuerDAG:
        print(f"Loading chain type: {chain_type.value}")
        try:
            embeddings[chain_type.value] = load_entity_embeddings(dataset_path, model_path, split, chain_type.value)
        except Exception as e:
            print(f"  Error: {e}")
    
    if not embeddings:
        raise ValueError("No valid chain types found")
    
    ref_emb = next(iter(embeddings.values()))
    for chain_type, emb in embeddings.items():
        assert torch.allclose(ref_emb, emb), f"Mismatch: {chain_type}"
    
    print(f"All {len(embeddings)} chain types match")

if __name__ == "__main__":
    assert_all_chain_types_match(
        dataset_path=str(DATASET_PATH),
        model_path=str(MODELS_PATH / "FB15k-237-model-rank-1000-epoch-100-1602508358.pt"),
        split="valid"
    )
    
    # Example usage
    entity_embeddings = load_entity_embeddings()
    print(f"Loaded {entity_embeddings.shape[0]} entity embeddings of dimension {entity_embeddings.shape[1]}.")

    splits = save_embedding_splits(entity_embeddings)
    for split_name, indices in splits.items():
        print(f"{split_name.capitalize()} split: {len(indices)} entities.")
    
    print("Splits saved successfully.")
