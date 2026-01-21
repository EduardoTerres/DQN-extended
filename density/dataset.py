import os
import os.path as osp
import sys

os.environ['PYTHONPATH'] = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, os.environ['PYTHONPATH'])

import pickle

import torch

from kbc.utils import QuerDAG
from kbc.utils import preload_env


def load_entity_embeddings(
    dataset_path="../data/FB15k-237",
    model_path="../models/FB15k-237-model-rank-1000-epoch-100-1602508358.pt",
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
    dataset_path="../data/FB15k-237",
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


def load_embedding_splits(dataset_path="../data/FB15k-237", filename="embedding_splits.pt"):
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

if __name__ == "__main__":
    # Example usage
    entity_embeddings = load_entity_embeddings()
    print(f"Loaded {entity_embeddings.shape[0]} entity embeddings of dimension {entity_embeddings.shape[1]}.")

    splits = save_embedding_splits(entity_embeddings)
    for split_name, indices in splits.items():
        print(f"{split_name.capitalize()} split: {len(indices)} entities.")
    
    print("Splits saved successfully.")
