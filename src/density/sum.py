import os
import os.path as osp
from pathlib import Path
import sys

os.environ['PYTHONPATH'] = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, os.environ['PYTHONPATH'])

import torch
import torch.nn.functional as F
from tqdm import tqdm
from kbc.utils import preload_env, QuerDAG

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data" / "FB15k-237"
MODELS_PATH = REPO_ROOT / "models"


def load_env():
    """Load KBC environment with embeddings."""
    import pickle
    dataset_path = str(DATASET_PATH)
    model_path = str(MODELS_PATH / "FB15k-237-model-rank-1000-epoch-100-1602508358.pt")
    dataset = osp.basename(dataset_path)
    
    data_hard = pickle.load(open(osp.join(dataset_path, f"{dataset}_valid_hard.pkl"), "rb"))
    data_complete = pickle.load(open(osp.join(dataset_path, f"{dataset}_valid_complete.pkl"), "rb"))
    
    preload_env(model_path, data_hard, QuerDAG.TYPE1_1.value, mode="hard")
    env = preload_env(model_path, data_complete, QuerDAG.TYPE1_1.value, mode="complete")
    return env


def s(r1, e2, env, limit=None):
    """Sum over e1 of Complex(e1, r1, e2)."""
    e1_embeddings = env.kbc.model.embeddings[0].weight
    if limit:
        e1_embeddings = e1_embeddings[:limit]
    rel_embedding = env.kbc.model.embeddings[1].weight[r1:r1+1]
    e2_emb = env.kbc.model.embeddings[0].weight[e2:e2+1]
    
    scores, _ = env.kbc.model.score_emb(e1_embeddings, rel_embedding.expand_as(e1_embeddings), e2_emb.expand_as(e1_embeddings))
    scores = torch.sigmoid(scores)
    return torch.sum(scores)


def s_batch(rel_embeddings, e2_embeddings, e_embeddings, scoring_fn):
    """Sum over e1 of Complex(e1, r1, e2) for a batch of (r1, e2) pairs.
    
    Args:
        rel_embeddings: tensor of relation embeddings [batch_size, dim]
        e2_embeddings: tensor of e2 entity embeddings [batch_size, dim]
        e1_embeddings: tensor of all e1 entity embeddings [n_entities, dim]
        scoring_fn: function to score (e1, rel, e2) triples
    
    Returns:
        tensor of sums, one per (r1, e2) pair [batch_size]
    """
    batch_size = rel_embeddings.shape[0]
    n_entities = e_embeddings.shape[0]
    
    # Expand e1 for each item in batch: [batch_size, n_entities, dim]
    e1_batch = e_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
    rel_batch = rel_embeddings.unsqueeze(1).expand(-1, n_entities, -1)
    e2_batch = e2_embeddings.unsqueeze(1).expand(-1, n_entities, -1)
    
    # Reshape to [batch_size * n_entities, dim] for scoring
    e1_flat = e1_batch.reshape(-1, e1_batch.shape[-1])
    rel_flat = rel_batch.reshape(-1, rel_batch.shape[-1])
    e2_flat = e2_batch.reshape(-1, e2_batch.shape[-1])
    
    scores, _ = scoring_fn(e1_flat, rel_flat, e2_flat)
    scores = torch.sigmoid(scores)
    
    # Reshape back to [batch_size, n_entities] and sum over e1
    scores = scores.reshape(batch_size, n_entities)
    return torch.sum(scores, dim=1)


def s_full(r1, env, limit=None):
    """Sum over e1 and e2 of Complex(e1, r1, e2)."""
    rel_embedding = env.kbc.model.embeddings[1].weight[r1]
    e_embeddings = env.kbc.model.embeddings[0].weight
    if limit:
        e_embeddings = e_embeddings[:limit]
    
    n = e_embeddings.shape[0]
    # Reshape to compute all (e1, e2) pairs: [n*n, dim]
    e1_tiled = e_embeddings.repeat(n, 1)  # [n*n, dim]
    e2_tiled = e_embeddings.repeat_interleave(n, 0)  # [n*n, dim]
    rel_tiled = rel_embedding.expand(n*n, -1)  # [n*n, dim]
    
    scores, _ = env.kbc.model.score_emb(e1_tiled, rel_tiled, e2_tiled)
    scores = torch.sigmoid(scores)
    return torch.sum(scores)


def s_full_batch(rel_embeddings, e_embeddings, scoring_fn):
    """Sum over e1 and e2 of Complex(e1, r1, e2) for a batch of relations.
    
    Args:
        rel_embeddings: tensor of relation embeddings [batch_size, dim]
        e_embeddings: tensor of entity embeddings [n_entities, dim]
        scoring_fn: function to score (e1, rel, e2) triples
    
    Returns:
        tensor of sums, one per relation [batch_size]
    """
    batch_size = rel_embeddings.shape[0]
    n = e_embeddings.shape[0]
    
    # For each relation in batch, compute all (e1, e2) pairs
    # Result shape: [batch_size, n*n]
    e1_tiled = e_embeddings.repeat(n, 1)  # [n*n, dim]
    e2_tiled = e_embeddings.repeat_interleave(n, 0)  # [n*n, dim]
    
    # Expand for batch: [batch_size, n*n, dim]
    e1_batch = e1_tiled.unsqueeze(0).expand(batch_size, -1, -1)
    e2_batch = e2_tiled.unsqueeze(0).expand(batch_size, -1, -1)
    rel_batch = rel_embeddings.unsqueeze(1).expand(-1, n*n, -1)
    
    # Reshape to [batch_size * n*n, dim] for scoring
    e1_flat = e1_batch.reshape(-1, e1_batch.shape[-1])
    e2_flat = e2_batch.reshape(-1, e2_batch.shape[-1])
    rel_flat = rel_batch.reshape(-1, rel_batch.shape[-1])
    
    scores, _ = scoring_fn(e1_flat, rel_flat, e2_flat)
    scores = torch.sigmoid(scores)
    
    # Reshape back to [batch_size, n*n] and sum over entity pairs
    scores = scores.reshape(batch_size, n*n)
    return torch.sum(scores, dim=1)


def s_loop(r1, e2, env):
    """Sum over e1 of Complex(e1, r1, e2) using loop."""
    total = 0.0
    rel_emb = env.kbc.model.embeddings[1].weight[r1]
    e2_emb = env.kbc.model.embeddings[0].weight[e2]
    
    for e1 in tqdm(range(10), desc="s_loop"):
        e1_emb = env.kbc.model.embeddings[0].weight[e1]
        score = env.kbc.model.score_emb(e1_emb.unsqueeze(0), rel_emb.unsqueeze(0), e2_emb.unsqueeze(0))[0]
        total += torch.sigmoid(score).item()
    return total


def s_full_loop(r1, env):
    """Sum over e1 and e2 of Complex(e1, r1, e2) using loops."""
    total = 0.0
    rel_emb = env.kbc.model.embeddings[1].weight[r1]
    
    for e1 in tqdm(range(10), desc="s_full_loop e1"):
        for e2 in tqdm(range(10), desc="s_full_loop e2", leave=False):
            e1_emb = env.kbc.model.embeddings[0].weight[e1]
            e2_emb = env.kbc.model.embeddings[0].weight[e2]
            score = env.kbc.model.score_emb(e1_emb.unsqueeze(0), rel_emb.unsqueeze(0), e2_emb.unsqueeze(0))[0]
            total += torch.sigmoid(score).item()
    return total


def sanity_check():
    env = load_env()
    
    r1, e2 = 1, 5
    print(f"s(r={r1}, e={e2}, limit=10) = {s(r1, e2, env, limit=10)}")
    print(f"s_loop(r={r1}, e={e2}) = {s_loop(r1, e2, env)}")
    print(f"Match: {torch.allclose(torch.tensor(s(r1, e2, env, limit=10)), torch.tensor(s_loop(r1, e2, env)), atol=1e-5)}")
    
    print(f"\ns_full(r={r1}, limit=10) = {s_full(r1, env, limit=10)}")
    print(f"s_full_loop(r={r1}) = {s_full_loop(r1, env)}")
    print(f"Match: {torch.allclose(torch.tensor(s_full(r1, env, limit=10)), torch.tensor(s_full_loop(r1, env)), atol=1e-5)}")


def sanity_check_batch():
    """Test batch functions against their non-batch versions."""
    env = load_env()
    scoring_fn = env.kbc.model.score_emb
    limit = 10
    
    # Test s_full_batch
    print("Testing s_full_batch...")
    r1_indices = [1, 2, 3, 5]
    rel_embeddings = env.kbc.model.embeddings[1].weight[r1_indices]
    e_embeddings = env.kbc.model.embeddings[0].weight[:limit]
    batch_results = s_full_batch(rel_embeddings, e_embeddings, scoring_fn)
    individual_results = torch.stack([s_full(r1, env, limit=limit) for r1 in r1_indices])
    match = torch.allclose(batch_results, individual_results, atol=1e-5)
    print(f"match s_full_batch correct" if match else f"not-match s_full_batch failed\nDiff: {torch.abs(batch_results - individual_results)}")
    
    # Test s_batch
    print("\nTesting s_batch...")
    r1_indices = [1, 2, 3, 5]
    e2_indices = [5, 10, 15, 20]
    rel_embeddings = env.kbc.model.embeddings[1].weight[r1_indices]
    e2_embeddings = env.kbc.model.embeddings[0].weight[e2_indices]
    e1_embeddings = env.kbc.model.embeddings[0].weight[:limit]
    batch_results = s_batch(rel_embeddings, e2_embeddings, e1_embeddings, scoring_fn)
    individual_results = torch.stack([s(r1, e2, env, limit=limit) for r1, e2 in zip(r1_indices, e2_indices)])
    match = torch.allclose(batch_results, individual_results, atol=1e-5)
    print(f"match s_batch correct" if match else f"not-match s_batch failed\nDiff: {torch.abs(batch_results - individual_results)}")


def get_normalization_constant_batch(scoring_fn, r1_emb, all_entity_emb, e1_emb=None):
    """Sum over e1 and e2 of Complex(e1, r1, e2)."""
    if e1_emb is None:
        return s_full_batch(r1_emb, all_entity_emb, scoring_fn)
    else:
        return s_batch(r1_emb, e1_emb, all_entity_emb, scoring_fn)


if __name__ == "__main__":
    sanity_check_batch()
    # sanity_check()
