import os
import os.path as osp
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

os.environ['PYTHONPATH'] = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, os.environ['PYTHONPATH'])

from dataset import (
    load_entity_embeddings,
    load_embedding_splits,
    create_random_vectors,
)

DATASET_PATH = "../../data/FB15k-237"
RESULTS_DIR = "root/results"

def plot_norms_distribution(vectors, label, filename):
    """
    Compute and plot the distribution of norms of embedding vectors.
    
    Args:
        vectors: Tensor or array of embedding vectors (n_samples, dim)
        label: String label for the plot title (e.g., 'Entity Embeddings')
        filename: Output filename for the plot
    """
    ensure_results_dir()
    
    # Convert to numpy if tensor
    if isinstance(vectors, torch.Tensor):
        vectors = vectors.detach().cpu().numpy()
    
    print(f"\n{label}:")
    print(f"  Shape: {vectors.shape}")
    
    # Compute norms
    norms = np.linalg.norm(vectors, axis=1)
    
    print(f"  Norm statistics:")
    print(f"    Mean: {np.mean(norms):.4f}")
    print(f"    Std: {np.std(norms):.4f}")
    print(f"    Min: {np.min(norms):.4f}")
    print(f"    Max: {np.max(norms):.4f}")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram
    axes[0, 0].hist(norms, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].set_xlabel('L2 Norm')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'{label} - Distribution of Norms (Histogram)')
    axes[0, 0].grid(alpha=0.3)
    
    # KDE plot
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(norms)
    x_range = np.linspace(norms.min(), norms.max(), 200)
    axes[0, 1].plot(x_range, kde(x_range), 'steelblue', linewidth=2)
    axes[0, 1].fill_between(x_range, kde(x_range), alpha=0.3, color='steelblue')
    axes[0, 1].set_xlabel('L2 Norm')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title(f'{label} - Distribution of Norms (KDE)')
    axes[0, 1].grid(alpha=0.3)
    
    # Box plot
    axes[1, 0].boxplot(norms, vert=True)
    axes[1, 0].set_ylabel('L2 Norm')
    axes[1, 0].set_title(f'{label} - Distribution of Norms (Box Plot)')
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Cumulative distribution
    sorted_norms = np.sort(norms)
    cumulative = np.arange(1, len(sorted_norms) + 1) / len(sorted_norms)
    axes[1, 1].plot(sorted_norms, cumulative, 'steelblue', linewidth=2)
    axes[1, 1].set_xlabel('L2 Norm')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title(f'{label} - Cumulative Distribution of Norms')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    out_path = osp.join(RESULTS_DIR, filename)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved to {out_path}")
    plt.close()

def compare_norms_distributions():
    """Compare norm distributions of entity embeddings vs uniformly sampled vectors."""
    ensure_results_dir()
    
    # Load entity embeddings
    entity_embs = load_entity_embeddings(dataset_path=DATASET_PATH)
    
    # Generate uniformly sampled vectors with same shape
    n_samples, dim = entity_embs.shape
    uniform_vecs = torch.from_numpy(np.random.uniform(-1, 1, size=(n_samples, dim))).float()
    
    print(entity_embs.shape)
    exit()

    # Plot distributions
    plot_norms_distribution(entity_embs, 'Entity Embeddings', 'entity_embeddings_norms_distribution.png')
    plot_norms_distribution(uniform_vecs, 'Uniform Random Vectors', 'uniform_random_norms_distribution.png')

def ensure_results_dir():
    if not osp.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

def visualize_embeddings():
    ensure_results_dir()
    
    # Load entity embeddings
    entity_embs = load_entity_embeddings(dataset_path=DATASET_PATH)
    
    n_entities = len(entity_embs)
    dim = entity_embs.shape[1]
    
    print(f"Entity embeddings: {entity_embs.shape}")
    
    # Generate 10x more random embeddings
    entity_embs_norm = entity_embs.norm(dim=1).mean().item()
    entity_embs_std = entity_embs.std().item()
    random_embs = create_random_vectors(
        mean=entity_embs_norm,
        std=entity_embs_std,
        dim=dim,
        num_vectors=n_entities * 10
    )
    
    # Apply t-SNE
    print("Computing t-SNE...")
    combined = np.vstack([entity_embs.detach().cpu().numpy(), random_embs.numpy()])
    tsne = TSNE(n_components=2, random_state=42, n_iter=1000, perplexity=30)
    embedded = tsne.fit_transform(combined)
    
    real_embedded = embedded[:n_entities]
    rand_embedded = embedded[n_entities:]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(real_embedded[:, 0], real_embedded[:, 1], alpha=0.6, s=20, label='Entity', c='blue')
    ax.scatter(rand_embedded[:, 0], rand_embedded[:, 1], alpha=0.3, s=10, label='Random (10x)', c='orange')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Entity vs Random Embeddings (t-SNE)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    out_path = osp.join(RESULTS_DIR, 'tsne_embeddings.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {out_path}")
    plt.close()

if __name__ == '__main__':
    compare_norms_distributions()
    # Uncomment below to also run t-SNE visualization
    # visualize_embeddings()
