import os
import os.path as osp
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

os.environ['PYTHONPATH'] = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, os.environ['PYTHONPATH'])

from dataset import load_entity_embeddings, load_embedding_splits

DATASET_PATH = "../../data/FB15k-237"
RESULTS_DIR = "results"

def ensure_results_dir():
    if not osp.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

def visualize_embeddings():
    ensure_results_dir()
    
    # Load entity embeddings
    entity_embs = load_entity_embeddings(dataset_path=DATASET_PATH)
    splits = load_embedding_splits(dataset_path=DATASET_PATH)
    
    n_entities = len(entity_embs)
    dim = entity_embs.shape[1]
    
    print(f"Entity embeddings: {entity_embs.shape}")
    
    # Generate 10x more random embeddings
    random_embs = torch.randn(n_entities * 2, dim)
    
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
    visualize_embeddings()
