import os
import os.path as osp
import sys

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

os.environ['PYTHONPATH'] = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, os.environ['PYTHONPATH'])

from flow import FlowMatching
from dataset import load_entity_embeddings, load_embedding_splits


MODEL_PATH = "results/flow_model_epoch_0.pt"
DATASET_PATH = "../../data/FB15k-237"
RESULTS_DIR = "results"

def load_flow_model(model_path=MODEL_PATH, device='mps'):
    checkpoint = torch.load(model_path, map_location=device)
    dim = checkpoint['net.0.weight'].shape[1] - 1
    model = FlowMatching(dim=dim)
    model.load_state_dict(checkpoint)
    return model.to(device).eval(), dim


def compute_likelihood(embeddings, flow_model, device='mps', n_steps=100, batch_size=256, show_progress=True, desc="likelihood"):
    embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
    all_likelihoods = []
    iterator = tqdm(range(0, len(embeddings), batch_size), desc=f"Computing {desc}") if show_progress else range(0, len(embeddings), batch_size)
    for i in iterator:
        batch = embeddings[i:i + batch_size]
        likelihoods = flow_model.bits_per_dim(batch)
        all_likelihoods.append(likelihoods.cpu())
    
    return torch.cat(all_likelihoods)


def get_stats(likelihoods, name=""):
    stats = {'mean': likelihoods.mean().item(), 'std': likelihoods.std().item(),
             'min': likelihoods.min().item(), 'max': likelihoods.max().item()}
    if name:
        print(f"{name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, min={stats['min']:.6f}, max={stats['max']:.6f}")
    return stats

def ensure_results_dir():
    if not osp.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)


def analyze_checkpoint(model_path=MODEL_PATH, dataset_path=DATASET_PATH, device=None):
    device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    ensure_results_dir()

    flow_model, dim = load_flow_model(model_path=model_path, device=device)
    entity_embeddings = load_entity_embeddings(dataset_path=dataset_path)
    splits = load_embedding_splits(dataset_path=dataset_path)

    print(f"Loaded {entity_embeddings.shape[0]} embeddings (dim={dim})")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}\n")

    # Likelihoods per split
    all_likelihoods = []
    for split_name, indices in splits.items():
        split_embeddings = entity_embeddings[indices].detach().cpu().numpy()
        likelihoods = compute_likelihood(split_embeddings, flow_model, device=device)
        get_stats(likelihoods, split_name.upper())
        all_likelihoods.append(likelihoods)

    real_all = torch.cat(all_likelihoods)
    print(f"Overall: mean={real_all.mean().item():.6f}\n")

    # Random baseline
    random_vectors = torch.randn(len(entity_embeddings), dim)
    random_likelihoods = compute_likelihood(random_vectors.numpy(), flow_model, device=device)
    baseline = get_stats(random_likelihoods, "Random")

    diff = real_all.mean().item() - baseline['mean']
    print(f"Real vs Random difference: {diff:.6f}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(real_all.numpy(), alpha=0.7, label='Real', density=True)
    axes[0].hist(random_likelihoods.numpy(), alpha=0.7, label='Random', density=True)
    axes[0].axvline(real_all.mean(), color='blue', linestyle='--', alpha=0.5)
    axes[0].axvline(random_likelihoods.mean(), color='orange', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Bits per Dimension')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distribution of Likelihood in Bits per Dimension')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].boxplot([real_all.numpy(), random_likelihoods.numpy()], labels=['Real', 'Random'])
    axes[1].set_ylabel('Bits per Dimension')
    axes[1].set_title('Bits per Dimension Comparison')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = osp.join(RESULTS_DIR, 'likelihood_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to {out_path}")
    plt.close(fig)


def plot_checkpoint_evolution(dataset_path=DATASET_PATH, device=None):
    device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    ensure_results_dir()

    epochs = [0] + list(range(100, 1100, 100)) + list(range(2000, 10001, 1000))

    # Use validation split to monitor shift
    entity_embeddings = load_entity_embeddings(dataset_path=dataset_path)
    splits = load_embedding_splits(dataset_path=dataset_path)
    val_embeddings = entity_embeddings[splits['val']].detach().cpu().numpy()

    # Collect likelihoods
    epoch_likelihoods = []
    epoch_random = []
    for epoch in tqdm(epochs, desc="Checkpoints", unit="ckpt"):
        model_path = osp.join(RESULTS_DIR, f"flow_model_epoch_{epoch}.pt")
        if not osp.exists(model_path):
            print(f"Skip epoch {epoch}: checkpoint missing at {model_path}")
            continue
        flow_model, dim = load_flow_model(model_path=model_path, device=device)
        lks = compute_likelihood(val_embeddings, flow_model, device=device, show_progress=False)
        epoch_likelihoods.append((epoch, lks))
        rand = torch.randn(len(val_embeddings), dim)
        rand_lks = compute_likelihood(rand.numpy(), flow_model, device=device, show_progress=False)
        epoch_random.append((epoch, rand_lks))

    if not epoch_likelihoods:
        print("No checkpoints found for evolution plot.")
        return

    # Shared bins for fair comparison
    all_vals = torch.cat([lk for _, lk in epoch_likelihoods] + [lk for _, lk in epoch_random])
    x_min, x_max = all_vals.min().item(), all_vals.max().item()
    bins = np.linspace(x_min, x_max, 100)  # 20 bins

    fig, axes = plt.subplots(len(epoch_likelihoods), 1, figsize=(10, 2 * len(epoch_likelihoods)), sharex=True)
    if len(epoch_likelihoods) == 1:
        axes = [axes]

    for ax, (epoch, lks) in zip(axes, epoch_likelihoods):
        rand_lks = dict(epoch_random)[epoch]
        ax.hist(lks.numpy(), bins=bins, alpha=0.7, color='steelblue', density=True, label='val')
        ax.hist(rand_lks.numpy(), bins=bins, alpha=0.5, color='orange', density=True, label='rand')
        ax.axvline(lks.mean(), color='darkred', linestyle='--', alpha=0.8)
        ax.axvline(rand_lks.mean(), color='olive', linestyle='--', alpha=0.8)
        ax.set_ylabel(f"ep {epoch}", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    axes[-1].set_xlabel('Bits per Dimension (shared scale)', fontsize=11)
    fig.suptitle('Likelihood Shift Across Checkpoints', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = osp.join(RESULTS_DIR, 'likelihood_evolution.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Evolution plot saved to {out_path}")
    plt.close(fig)


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    analyze_checkpoint(model_path=MODEL_PATH, dataset_path=DATASET_PATH, device=device)
    plot_checkpoint_evolution(dataset_path=DATASET_PATH, device=device)


if __name__ == '__main__':
    main()
