import os
import os.path as osp
import sys
from pathlib import Path

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

os.environ['PYTHONPATH'] = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, os.environ['PYTHONPATH'])

from density.dataset import load_entity_embeddings, load_embedding_splits, create_random_vectors
from density.flow import FlowMatching
from density.vae import VAE


DATASET = 'FB15k-237'

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH_FLOW = REPO_ROOT / "results" / f"flow_model_{DATASET}_2" / "flow_model_epoch_7000.pt"
MODEL_PATH_VAE = REPO_ROOT / "results" / f"vae_model_{DATASET}" / "vae_model_final.pt"

DATASET_PATH = REPO_ROOT / "data" / DATASET
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_IMAGE_DIR = RESULTS_DIR / "images"

MODEL_TYPE = "flow"  # "flow" or "vae"

def load_model(model_path: str, model_type: str, device: str = 'mps'):
    def _load_flow_model(model_path, device='mps'):
        checkpoint = torch.load(model_path, map_location=device)
        dim = checkpoint['net.0.weight'].shape[1] - 1
        model = FlowMatching(dim=dim)
        model.load_state_dict(checkpoint)
        return model.to(device), dim

    def _load_vae_model(model_path, device='mps'):
        # Load a VAE saved via state_dict
        checkpoint = torch.load(model_path, map_location=device)
        # Infer dimensions from weights
        input_dim = checkpoint['encoder.0.weight'].shape[1]
        hidden_dim = checkpoint['encoder.0.weight'].shape[0]
        latent_dim = checkpoint['fc_mu.weight'].shape[0]
        model = VAE(dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
        model.load_state_dict(checkpoint)
        return model.to(device), input_dim

    if model_type == 'flow':
        return _load_flow_model(model_path, device)
    elif model_type == 'vae':
        return _load_vae_model(model_path, device)
    else:
        raise ValueError(f"Unknown model type {model_type}. Supported types are 'flow' and 'vae'.")

def compute_likelihood(
        embeddings, model, model_type, device='mps',
        n_steps=100, batch_size=256, show_progress=True,
    ):
    def _compute_likelihood_flow(embeddings, flow_model, device='mps', n_steps=100, batch_size=256, show_progress=True):
        embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
        # Embeddings with norm 1
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        all_likelihoods = []
        iterator = tqdm(range(0, len(embeddings), batch_size), desc=f"Computing likelihood") if show_progress else range(0, len(embeddings), batch_size)
        for i in iterator:
            batch = embeddings[i:i + batch_size]
            likelihoods = flow_model.log_likelihood(batch, n_steps=n_steps)
            all_likelihoods.append(likelihoods)
        
        return torch.cat(all_likelihoods)

    def _compute_likelihood_vae(embeddings, vae_model, device='mps', batch_size=256, show_progress=True):
        all_likelihoods = []
        iterator = tqdm(range(0, len(embeddings), batch_size), desc=f"Computing ELBO") if show_progress else range(0, len(embeddings), batch_size)
        for i in iterator:
            batch = embeddings[i:i + batch_size]
            for x in batch:
                elbo = vae_model.elbo(x.unsqueeze(0))
                all_likelihoods.append(elbo)
        return torch.stack(all_likelihoods)

    if model_type == "flow":
        return _compute_likelihood_flow(embeddings, model, device, n_steps, batch_size, show_progress)
    elif model_type == "vae":
        return _compute_likelihood_vae(embeddings, model, device, batch_size, show_progress)
    else:
        raise ValueError(f"Unknown model type {type(model)}. Supported types are FlowMatching and VAE.")

def get_stats(likelihoods, name=""):
    stats = {'mean': likelihoods.mean().item(), 'std': likelihoods.std().item(),
             'min': likelihoods.min().item(), 'max': likelihoods.max().item()}
    if name:
        print(f"{name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, min={stats['min']:.6f}, max={stats['max']:.6f}")
    return stats

def ensure_results_dir():
    RESULTS_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def analyze_checkpoint(device):
    device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    ensure_results_dir()

    model_path = (
        MODEL_PATH_VAE if MODEL_TYPE == "vae" else MODEL_PATH_FLOW
    )
    model, dim = load_model(model_path=model_path, model_type=MODEL_TYPE, device=device)
    entity_embeddings = load_entity_embeddings(dataset_path=str(DATASET_PATH))
    splits = load_embedding_splits(dataset_path=str(DATASET_PATH))

    print(f"Loaded {entity_embeddings.shape[0]} embeddings (dim={dim})")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}\n")

    entity_embeddings = entity_embeddings.to(device)

    # Likelihoods per split
    all_likelihoods = []
    for split_name, indices in splits.items():
        split_embeddings = entity_embeddings[indices]
        likelihoods = compute_likelihood(split_embeddings, model, model_type=MODEL_TYPE, device=device)
        get_stats(likelihoods, split_name.upper())
        all_likelihoods.append(likelihoods)

    real_all = torch.cat(all_likelihoods)
    print(f"Overall: mean={real_all.mean().item():.6f}\n")

    # Random baseline
    entity_embs_norm = entity_embeddings.norm(dim=1).mean().item()
    entity_embs_std = entity_embeddings.std().item()
    random_vectors = create_random_vectors(
        mean=entity_embs_norm,
        std=entity_embs_std,
        unit_norm=True,
        dim=entity_embeddings.shape[1],
        num_vectors=entity_embeddings.shape[0]
    )

    random_vectors = random_vectors.to(device)

    random_likelihoods = compute_likelihood(random_vectors, model, model_type=MODEL_TYPE, device=device)
    baseline = get_stats(random_likelihoods, "Random")

    diff = real_all.mean().item() - baseline['mean']
    print(f"Real vs Random difference: {diff:.6f}")

    # Plots with shared bins
    all_vals = torch.cat([real_all, random_likelihoods])
    x_min, x_max = all_vals.min().item(), all_vals.max().item()
    bins = np.linspace(x_min, x_max, 100)

    # Convert to CPU numpy for plotting
    real_all = real_all.detach().cpu().numpy()
    random_likelihoods = random_likelihoods.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(real_all, bins=bins, alpha=0.7, label='Real', density=True)
    axes[0].hist(random_likelihoods, bins=bins, alpha=0.7, label='Random', density=True)
    axes[0].axvline(real_all.mean(), color='blue', linestyle='--', alpha=0.5)
    axes[0].axvline(random_likelihoods.mean(), color='orange', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Bits per Dimension')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distribution of Likelihood in Bits per Dimension')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].boxplot([real_all, random_likelihoods], labels=['Real', 'Random'])
    axes[1].set_ylabel('Bits per Dimension')
    axes[1].set_title('Bits per Dimension Comparison')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_IMAGE_DIR / 'likelihood_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to {out_path}")
    plt.close(fig)


def plot_checkpoint_evolution(device=None):
    device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    ensure_results_dir()

    epochs = [0] + list(range(100, 1100, 100)) + list(range(2000, 10001, 1000))

    # Use validation split to monitor shift
    entity_embeddings = load_entity_embeddings(dataset_path=str(DATASET_PATH))
    splits = load_embedding_splits(dataset_path=str(DATASET_PATH))
    val_embeddings = entity_embeddings[splits['val']]

    val_embeddings = val_embeddings.to(device)
    entity_embeddings = entity_embeddings.to(device)

    entity_embs_norm = entity_embeddings.norm(dim=1).mean().item()
    entity_embs_std = entity_embeddings.std().item()
    random_vectors = create_random_vectors(
        mean=entity_embs_norm,
        std=entity_embs_std,
        dim=entity_embeddings.shape[1],
        num_vectors=val_embeddings.shape[0]
    )

    random_vectors = random_vectors.to(device)

    # Collect likelihoods
    epoch_likelihoods = []
    epoch_random = []
    for epoch in tqdm(epochs, desc="Checkpoints", unit="ckpt"):
        model_path = MODEL_PATH_VAE if MODEL_TYPE == "vae" else MODEL_PATH_FLOW
        if not model_path.exists():
            print(f"Skip epoch {epoch}: checkpoint missing at {model_path}")
            continue
        model, _ = load_model(model_path=model_path, model_type=MODEL_TYPE, device=device)
        lks = compute_likelihood(val_embeddings, model, model_type=MODEL_TYPE, device=device, show_progress=False)
        epoch_likelihoods.append((epoch, lks))
        rand_lks = compute_likelihood(random_vectors, model, model_type=MODEL_TYPE, device=device, show_progress=False)
        epoch_random.append((epoch, rand_lks))

    if not epoch_likelihoods:
        print("No checkpoints found for evolution plot.")
        return

    all_vals = torch.cat([lk for _, lk in epoch_likelihoods] + [lk for _, lk in epoch_random])
    x_min, x_max = all_vals.min().item(), all_vals.max().item()
    bins = np.linspace(x_min, x_max, 100)

    fig, axes = plt.subplots(len(epoch_likelihoods), 1, figsize=(10, 2 * len(epoch_likelihoods)), sharex=True)
    if len(epoch_likelihoods) == 1:
        axes = [axes]

    for ax, (epoch, lks) in zip(axes, epoch_likelihoods):
        rand_lks = dict(epoch_random)[epoch]

        lks = lks.detach().cpu().numpy()
        rand_lks = rand_lks.detach().cpu().numpy()

        ax.hist(lks, bins=bins, alpha=0.7, color='steelblue', density=True, label='val')
        ax.hist(rand_lks, bins=bins, alpha=0.5, color='orange', density=True, label='rand')
        ax.axvline(lks.mean(), color='darkred', linestyle='--', alpha=0.8)
        ax.axvline(rand_lks.mean(), color='olive', linestyle='--', alpha=0.8)
        ax.set_ylabel(f"ep {epoch}", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    axes[-1].set_xlabel('Bits per Dimension (shared scale)', fontsize=11)
    fig.suptitle('Likelihood Shift Across Checkpoints', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = RESULTS_IMAGE_DIR / 'likelihood_evolution.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Evolution plot saved to {out_path}")
    plt.close(fig)


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    analyze_checkpoint(device=device)
    # plot_checkpoint_evolution(device=device)


if __name__ == '__main__':
    main()
