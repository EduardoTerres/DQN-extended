import os
import os.path as osp
from pathlib import Path
import sys

# for lazy ppl
os.environ['PYTHONPATH'] = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, os.environ['PYTHONPATH'])

import argparse
import random

import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from density.dataset import load_entity_embeddings, load_embedding_splits

WANDB_PROJECT = "ML4Graphs"
WANDB_ENTITY = "scale-gmns"
EXPERIMENT_NAME = "flow-matching-cqd"


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = "FB15k"
DATASET_PATH = REPO_ROOT / "data" / DATASET
RESULTS_DIR = REPO_ROOT / "results"

DEFAULT_SEED = 987


def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility across torch, numpy, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FlowMatching(nn.Module):
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
    
    def forward(self, x, t):
        """Predict velocity field at time t"""
        inp = torch.cat([x, t.view(-1, 1)], dim=1)
        output = self.net(inp)
        # L2 normalize each vector in the batch to have unit norm
        output = torch.nn.functional.normalize(output, p=2, dim=1)
        return output
    
    def sample(self, n_samples, device='cpu', n_steps=100):
        """Sample from the model using Euler integration"""
        x = torch.randn(n_samples, self.dim, device=device)
        dt = 1.0 / n_steps
        
        for i in range(n_steps):
            t = torch.full((n_samples,), i * dt, device=device)
            with torch.no_grad():
                v = self.forward(x, t)
                x = x + v * dt
        return x
    
    def bits_per_dim(self, x, n_steps=100, hutchinson=True, n_hutchinson_samples=1):
        """Compute bits per dimension of samples"""
        log_prob = self.log_likelihood(x, n_steps, hutchinson, n_hutchinson_samples)
        bpd = -log_prob / (self.dim * torch.log(torch.tensor(2.0)))
        return bpd

    def log_likelihood(self, x, n_steps=20, hutchinson=True, n_hutchinson_samples=1):
        """
        Compute log likelihood of samples.
        
        Args:
            x: Input samples (batch_size, dim)
            n_steps: Number of ODE integration steps
            hutchinson: If True, use Hutchinson's trace estimator (fast, stochastic).
                       If False, use exact trace computation (slow but exact).
            n_hutchinson_samples: Number of random vectors for Hutchinson estimator (default: 1)
        
        Returns:
            log_prob: Log probabilities (batch_size,)
        """
        if hutchinson:
            return self._log_likelihood_hutchinson(x, n_steps, n_hutchinson_samples)
        else:
            return self._log_likelihood_exact(x, n_steps)
    
    def _log_likelihood_exact(self, x, n_steps=100):
        """Exact log likelihood computation (slow for high dimensions)"""
        device = x.device
        dt = 1.0 / n_steps
        logp = torch.zeros(x.shape[0], device=device) 
        
        for i in range(n_steps, 0, -1):
            t = torch.full((x.shape[0],), i * dt, device=device)
            x.requires_grad_(True)
            v = self.forward(x, t)
            
            # Compute divergence: trace of the Jacobian
            div = sum(torch.autograd.grad(v[:, j].sum(), x, create_graph=True)[0][:, j] 
                    for j in range(self.dim))
            
            with torch.no_grad():
                x = x - v * dt  # Move backward in time
                logp -= div * dt # Accumulate change in density
                
        # Add log-density of the base distribution (Standard Normal)
        logp += torch.distributions.Normal(0, 1).log_prob(x).sum(-1)
        return logp
    
    def _log_likelihood_hutchinson(self, x, n_steps=20, n_samples=1):
        """Fast log likelihood using Hutchinson's trace estimator"""
        device = x.device
        batch_size = x.shape[0]
        dt = 1.0 / n_steps
        logp = torch.zeros(batch_size, device=device)
        
        for i in range(n_steps, 0, -1):
            t = torch.full((batch_size,), i * dt, device=device)
            x.requires_grad_(True)
            v = self.forward(x, t)
            
            # Hutchinson's trace estimator: E[eps^T * J * eps] = trace(J)
            # Average over multiple random vectors for better estimate
            div_estimate = torch.zeros(batch_size, device=device)
            
            for _ in range(n_samples):
                # Sample random Rademacher vector (±1 with equal probability)
                eps = torch.randn_like(x)
                
                # Compute Jacobian-vector product: J * eps
                # This is equivalent to the gradient of (v^T * eps) w.r.t. x
                eps_v = (v * eps).sum()
                vjp = torch.autograd.grad(eps_v, x, create_graph=True)[0]
                
                # Estimate trace: eps^T * J * eps = sum(vjp * eps)
                div_estimate += (vjp * eps).sum(dim=1)
            
            # Average over samples
            div = div_estimate / n_samples
            
            with torch.no_grad():
                x = x - v * dt  # Move backward in time
                logp -= div * dt  # Accumulate change in density
        
        # Add log-density of the base distribution (Standard Normal)
        logp += torch.distributions.Normal(0, 1).log_prob(x).sum(-1)
        return logp
    
def compute_validation_loss(model, vectors, batch_size, device):
    """Compute validation loss"""
    model.eval()
    vectors = torch.tensor(vectors, dtype=torch.float32, device=device)
    n_samples = len(vectors)
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            x1 = vectors[i:min(i + batch_size, n_samples)]
            
            # Sample random time and noise
            t = torch.rand(len(x1), device=device)
            x0 = torch.randn_like(x1)
            
            # Flow matching objective: predict normalized (x1 - x0)
            x_t = (1 - t.view(-1, 1)) * x0 + t.view(-1, 1) * x1
            target = torch.nn.functional.normalize(x1 - x0, p=2, dim=1, eps=1e-8)
            
            pred = model(x_t, t)
            loss = ((pred - target) ** 2).mean()
            
            total_loss += loss.item()
            n_batches += 1
    
    model.train()
    return total_loss / n_batches


def train_flow_matching(
    train_vectors,
    valid_vectors=None,
    test_vectors=None,
    epochs=10_000,
    batch_size=1024,
    lr=1e-5,
    device='mps',
    log_wandb=True,
    val_freq=10,
    save_freq=1_000,
    save_path=f'{RESULTS_DIR}/flow_model',
    special_epochs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    seed=DEFAULT_SEED,
    use_lr_scheduler=True,
    scheduler_patience=20,
    scheduler_factor=0.5,
    scheduler_min_lr=1e-8,
):
    """Train flow matching model on entity embedding vectors."""
    # Create results directory if it doesn't exist
    if not osp.exists(save_path):
        os.makedirs(save_path)

    # Set seed for reproducibility
    set_seed(seed)
    
    model = FlowMatching(dim=train_vectors.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize learning rate scheduler
    scheduler = None
    if use_lr_scheduler and valid_vectors is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True,
            min_lr=scheduler_min_lr
        )
        print(f"Learning rate scheduler enabled: ReduceLROnPlateau(patience={scheduler_patience}, factor={scheduler_factor}, min_lr={scheduler_min_lr})")
    
    # Save initial random model (epoch 0)
    initial_checkpoint_path = f"{save_path}/flow_model_epoch_0.pt"
    torch.save(model.state_dict(), initial_checkpoint_path)
    print(f"Initial model saved to {initial_checkpoint_path}")

    # Initialize wandb
    if log_wandb:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=EXPERIMENT_NAME,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "dim": train_vectors.shape[1],
                "n_samples": len(train_vectors),
                "device": device,
                "use_lr_scheduler": use_lr_scheduler,
                "scheduler_patience": scheduler_patience if use_lr_scheduler else None,
                "scheduler_factor": scheduler_factor if use_lr_scheduler else None,
                "scheduler_min_lr": scheduler_min_lr if use_lr_scheduler else None
            }
        )
        wandb.watch(model, log="all", log_freq=10)
    
    train_vectors = torch.tensor(train_vectors, dtype=torch.float32, device=device)
    n_samples = len(train_vectors)
    
    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        
        # Shuffle data
        perm = torch.randperm(n_samples, device=device)
        
        for i in range(0, n_samples, batch_size):
            idx = perm[i:min(i + batch_size, n_samples)]
            x1 = train_vectors[idx]
            
            # Sample random time and noise
            t = torch.rand(len(idx), device=device)
            x0 = torch.randn_like(x1)
            
            # Flow matching objective: predict normalized (x1 - x0)
            x_t = (1 - t.view(-1, 1)) * x0 + t.view(-1, 1) * x1
            target = torch.nn.functional.normalize(x1 - x0, p=2, dim=1, eps=1e-8)
            
            pred = model(x_t, t)
            loss = ((pred - target) ** 2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        
        # Compute validation loss
        val_loss = None
        if valid_vectors is not None and (epoch + 1) % val_freq == 0:
            val_loss = compute_validation_loss(model, valid_vectors, batch_size, device)
            
            # Step the learning rate scheduler based on validation loss
            if scheduler is not None:
                scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            if val_loss is not None:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.6f}")
        
        # Log to wandb
        if log_wandb:
            # Get current learning rate from optimizer
            current_lr = optimizer.param_groups[0]['lr']
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "learning_rate": current_lr
            }
            if val_loss is not None:
                log_dict["val_loss"] = val_loss
            wandb.log(log_dict)
        
        # Save model periodically
        if (epoch + 1) % save_freq == 0 or (epoch + 1) in special_epochs:
            checkpoint_path = f"{save_path}/flow_model_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")
    
    # Compute final test loss
    if test_vectors is not None:
        test_loss = compute_validation_loss(model, test_vectors, batch_size, device)
        print(f"\nFinal Test Loss: {test_loss:.6f}")
        if log_wandb:
            wandb.log({"test_loss": test_loss})
    
    if log_wandb:
        wandb.finish()
    
    return model

def main():
    # Load embeddings
    entity_embeddings = load_entity_embeddings(dataset_path=DATASET_PATH)
    
    # Save or load splits
    splits = load_embedding_splits(dataset_path=DATASET_PATH)

    # Extract train/val/test embeddings
    train_entity_embeddings = entity_embeddings[splits['train']]
    val_entity_embeddings = entity_embeddings[splits['val']]
    test_entity_embeddings = entity_embeddings[splits['test']]

    # Normalize embeddings to unit length
    train_entity_embeddings = torch.nn.functional.normalize(train_entity_embeddings, p=2, dim=1)
    val_entity_embeddings = torch.nn.functional.normalize(val_entity_embeddings, p=2, dim=1)
    test_entity_embeddings = torch.nn.functional.normalize(test_entity_embeddings, p=2, dim=1)

    print(f"Train split size: {len(splits['train'])}, Val split size: {len(splits['val'])}, Test split size: {len(splits['test'])}")

    # Train flow matching model
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    flow_model = train_flow_matching(
        train_vectors=train_entity_embeddings.detach().cpu().numpy(),
        valid_vectors=val_entity_embeddings.detach().cpu().numpy(),
        test_vectors=test_entity_embeddings.detach().cpu().numpy(),
        device=device,
        val_freq=10,
        save_freq=1_000,
        save_path=f'{RESULTS_DIR}/flow_model_{DATASET}_2'
    )

    # Save final model
    model_save_path = f'{RESULTS_DIR}/flow_model/flow_model_final.pt'
    torch.save(flow_model.state_dict(), model_save_path)
    print(f"Final model saved to {model_save_path}")

if __name__ == "__main__":
    main()