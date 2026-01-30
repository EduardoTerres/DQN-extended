import os
import os.path as osp
from pathlib import Path
import sys

os.environ['PYTHONPATH'] = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, os.environ['PYTHONPATH'])

import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from density.dataset import load_entity_embeddings, load_embedding_splits

WANDB_PROJECT = "ML4Graphs"
WANDB_ENTITY = "scale-gmns"
EXPERIMENT_NAME = "vae-cqd"

DATASET = "FB15k-237"
REPO_ROOT = Path(__file__).resolve().parents[2]
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


class VAE(nn.Module):
    def __init__(self, dim, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

        # Proper weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Overrides for output and distribution heads
        nn.init.xavier_uniform_(self.fc_mu.weight)
        if self.fc_mu.bias is not None:
            nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_uniform_(self.fc_logvar.weight)
        if self.fc_logvar.bias is not None:
            nn.init.zeros_(self.fc_logvar.bias)
        last_linear = self.decoder[-1]
        nn.init.xavier_uniform_(last_linear.weight)
        if last_linear.bias is not None:
            nn.init.zeros_(last_linear.bias)
    
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to reconstruction"""
        reconstructed = self.decoder(z)
        # Normalize vectors rowwise
        reconstructed = reconstructed / reconstructed.norm(dim=1, keepdim=True)
        return reconstructed
    
    def forward(self, x):
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def sample(self, n_samples, device='cpu'):
        """Sample from the model"""
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=device)
            # Scale latent variance inversely with dimension for stability
            z = z / (self.latent_dim ** 0.5)
            samples = self.decode(z)
        return samples
    
    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        """
        VAE loss = Reconstruction loss + KL divergence
        beta: weight for KL term (beta-VAE)
        """
        # Reconstruction loss (MSE)
        # recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
        
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # KL divergence per-sample (sum over latent dims), then average over batch
        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = kl_per_sample.mean()

        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    def elbo(self, x):
        """Compute ELBO (negative loss)"""
        recon_x, mu, logvar = self.forward(x)
        loss, _, _ = self.loss_function(recon_x, x, mu, logvar, beta=1.0)
        return -loss


def compute_validation_loss(model, vectors, batch_size, device, beta=1.0):
    """Compute validation loss"""
    model.eval()
    vectors = torch.tensor(vectors, dtype=torch.float32, device=device)
    n_samples = len(vectors)
    total_loss = 0
    total_recon = 0
    total_kl = 0
    n_batches = 0
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            x = vectors[i:min(i + batch_size, n_samples)]
            recon_x, mu, logvar = model(x)
            loss, recon_loss, kl_loss = model.loss_function(recon_x, x, mu, logvar, beta)
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1
    
    model.train()
    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


def train_vae(
    train_vectors,
    valid_vectors=None,
    test_vectors=None,
    epochs=20_000,
    batch_size=1024,
    lr=1e-5,
    latent_dim=128,
    hidden_dim=256,
    beta=1.0,
    device='mps',
    log_wandb=True,
    val_freq=10,
    save_freq=1000,
    save_path=f'{RESULTS_DIR}/vae_model_{DATASET}',
    special_epochs=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    seed=DEFAULT_SEED,
):
    """Train VAE model on entity embedding vectors."""
    # Ensure the save directory (e.g., results/vae_model) exists
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    set_seed(seed)
    
    model = VAE(dim=train_vectors.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Save initial random model (epoch 0)
    initial_checkpoint_path = f"{save_path}/vae_model_epoch_0.pt"
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
                "latent_dim": latent_dim,
                "hidden_dim": hidden_dim,
                "beta": beta,
                "n_samples": len(train_vectors),
                "device": device
            }
        )
        wandb.watch(model, log="all", log_freq=10)
    
    train_vectors = torch.tensor(train_vectors, dtype=torch.float32, device=device)
    n_samples = len(train_vectors)
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        n_batches = 0
        
        # Shuffle data
        perm = torch.randperm(n_samples, device=device)
        
        for i in range(0, n_samples, batch_size):
            idx = perm[i:min(i + batch_size, n_samples)]
            x = train_vectors[idx]
            
            # Forward pass
            recon_x, mu, logvar = model(x)
            loss, recon_loss, kl_loss = model.loss_function(recon_x, x, mu, logvar, beta)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches
        
        # Compute validation loss
        val_loss = val_recon = val_kl = None
        if valid_vectors is not None and (epoch + 1) % val_freq == 0:
            val_loss, val_recon, val_kl = compute_validation_loss(model, valid_vectors, batch_size, device, beta)
        
        if (epoch + 1) % 10 == 0:
            if val_loss is not None:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.6f} (Recon: {avg_recon:.6f}, KL: {avg_kl:.6f}), Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.6f} (Recon: {avg_recon:.6f}, KL: {avg_kl:.6f})")
        
        # Log to wandb
        if log_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_recon": avg_recon,
                "train_kl": avg_kl,
                "learning_rate": lr
            }
            if val_loss is not None:
                log_dict["val_loss"] = val_loss
                log_dict["val_recon"] = val_recon
                log_dict["val_kl"] = val_kl
            wandb.log(log_dict)
        
        # Save model periodically
        if (epoch + 1) % save_freq == 0 or (epoch + 1) in special_epochs:
            checkpoint_path = f"{save_path}/vae_model_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")
    
    # Compute final test loss
    if test_vectors is not None:
        test_loss, test_recon, test_kl = compute_validation_loss(model, test_vectors, batch_size, device, beta)
        print(f"\nFinal Test Loss: {test_loss:.6f} (Recon: {test_recon:.6f}, KL: {test_kl:.6f})")
        if log_wandb:
            wandb.log({"test_loss": test_loss, "test_recon": test_recon, "test_kl": test_kl})
    
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

    # Train VAE model
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    vae_model = train_vae(
        train_vectors=train_entity_embeddings.detach().cpu().numpy(),
        valid_vectors=val_entity_embeddings.detach().cpu().numpy(),
        test_vectors=test_entity_embeddings.detach().cpu().numpy(),
        device=device,
        val_freq=10,
        save_freq=1_000,
        save_path=f'{RESULTS_DIR}/vae_model_{DATASET}'
    )

    # Save final model
    model_save_path = f'{RESULTS_DIR}/vae_model/vae_model_final.pt'
    torch.save(vae_model.state_dict(), model_save_path)
    print(f"Final model saved to {model_save_path}")


if __name__ == "__main__":
    main()
