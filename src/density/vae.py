import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm


class VAE(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=256, latent_dim=128):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, embedding_dim)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        return self.fc4(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld


def train_vae(kbc_model, epochs=100, batch_size=256, lr=1e-3, device='cuda'):
    # Extract entity embeddings
    if hasattr(kbc_model, 'embeddings'):
        entity_emb = kbc_model.embeddings[0].weight.data
    else:
        entity_emb = kbc_model.lhs.weight.data
    
    embedding_dim = entity_emb.shape[1]
    vae = VAE(embedding_dim).to(device)
    optimizer = Adam(vae.parameters(), lr=lr)
    
    entity_emb = entity_emb.to(device)
    num_entities = entity_emb.shape[0]
    
    for epoch in range(epochs):
        vae.train()
        total_loss = 0
        
        indices = torch.randperm(num_entities)
        for i in tqdm(range(0, num_entities, batch_size), desc=f'Epoch {epoch+1}/{epochs}'):
            batch_indices = indices[i:i+batch_size]
            batch = entity_emb[batch_indices]
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(batch)
            loss = loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_entities
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
    
    return vae
