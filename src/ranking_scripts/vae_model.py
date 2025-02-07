import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, beta):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder (2-layer MLP)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)  # Additional layer
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)

        # Decoder (2-layer MLP)
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, 256)  # Additional layer
        self.decoder = nn.Linear(256, input_dim)
        
        # Ranking Layer
        self.ranking_layer = nn.Linear(128, 1)  # Use 128, not latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder
        x_emb = F.relu(self.fc1(x))
        x_emb = F.relu(self.fc2(x_emb))
        mu = self.mu(x_emb)
        logvar = self.logvar(x_emb)
        z = self.reparameterize(mu, logvar)

        # Decoder
        recon_x_emb = F.relu(self.fc3(z))
        recon_x_emb = F.relu(self.fc4(recon_x_emb)) 
        recon_x = self.decoder(recon_x_emb)
        # Ranking Score
        ranking_score = torch.sigmoid(self.ranking_layer(x_emb))
        return ranking_score, recon_x, mu, logvar

    def loss_function(self, ranking_score, recon_x, x, mu, logvar, y_true, beta):
        # Reconstruction Loss (MSE, comparing with original input x)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')  # changed to 'mean'
        
        # KL Divergence
        kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # changed to 'mean'

        # Ranking Loss (BCE)
        ranking_loss = F.binary_cross_entropy(ranking_score, y_true.unsqueeze(1).float(), reduction='mean') # changed to 'mean'

        total_loss = recon_loss + beta * kl_divergence + ranking_loss
        return total_loss