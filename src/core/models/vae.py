import torch
import torch.nn as nn
import numpy as np

class VAE_model(nn.Module):

    def __init__(self, d, D, H, activFun):
        super(VAE_model, self).__init__()

        # Backbone for encoder
        self.enc = nn.Sequential(
            nn.Linear(D, H),
            activFun,
            nn.Linear(H, H),
            activFun
        )
        # Two heads to predict mean and log variance of latent distribution
        self.mu_enc_head    = nn.Linear(H, d)
        self.log_var_enc_head = nn.Linear(H, d)

        # Backbone for decoder
        self.dec = nn.Sequential(
            nn.Linear(d, H),
            activFun,
            nn.Linear(H, H),
            activFun
        )
        self.mu_dec_head      = nn.Linear(H, D)

    def encode(self, x):
        h = self.enc(x)
        return self.mu_enc_head(h), self.log_var_enc_head(h)

    def decode(self, z):
        h = self.dec(z)
        return self.mu_dec_head(h)

    @staticmethod
    def reparametrization_trick(mu, log_var):
        epsilon = torch.randn_like(mu)
        return mu + torch.exp(0.5 * log_var) * epsilon

    def forward(self, x):
        mu_z, log_var_z = self.encode(x)
        z_rep = self.reparametrization_trick(mu_z, log_var_z)
        mu_x = self.decode(z_rep)
        return mu_x, z_rep, mu_z, log_var_z


def VAE_loss(x, mu_x, mu_z, log_var_z, r=1.0):
    # Recon loss with fixed variance = 1 for simplicity
    recon_loss = 0.5 * ((x - mu_x)**2).sum(dim=1).mean()

    kl_loss = 0.5 * (
        mu_z**2 + log_var_z.exp() - log_var_z - 1
    ).sum(dim=1).mean()

    return recon_loss + r * kl_loss, recon_loss.item(), kl_loss.item()