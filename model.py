"""Convolutional VAE for grayscale chest X-rays, implemented in PyTorch."""
import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, image_size=64, latent_dim=64):
        super().__init__()
        if image_size not in (64, 128):
            raise ValueError("image_size must be 64 or 128")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.spatial = image_size // 16
        channels = [1, 32, 64, 128, 256]
        layers = []
        for a, b in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv2d(a, b, 4, 2, 1), nn.GroupNorm(8, b), nn.SiLU()]
        self.encoder = nn.Sequential(*layers, nn.Flatten())
        width = 256 * self.spatial**2
        self.mu = nn.Linear(width, latent_dim)
        self.logvar = nn.Linear(width, latent_dim)
        self.from_z = nn.Linear(latent_dim, width)
        layers = []
        for a, b in [(256, 128), (128, 64), (64, 32)]:
            layers += [nn.ConvTranspose2d(a, b, 4, 2, 1), nn.GroupNorm(8, b), nn.SiLU()]
        # Return logits: BCEWithLogitsLoss is more stable than sigmoid + BCE.
        self.decoder = nn.Sequential(*layers, nn.ConvTranspose2d(32, 1, 4, 2, 1))

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        h = self.from_z(z).view(-1, 256, self.spatial, self.spatial)
        return self.decoder(h)

    def forward(self, x, sample=True):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        return self.decode(z), mu, logvar

    def generate(self, n, z=None):
        if z is None:
            z = torch.randn(n, self.latent_dim, device=next(self.parameters()).device)
        return self.decode(z).sigmoid()


def vae_loss(logits, x, mu, logvar, beta=1.0):
    """Sum pixels/latent coordinates per image, then average over the batch."""
    reconstruction = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    reconstruction = reconstruction.flatten(1).sum(1).mean()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
    return reconstruction + beta * kl, reconstruction, kl


def select_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
