import torch
import torch.nn as nn
import torch.nn.functional as F

from models.real_nvp.real_nvp import RealNVP
from models.resnet.resnet import ResNet
from util import squeeze_2x2


class ACVAEMLP(nn.Module):
    """RealNVP Model

    Based on the paper:
    "Density estimation using Real NVP"
    by Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio
    (https://arxiv.org/abs/1605.08803).

    Args:
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
        `Coupling` layers.
    """
    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, encoder_hidden_dim, latent_dim):
        super().__init__()
        # Register data_constraint to pre-process images, not learnable
        self.register_buffer('data_constraint', torch.tensor([0.9], dtype=torch.float32))
        self.latent_dim = latent_dim
        encoder_layers = [
                ResNet(in_channels=in_channels,
                       mid_channels=mid_channels,
                       out_channels=in_channels * 2,
                       num_blocks=num_blocks * 2,
                       kernel_size=3,
                       padding=1,
                       False),
                nn.Flatten()
            ]
        self.encoder = nn.Sequential(**encoder_layers)

        self.decoder = RealNVP(num_scales=num_scales,
                               in_channels=in_channels,
                               mid_channels=mid_channels,
                               num_blocks=num_blocks)


    def encode(self, x):
        out = self.encoder(x)
        mean = out[..., :self.latent_dim]
        logvar = out[..., self.latent_dim:]
        return mean, logvar

    def decode(self, z):
        return self.decoder(z, reverse=True)


    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

    def loss_fn(self, x_hat, x, mean, logvar):
        reconstruction_loss = F.mse_loss(x_hat, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim=1), dim=0)
        vae_loss = reconstruction_loss + kld_loss
        return vae_loss, kld_loss, reconstruction_loss
