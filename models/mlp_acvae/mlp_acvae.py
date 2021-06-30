import torch
import torch.nn as nn
import numpy as np

from models.real_nvp.real_nvp import RealNVP
from models.resnet.resnet import ResNet


class MLP_ACVAE(nn.Module):
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
    def __init__(self, shape, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, conv_encoder=False):
        super().__init__()
        # Register data_constraint to pre-process images, not learnable
        self.register_buffer('data_constraint', torch.tensor([0.9], dtype=torch.float32))
        self.shape = shape
        self.latent_dim = np.product(shape)
        if conv_encoder:
            encoder_layers = [
                    ResNet(in_channels=in_channels,
                           mid_channels=mid_channels,
                           out_channels=in_channels * 2,
                           num_blocks=num_blocks * 2,
                           kernel_size=3,
                           padding=1,
                           double_after_norm=False),
                    nn.Flatten()
                ]
        else:
            encoder_layers = [
                    nn.Flatten(),
                    nn.Linear(self.latent_dim, self.latent_dim),
                    nn.ReLU(),
                    nn.Linear(self.latent_dim, 2 * self.latent_dim),
                ]
        self.encoder = nn.Sequential(*encoder_layers)

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
        return self.decoder(z, reverse=True)[0]

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def forward(self, x, sample=False):
        if sample:
            return self.decode(x)
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar).reshape([-1] + list(self.shape))
        x_hat = self.decode(z)
        return x_hat, mean, logvar
