
class VAE(nn.Module):
    def __init__(self, shape, latent_dim, hidden_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.input_size = np.product(shape)
        encoder_layers = [
                nn.Flatten(),
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 2 * self.latent_dim),
            ]
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = [
                nn.Linear(self.latent_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.input_size),
                nn.Sigmoid()
            ]
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        out = self.encoder(x)
        mean = out[..., :self.latent_dim]
        logvar = out[..., self.latent_dim:]
        return mean, logvar

    def decode(self, z):
        flat_output = self.decoder(z, reverse=True)[0]
        return flat_output.reshape([-1] + list(self.shape))

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


def VAELoss(x, x_hat, mean, logvar):
    reconstruction_loss = F.mse_loss(x_hat, x)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim=1), dim=0)
    vae_loss = reconstruction_loss + kld_loss
    return vae_loss, kld_loss, reconstruction_loss


def VAELossCE(x, x_hat, mean, logvar):
    eps = 1e-8
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim=1), dim=0)
    ce_loss = torch.sum(-x * torch.log(x_hat + eps) - (1 - x) * torch.log(1 - x_hat + eps))
    vae_loss = kld_loss + ce_loss
    return vae_loss, kld_loss, ce_loss
