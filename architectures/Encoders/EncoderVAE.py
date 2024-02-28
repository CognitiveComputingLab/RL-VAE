import torch.nn as nn
from architectures.Encoders.Encoder import GeneralModel


class EncoderVAE(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(EncoderVAE, self).__init__()
        self.gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])
        self.linearM = nn.Linear(4096, latent_dims)
        self.linearS = nn.Linear(4096, latent_dims)

    def forward(self, x):
        x = self.gm(x)
        mu = self.linearM(x)
        log_var = self.linearS(x)
        return mu, log_var
