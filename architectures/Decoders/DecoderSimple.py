import torch.nn as nn
from architectures.Encoders.Encoder import GeneralModel


class DecoderSimple(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(DecoderSimple, self).__init__()
        self.gm = GeneralModel(latent_dims, [512, 1024, 2024, 4096])
        self.linear = nn.Linear(4096, input_dim)

    def forward(self, z):
        z = self.gm(z)
        z = self.linear(z)
        return z
