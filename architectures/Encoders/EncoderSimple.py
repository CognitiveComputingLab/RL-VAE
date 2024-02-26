import torch.nn as nn
from architectures.Encoders.Encoder import GeneralModel


class EncoderSimple(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(EncoderSimple, self).__init__()
        # Assuming each point has the same dimension as input_dim
        self.gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])

        # Output layers for each point
        self.linear1 = nn.Linear(4096, latent_dims)

    def forward(self, x):
        # Concatenate the two input points
        x = self.gm(x)

        # Compute outputs for each point
        mu1 = self.linear1(x)
        return mu1
