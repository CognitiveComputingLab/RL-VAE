import torch.nn as nn
from architectures.Encoders.Encoder import GeneralModel


class EncoderKHeadVAE(nn.Module):
    def __init__(self, input_dim, latent_dims, k):
        super(EncoderKHeadVAE, self).__init__()
        self.k = k
        self.gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])
        self.linearM = nn.Linear(4096, k * latent_dims)
        self.linearS = nn.Linear(4096, k * latent_dims)
        self.weight_gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])
        self.linear_weight = nn.Linear(4096, k)

    def forward(self, x):
        x2 = self.gm(x)

        # output the means
        mu = self.linearM(x2)
        batch_size = mu.size(0)
        mu = mu.view(batch_size, self.k, -1)

        # output the log-vars
        logvar = self.linearS(x2)
        batch_size = logvar.size(0)
        logvar = logvar.view(batch_size, self.k, -1)

        weights = self.weight_gm(x)
        weights = self.linear_weight(weights)
        weights = nn.functional.softmax(weights, dim=1)

        return mu, logvar, weights
