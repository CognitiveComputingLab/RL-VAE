import torch.nn as nn


class GeneralModel(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(GeneralModel, self).__init__()
        self.layers = nn.ModuleList()
        for h_dim in hidden_dims:
            self.layers.append(nn.Sequential(
                nn.Linear(input_dim, h_dim),
                nn.ReLU(),
            ))
            input_dim = h_dim

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EncoderAgent(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(EncoderAgent, self).__init__()
        self.gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])
        self.linearM = nn.Linear(4096, latent_dims)
        self.linearS = nn.Linear(4096, latent_dims)

    def forward(self, x):
        x = self.gm(x)
        mu = self.linearM(x)
        log_var = self.linearS(x)
        return mu, log_var


class EncoderAgentUMAP(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(EncoderAgentUMAP, self).__init__()
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
