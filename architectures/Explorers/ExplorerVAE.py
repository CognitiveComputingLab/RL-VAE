import torch
from architectures.Explorers.Explorer import Explorer


class ExplorerVAE(Explorer):
    def __init__(self, device):
        super().__init__(device)

    def get_point_from_output(self, out):
        """
        get single point from encoder output
        in this case, sample from the output distribution via the re-parameterization trick
        """
        mu, log_var = out
        # compute the standard deviation
        std = torch.exp(log_var / 2)
        # compute the normal distribution with the same standard deviation
        eps = torch.randn_like(std)
        # generate a sample
        sample = mu + std * eps
        return sample
