import torch
from architectures.Explorers.Explorer import Explorer


class ExplorerVAE(Explorer):
    def __init__(self, device):
        super().__init__(device)

    @property
    def evaluation_active(self):
        return self._evaluation_active

    @evaluation_active.setter
    def evaluation_active(self, value):
        self._evaluation_active = value

    def get_point_from_output(self, out):
        """
        get single point from encoder output
        in this case, sample from the output distribution via the re-parameterization trick
        """
        # get distribution variables
        mu, log_var = out

        # no exploration
        if self.evaluation_active:
            return mu

        # re-parameterization trick
        # compute the standard deviation
        std = torch.exp(log_var / 2)
        # compute the normal distribution with the same standard deviation
        eps = torch.randn_like(std)
        # generate a sample
        sample = mu + std * eps
        return sample
