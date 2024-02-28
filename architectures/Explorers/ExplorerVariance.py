import torch
from architectures.Explorers.Explorer import Explorer


class ExplorerVariance(Explorer):
    def __init__(self, device):
        super().__init__(device)

        # hyperparameters
        self.starting_exploration = 1

        # keep track of exploration
        self.current_exploration = self.starting_exploration

    @property
    def evaluation_active(self):
        return self._evaluation_active

    @evaluation_active.setter
    def evaluation_active(self, value):
        self._evaluation_active = value

    def exploration_function(self, epoch):
        """
        manual control of variance over time
        constant exploration function
        """
        return

    def get_point_from_output(self, out, epoch=None):
        """
        get single point from encoder output
        manually control log_var of output distribution and get mu from encoder
        sample from the distribution via the re-parameterization trick
        """
        # get distribution variables
        mu = out
        log_var = torch.Tensor([[self.current_exploration] * mu.shape[1] for _ in range(mu.shape[0])]).to(self._device)

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


class ExplorerVarianceDecreasing(ExplorerVariance):
    def __init__(self, device):
        super().__init__(device)

        # hyperparameters
        self.exploration_decay = 0.9
        self.min_exploration = 0.01

    def exploration_function(self, epoch):
        """
        decrease exploration over time
        """
        self.current_exploration = max(self.min_exploration, self.current_exploration * self.exploration_decay)
