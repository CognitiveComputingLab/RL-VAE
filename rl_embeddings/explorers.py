import abc
import torch
import torch.nn as nn
from torch.nn.modules.module import T


class Explorer(nn.Module, abc.ABC):
    def __init__(self, device):
        super().__init__()
        self._device = device

    @abc.abstractmethod
    def forward(self, encoder_out, epoch):
        """
        get single point from neural network output
        :param encoder_out:  directly from neural network component (encoder)
        :param epoch: the current epoch as int, used for regulating exploration
        """
        raise NotImplementedError


class ExplorerIdentity(Explorer):
    def __init__(self, device):
        super().__init__(device)

    def forward(self, encoder_out, epoch):
        """
        identity function
        """
        return encoder_out


class ExplorerVAE(Explorer):
    def __init__(self, device):
        super().__init__(device)

    def forward(self, encoder_out, epoch):
        """
        get single point from encoder output
        sample from the output distribution via the re-parameterization trick
        """
        # get distribution variables
        mu, log_var = encoder_out

        # no exploration
        if not self.training:
            return mu

        # re-parameterization trick
        # compute the standard deviation
        std = torch.exp(log_var / 2)
        # compute the normal distribution with the same standard deviation
        eps = torch.randn_like(std)
        # generate a sample
        sample = mu + std * eps
        return sample


class ExplorerKHeadVAE(Explorer):
    def __init__(self, device):
        super().__init__(device)

        # hyperparameters
        self.min_epsilon = 0.1
        self.decay_rate = 0.9
        self.epsilon_start = 0.9

        # exploration tracking
        self.epsilon = self.epsilon_start
        self.epsilon_save = self.epsilon_start
        self.previous_epoch = 0

    def train(self: T, mode: bool = True) -> T:
        # activate exploration
        self.epsilon = self.epsilon_save

        # train for pytorch module
        return super().train(mode)

    def eval(self: T) -> T:
        # no exploration during evaluation
        self.epsilon_save = self.epsilon
        self.epsilon = 0

        # eval for pytorch module
        return super().eval()

    def exploration_function(self, epoch):
        """
        constant exploration
        :param epoch: current training epoch
        """
        return

    def forward(self, encoder_out, epoch):
        """
        get point from encoder output
        choose specific head based on weighted probabilities
        then pass mean and variance through re-parameterization to produce single point
        :return:
        """
        # init
        if epoch:
            self.exploration_function(epoch)
        mu, log_var, weight = encoder_out

        batch_size, num_choices = weight.shape

        # choose points in batch based on probability
        random_selection_mask = torch.rand(batch_size, device=self._device) < self.epsilon

        # get max weight index for each point in batch
        argmax_indices = torch.argmax(weight, dim=1)

        # Step 3: Generate random indices for random selection
        random_indices = torch.randint(0, num_choices, (batch_size,), device=self._device)
        chosen_indices = torch.where(random_selection_mask, random_indices, argmax_indices)

        # get chosen mu / log_var for each point in batch
        expanded_indices = chosen_indices.view(batch_size, 1, 1).expand(-1, -1, mu.shape[2])
        chosen_mu = torch.gather(mu, 1, expanded_indices).squeeze(1)
        chosen_log_var = torch.gather(log_var, 1, expanded_indices).squeeze(1)

        # re-parameterize
        # compute the standard deviation
        std = torch.exp(chosen_log_var / 2)
        # compute the normal distribution with the same standard deviation
        eps = torch.randn_like(std)
        # generate a sample
        sample = chosen_mu + std * eps

        return sample, chosen_indices, chosen_mu, chosen_log_var


class ExplorerKHeadVAEDecreasing(ExplorerKHeadVAE):
    def __init__(self, device):
        super().__init__(device)

    def exploration_function(self, epoch):
        """
        decay amount of exploration over time
        :param epoch: current training epoch
        """
        # no exploration on eval
        if not self.training:
            return

        # decrease exploration every new epoch
        if self.previous_epoch < epoch:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
            self.previous_epoch = epoch


class ExplorerVariance(Explorer):
    def __init__(self, device):
        super().__init__(device)

        # hyperparameters
        self.starting_exploration = 1

        # keep track of exploration
        self.current_exploration = self.starting_exploration
        self.exploration_save = self.current_exploration

    def eval(self: T) -> T:
        self.exploration_save = self.current_exploration
        self.current_exploration = 0

        return super().eval()

    def exploration_function(self, epoch):
        """
        manual control of variance over time
        constant exploration function
        """
        return

    def forward(self, encoder_out, epoch):
        """
        get single point from encoder output
        manually control log_var of output distribution and get mu from encoder
        sample from the distribution via the re-parameterization trick
        """
        # get distribution variables
        mu = encoder_out
        log_var = torch.Tensor([[self.current_exploration] * mu.shape[1] for _ in range(mu.shape[0])]).to(self._device)

        # no exploration
        if not self.training:
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
        # no exploration in eval mode
        if not self.training:
            return

        # adjust exploration over time
        self.current_exploration = max(self.min_exploration, self.current_exploration * self.exploration_decay)
