import abc
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.modules.module import T
from rl_embeddings.components import Component


class Explorer(nn.Module, Component, abc.ABC):
    def __init__(self, device):
        super().__init__()
        Component.__init__(self)
        self._device = device

    @abc.abstractmethod
    def forward(self, **kwargs):
        """
        get single point from neural network output
        """
        raise NotImplementedError


class ExplorerVAE(Explorer):
    def __init__(self, device):
        super().__init__(device)
        self._required_inputs = ["means", "log_vars"]

    def forward(self, **kwargs):
        """
        get single point from encoder output
        sample from the output distribution via the re-parameterization trick
        """
        # check required arguments
        self.check_required_input(**kwargs)

        # get distribution variables
        mu = kwargs["means"]
        log_var = kwargs["log_vars"]

        # no exploration
        if not self.training:
            return {"encoded_points": mu}

        # re-parameterization trick
        # compute the standard deviation
        std = torch.exp(log_var / 2)
        # compute the normal distribution with the same standard deviation
        eps = torch.randn_like(std)
        # generate a sample
        sample = mu + std * eps

        return {"encoded_points": sample}


class ExplorerKHeadVAE(Explorer):
    def __init__(self, device):
        super().__init__(device)
        self._required_inputs = ["epoch", "head_means", "head_log_vars", "head_weights"]

        # hyperparameters
        self.min_exploration = 0.01
        self.decay_rate = 0.99
        self.starting_exploration = 1

        # exploration tracking
        self.current_exploration = self.starting_exploration
        self.exploration_save = self.starting_exploration
        self.previous_epoch = 0

    def train(self: T, mode: bool = True) -> T:
        # eval calls
        if not mode:
            # no exploration during evaluation
            self.exploration_save = self.current_exploration
            self.current_exploration = self.min_exploration
        # training mode
        else:
            # activate exploration
            self.current_exploration = self.exploration_save

        # train for pytorch module
        return super().train(mode)

    def exploration_function(self, epoch):
        """
        constant exploration
        :param epoch: current training epoch
        """
        return

    def forward(self, **kwargs):
        """
        get point from encoder output
        choose specific head based on weighted probabilities
        then pass mean and variance through re-parameterization to produce single point
        :return:
        """
        # check required arguments
        self.check_required_input(**kwargs)

        # get info from input
        epoch = kwargs["epoch"]
        mu = kwargs["head_means"]
        log_var = kwargs["head_log_vars"]
        weight = kwargs["head_weights"]

        # init
        if epoch:
            self.exploration_function(epoch)

        # get max weight index for each point in batch
        gumbel_softmax_indices = f.gumbel_softmax(weight, tau=self.current_exploration, hard=True, dim=1)

        # get weights
        chosen_weights = torch.sum(gumbel_softmax_indices * weight, dim=1)

        # get chosen mu and log_var by multiplying with one hot choices
        gumbel_softmax_indices = gumbel_softmax_indices.view(-1, gumbel_softmax_indices.size(1), 1)
        chosen_mu = torch.sum(gumbel_softmax_indices * mu, dim=1)
        chosen_log_var = torch.sum(gumbel_softmax_indices * log_var, dim=1)

        # re-parameterize
        # compute the standard deviation
        std = torch.exp(chosen_log_var / 2)
        # compute the normal distribution with the same standard deviation
        eps = torch.randn_like(std)
        # generate a sample
        sample = chosen_mu + std * eps

        return {"encoded_points": sample, "chosen_weights": chosen_weights, "chosen_means": chosen_mu,
                "chosen_log_vars": chosen_log_var}


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
            self.min_exploration = max(self.min_exploration, self.min_exploration * self.decay_rate)
            self.previous_epoch = epoch


class ExplorerVariance(Explorer):
    def __init__(self, device):
        super().__init__(device)
        self._required_inputs = ["encoded_points", "epoch"]

        # hyperparameters
        self.starting_exploration = 1

        # keep track of exploration
        self.current_exploration = self.starting_exploration
        self.exploration_save = self.current_exploration

    def train(self: T, mode: bool = True) -> T:
        # eval calls
        if not mode:
            # no exploration during evaluation
            self.exploration_save = self.current_exploration
            self.current_exploration = 0
        # training mode
        else:
            # activate exploration
            self.current_exploration = self.exploration_save

        # train for pytorch module
        return super().train(mode)

    def exploration_function(self, epoch):
        """
        manual control of variance over time
        constant exploration function
        """
        return

    def forward(self, **kwargs):
        """
        get single point from encoder output
        manually control log_var of output distribution and get mu from encoder
        sample from the distribution via the re-parameterization trick
        """
        # check required arguments
        self.check_required_input(**kwargs)

        # get info from input
        mu = kwargs["encoded_points"]
        epoch = kwargs["epoch"]

        # get distribution variables
        log_var = torch.Tensor([[self.current_exploration] * mu.shape[1] for _ in range(mu.shape[0])]).to(self._device)

        # no exploration
        if not self.training:
            return {"encoded_points": mu}

        # change exploration
        self.exploration_function(epoch)

        # re-parameterization trick
        # compute the standard deviation
        std = torch.exp(log_var / 2)
        # compute the normal distribution with the same standard deviation
        eps = torch.randn_like(std)
        # generate a sample
        sample = mu + std * eps

        return {"encoded_points": sample, "log_vars": log_var, "means": mu}


class ExplorerVarianceDecreasing(ExplorerVariance):
    def __init__(self, device):
        super().__init__(device)

        # hyperparameters
        self.exploration_decay = 0.99
        self.min_exploration = 0.01

        # keep track of epoch
        self._current_epoch = 0

    def exploration_function(self, epoch):
        """
        decrease exploration over time
        """
        # no exploration in eval mode
        if not self.training:
            return
        # only update exploration every epoch
        if self._current_epoch == epoch:
            return
        self._current_epoch = epoch

        # adjust exploration over time
        self.current_exploration = max(self.min_exploration, self.current_exploration * self.exploration_decay)