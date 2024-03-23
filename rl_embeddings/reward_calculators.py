import abc
import torch
import torch.nn as nn
import torch.nn.functional as f
from rl_embeddings.components import Component


class RewardCalculator(nn.Module, Component, abc.ABC):
    def __init__(self, device):
        super().__init__()
        Component.__init__(self)
        self._device = device

        # general trivial reward
        self._trivial_reward = torch.tensor(0., requires_grad=True).to(self._device)

    @abc.abstractmethod
    def forward(self, **kwargs):
        """
        compute reward for encoder and decoder
        """
        raise NotImplementedError


class RewardCalculatorVAE(RewardCalculator):
    def __init__(self, device):
        super().__init__(device)
        self._required_inputs = ["means", "log_vars", "points", "decoded_points"]

        # hyperparameters
        self.success_weight = 1
        self.kl_weight = 1

    def forward(self, **kwargs):
        """
        compute reward for encoder and decoder
        a VAE only trains the encoder and decoder jointly
        therefore only the total_reward is non-trivial
        """
        # check required arguments
        self.check_required_input(**kwargs)

        # get information from different steps of embedding process
        mu = kwargs["means"]
        log_var = kwargs["log_vars"]
        x_a, _ = kwargs["points"]
        x_b = kwargs["decoded_points"]

        # KL term with prior as gaussian
        kl_divergence = 0.5 * torch.sum(-1 - log_var + mu.pow(2) + log_var.exp())

        # reconstruction term
        total_loss = f.mse_loss(x_b, x_a, reduction='sum') * self.success_weight + kl_divergence * self.kl_weight
        total_reward = (-1) * total_loss

        return {"total_reward": total_reward}


class RewardCalculatorKHeadVAE(RewardCalculator):
    def __init__(self, device):
        super().__init__(device)
        self._required_inputs = ["head_weights", "points", "decoded_points", "chosen_indices", "chosen_means",
                                 "chosen_log_vars"]

        # hyperparameters
        self.success_weight = 1

    def forward(self, **kwargs):
        """
        encoder and decoder are trained jointly
        reward also considers the weight of the chosen head
        """
        # check required arguments
        self.check_required_input(**kwargs)

        # get information from different steps of embedding process
        weight = kwargs["head_weights"]
        x_a, _ = kwargs["points"]
        x_b = kwargs["decoded_points"]
        chosen_indices = kwargs["chosen_indices"]
        chosen_mu = kwargs["chosen_means"]
        chosen_log_var = kwargs["chosen_log_vars"]

        # tensor for multiplying reward with probability
        mean_weights = weight.gather(1, chosen_indices.unsqueeze(1))

        # similar to VAE loss, without KL term
        variance = torch.exp(chosen_log_var)
        surprise = variance + torch.square(chosen_mu)
        success = f.mse_loss(x_a, x_b)
        total_loss = torch.sum((surprise + (success * self.success_weight)) * mean_weights)
        total_reward = (-1) * total_loss

        return {"total_reward": total_reward}


class RewardCalculatorUMAP(RewardCalculator):
    def __init__(self, device):
        super().__init__(device)
        self._required_inputs = ["low_dim_property", "high_dim_property"]

    def forward(self, **kwargs):
        """
        encoder is trained on high- / low-dim property differences
        decoder is trained on reconstruction
        """
        # check required arguments
        self.check_required_input(**kwargs)

        # get information from different steps of embedding process
        low_dim_prop = kwargs["low_dim_property"]
        high_dim_prop = kwargs["high_dim_property"]

        # encoder reward based on properties
        encoder_loss = f.binary_cross_entropy(low_dim_prop, high_dim_prop)
        encoder_reward = (-1) * encoder_loss

        return {"encoder_reward": encoder_reward}


class RewardCalculatorTSNE(RewardCalculator):
    def __init__(self, device):
        super().__init__(device)
        self._required_inputs = ["low_dim_property", "high_dim_property"]

    def forward(self, **kwargs):
        """
        the reward for TSNE is based on the KL-divergence between high and low dim distributions
        these distributions are given in the form of matrices (tensors)
        both matrices need to sum up to 1 as they represent probability distributions
        """
        # check required arguments
        self.check_required_input(**kwargs)

        # get information from different steps of embedding process
        q = kwargs["low_dim_property"]
        p = kwargs["high_dim_property"]

        # prevent division by 0
        p += 1e-8
        q += 1e-8

        # compute KL divergence loss as in T-SNE paper
        kl_divergence = p * (torch.log(p) - torch.log(q))
        kl_divergence = kl_divergence.sum()
        kl_reward = (-1) * kl_divergence

        return {"encoder_reward": kl_reward}
