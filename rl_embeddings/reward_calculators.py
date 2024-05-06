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
        self.success_weight = 10
        self.kl_weight = 1

    def forward(self, **kwargs):
        """
        compute reward for encoder and decoder
        a VAE only trains the encoder and decoder jointly
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

        # add reconstruction term
        total_loss = f.mse_loss(x_b, x_a, reduction='sum') * self.success_weight + kl_divergence * self.kl_weight
        total_reward = (-1) * total_loss

        return {"total_reward": total_reward}


class RewardCalculatorMSE(RewardCalculator):
    def __init__(self, device):
        super().__init__(device)
        self._required_inputs = ["points", "decoded_points"]

        # hyperparameters
        self.success_weight = 1

    def forward(self, **kwargs):
        """
        MSE difference between encoder input and decoder output
        """
        # check required arguments
        self.check_required_input(**kwargs)

        # get information from different steps of embedding process
        x_a, _ = kwargs["points"]
        x_b = kwargs["decoded_points"]

        # add reconstruction term
        total_loss = f.mse_loss(x_b, x_a, reduction='sum') * self.success_weight
        total_reward = (-1) * total_loss

        return {"total_reward": total_reward}


class RewardCalculatorKHeadVAE(RewardCalculator):
    def __init__(self, device):
        super().__init__(device)
        self._required_inputs = ["points", "decoded_points", "chosen_weights", "chosen_means", "chosen_log_vars"]

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
        x_a, _ = kwargs["points"]
        x_b = kwargs["decoded_points"]
        chosen_weights = kwargs["chosen_weights"]
        chosen_mu = kwargs["chosen_means"]
        chosen_log_var = kwargs["chosen_log_vars"]

        chosen_weights = chosen_weights.view(chosen_weights.shape[0], 1)
        chosen_weights = f.softmax(chosen_weights, dim=0)

        # similar to VAE loss, without KL term and weighted by weight output
        variance = torch.exp(chosen_log_var)
        surprise = variance + torch.square(chosen_mu)
        success = f.mse_loss(x_a, x_b)
        total_loss = torch.sum((surprise + (success * self.success_weight)) * chosen_weights)
        total_reward = (-1) * total_loss

        return {"total_reward": total_reward}


class RewardCalculatorVAE_UMAP(RewardCalculator):
    def __init__(self, device):
        super().__init__(device)
        self._required_inputs = ["means", "log_vars", "points", "low_dim_similarity", "high_dim_similarity"]
        self.umap_weight = 100
        self.kl_weight = 0.01

    def forward(self, **kwargs):
        """
        combine VAE and UMAP reward functions
        """
        # check required arguments
        self.check_required_input(**kwargs)

        # get information from different steps of embedding process
        mu = kwargs["means"]
        log_var = kwargs["log_vars"]
        x_a, _ = kwargs["points"]
        low_dim_prop = kwargs["low_dim_similarity"]
        high_dim_prop = kwargs["high_dim_similarity"]

        # KL term with prior as gaussian
        kl_divergence = 0.5 * torch.sum(-1 - log_var + mu.pow(2) + log_var.exp())

        # encoder reward based on similarities
        encoder_loss = f.binary_cross_entropy(low_dim_prop, high_dim_prop)

        # add reconstruction term
        total_loss = self.umap_weight * encoder_loss + self.kl_weight * kl_divergence
        total_reward = (-1) * total_loss

        return {"encoder_reward": total_reward}


class RewardCalculatorUMAP(RewardCalculator):
    def __init__(self, device):
        super().__init__(device)
        self._required_inputs = ["low_dim_similarity", "high_dim_similarity"]

    def forward(self, **kwargs):
        """
        encoder is trained on high- / low-dim similarity differences
        decoder is trained on reconstruction
        """
        # check required arguments
        self.check_required_input(**kwargs)

        # get information from different steps of embedding process
        low_dim_prop = kwargs["low_dim_similarity"]
        high_dim_prop = kwargs["high_dim_similarity"]

        # encoder reward based on similarities
        encoder_loss = f.binary_cross_entropy(low_dim_prop, high_dim_prop)
        encoder_reward = (-1) * encoder_loss

        return {"encoder_reward": encoder_reward}


class RewardCalculatorTSNE(RewardCalculator):
    def __init__(self, device):
        super().__init__(device)
        self._required_inputs = ["low_dim_similarity", "high_dim_similarity"]

    def forward(self, **kwargs):
        """
        the reward for TSNE is based on the KL-divergence between high and low dim distributions
        these distributions are given in the form of matrices (tensors)
        both matrices need to sum up to 1 as they represent probability distributions
        """
        # check required arguments
        self.check_required_input(**kwargs)

        # get information from different steps of embedding process
        q = kwargs["low_dim_similarity"]
        p = kwargs["high_dim_similarity"]

        # prevent division by 0
        p += 1e-8
        q += 1e-8

        # compute KL divergence loss as in T-SNE paper
        kl_divergence = p * (torch.log(p) - torch.log(q))
        kl_divergence = kl_divergence.sum()
        kl_reward = (-1) * kl_divergence

        return {"encoder_reward": kl_reward}
