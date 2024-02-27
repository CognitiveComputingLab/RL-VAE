import torch
import torch.nn.functional as f
from architectures.RewardCalculators.RewardCalculator import RewardCalculator


class RewardCalculatorVAE(RewardCalculator):
    def __init__(self, device):
        super().__init__(device)

        # hyper parameters
        self.success_weight = 1
        self.kl_weight = 1

    def calculate_property_reward(self, high_dim_prop, low_dim_prop):
        raise ValueError("RewardCalculator Object not compatible. Tried to calculate property reward.")

    def calculate_reconstruction_reward(self, x_a, x_b, out):
        """
        calculate loss / reward based on reconstruction of points when passed through encoder and decoder
        includes KL divergence loss and success loss
        """
        # compute loss
        log_var, mu = out
        kl_divergence = 0.5 * torch.sum(-1 - log_var + mu.pow(2) + log_var.exp())
        loss = f.mse_loss(x_b, x_a, reduction='sum') * self.success_weight + kl_divergence * self.kl_weight

        # return computed loss
        return -loss
