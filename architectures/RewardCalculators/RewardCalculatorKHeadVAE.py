import torch
import torch.nn.functional as f
from architectures.RewardCalculators.RewardCalculator import RewardCalculator


class RewardCalculatorKHeadVAE(RewardCalculator):
    def __init__(self, device):
        super().__init__(device)

        # hyperparameters
        self.success_weight = 1

    def calculate_property_reward(self, high_dim_prop, low_dim_prop):
        raise ValueError("RewardCalculator Object not compatible. Tried to calculate property reward.")

    def calculate_reconstruction_reward(self, x_a, x_b, out, explorer):
        """
        calculate loss / reward based on reconstruction of points when passed through encoder and decoder
        includes KL divergence loss and success loss
        """
        mu, log_var, weight = out

        # tensor for multiplying reward with probability
        mean_weights = weight.gather(1, explorer.chosen_indices.unsqueeze(1))

        variance = torch.exp(explorer.chosen_log_var)
        surprise = variance + torch.square(explorer.chosen_mu)
        success = f.mse_loss(x_a, x_b)
        loss = torch.sum((surprise + (success * self.success_weight)) * mean_weights)
        return -loss
