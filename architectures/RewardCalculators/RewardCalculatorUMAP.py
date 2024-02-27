import torch.nn.functional as f
from architectures.RewardCalculators.RewardCalculator import RewardCalculator


class RewardCalculatorUMAP(RewardCalculator):
    def __init__(self, device):
        super().__init__(device)

    def calculate_property_reward(self, high_dim_prop, low_dim_prop):
        """
        calculate reward based on similarity between high and low dimensional probability
        """
        loss = f.binary_cross_entropy(low_dim_prop, high_dim_prop)
        return -loss

    def calculate_reconstruction_reward(self, x_a, x_b, out):
        """
        calculate rewards based on difference between original point and reconstructed point
        """
        loss = f.mse_loss(x_b, x_a, reduction='sum')
        return -loss
