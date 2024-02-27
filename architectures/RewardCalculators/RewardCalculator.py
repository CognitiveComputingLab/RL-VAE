from abc import ABC, abstractmethod


class RewardCalculator(ABC):
    def __init__(self, device):
        self._device = device

    @abstractmethod
    def calculate_property_reward(self, high_dim_prop, low_dim_prop):
        """
        calculate reward by comparing similarity of high dim and low dim properties
        :param high_dim_prop: batch of high dim properties between pairs of points
        :param low_dim_prop: batch of low dim properties between the same pairs of points as high dim
        :return: pytorch loss with grad
        """
        pass

    @abstractmethod
    def calculate_reconstruction_reward(self, x_a, x_b, out):
        """
        calculate the reward for reconstructing the point after full embedding process
        :param x_a: the original datapoint batch
        :param x_b: the reconstructed datapoint batch
        :param out: the output of the encoder network
        :return: pytorch loss with grad
        """
        pass
