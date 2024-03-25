import torch.nn as nn
import abc
from rl_embeddings.components import Component


class Transmitter(nn.Module, Component, abc.ABC):
    def __init__(self, device):
        super().__init__()
        Component.__init__(self)
        self._device = device

    @abc.abstractmethod
    def forward(self, **kwargs):
        """
        communication channel between encoder and decoder
        :return: the information after it has been passed through the channel
        """
        raise NotImplementedError


class TransmitterIdentity(Transmitter):
    def __init__(self, device):
        super().__init__(device)

    def forward(self, **kwargs):
        """
        clear communication channel without loss
        implemented by identity function
        """
        # check required arguments
        self.check_required_input(**kwargs)

        return kwargs


class TransmitterCircle(Transmitter):
    def __init__(self, device):
        super().__init__(device)

        self._required_inputs = ["encoded_points"]

    def forward(self, **kwargs):
        """
        transmit to create a noisy circle on a circular latent space
        """
        # check required arguments
        self.check_required_input(**kwargs)

        # get input
        data = kwargs["encoded_points"]

        # wrap around a circle
        data_normalized = (data - data.min()) / (data.max() - data.min())
        data_circular = data_normalized % 1.0

        return {"transmitted_points": data_circular}

