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
