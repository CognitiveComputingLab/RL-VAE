import torch.nn as nn
import abc


class Transmitter(nn.Module, abc.ABC):
    def __init__(self, device):
        super().__init__()
        self._device = device

    @abc.abstractmethod
    def forward(self, x):
        """
        communication channel between encoder and decoder
        :param x: the information that is transmitted (communicated) between them
        :return: the information after it has been passed through the channel
        """
        raise NotImplementedError


class TransmitterIdentity(Transmitter):
    def __init__(self, device):
        super().__init__(device)

    def forward(self, x):
        """
        clear communication channel without loss
        implemented by identity function
        """
        return x
