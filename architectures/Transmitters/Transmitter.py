from abc import ABC, abstractmethod


class Transmitter(ABC):
    def __init__(self, device):
        self._device = device

    @abstractmethod
    def transmit(self, x):
        """
        communication channel between encoder and decoder
        :param x: the information that is transmitted (communicated) between them
        :return: the information after it has been passed through the channel
        """
        pass