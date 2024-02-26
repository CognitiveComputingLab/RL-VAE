from abc import ABC, abstractmethod


class Explorer(ABC):
    def __init__(self, device):
        self._device = device

    @abstractmethod
    def get_point_from_output(self, out):
        """
        get single point from neural network output
        :param out: output directly from neural network component (encoder)
        """
        pass
