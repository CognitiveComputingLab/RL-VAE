from abc import ABC, abstractmethod


class Explorer(ABC):
    def __init__(self, device):
        self._device = device
        self._evaluation_active = False

    @property
    @abstractmethod
    def evaluation_active(self):
        pass

    @evaluation_active.setter
    @abstractmethod
    def evaluation_active(self, value):
        """
        deactivate / activate any exploration
        """
        pass

    @abstractmethod
    def get_point_from_output(self, out, epoch):
        """
        get single point from neural network output
        :param out:  directly from neural network component (encoder)
        :param epoch: the current epoch as int, used for regulating exploration
        """
        pass
