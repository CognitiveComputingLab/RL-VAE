from abc import ABC, abstractmethod


class PropertyCalculator(ABC):
    def __init__(self, device, data_loader):
        self._device = device
        self._data_loader = data_loader

    @property
    @abstractmethod
    def high_dim_property(self):
        return

    @abstractmethod
    def symmetrize(self, prob):
        """
        symmetrize the high dimensional property
        """
        pass

    @abstractmethod
    def calculate_high_dim_property(self):
        """
        compute all properties required for comparing high dimensional points
        """
        pass

    @abstractmethod
    def get_low_dim_property(self, p1, p2):
        """
        calculate low dimensional property
        :param p1: first point as pytorch Tensor
        :param p2: second point as pytorch Tensor
        """
        pass


