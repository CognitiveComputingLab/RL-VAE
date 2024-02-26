from abc import ABC, abstractmethod


class Sampler(ABC):
    def __init__(self, device, data_loader):
        self.device = device
        self.__data_loader = data_loader
        self.__epoch_done = True

        # keep track
        self.__current_index = 0
        self._current_x_batch = []

    @property
    def epoch_done(self):
        return self.__epoch_done

    def reset_epoch(self):
        self.__epoch_done = False
        self.__current_index = 0
        self._current_x_batch = []

    def get_points_from_indices(self, indices):
        """
        retrieves actual points from given index tensor
        """
        points, colours = self.__data_loader.dataset[indices]
        points = points.squeeze(0)
        colours = colours.squeeze(0)
        return points, colours

    @abstractmethod
    def next_batch_indices(self):
        """
        get next batch of points
        return only the indices in a pytorch tensor
        """
        pass

    @abstractmethod
    def next_complementary_indices(self, property_calculator):
        """
        get indices for the points that are used as complementary points for the previous batch
        these points are used to compute and compare the low / high dimensional properties
        :param property_calculator: PropertyCalculator object with the relevant high / low dimensional properties saved
        """
        pass
