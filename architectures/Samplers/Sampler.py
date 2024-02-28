from abc import ABC, abstractmethod


class Sampler(ABC):
    def __init__(self, device, data_loader):
        self._device = device
        self._data_loader = data_loader
        self._epoch_done = True

        # keep track
        self._current_index = 0
        self._current_x_batch = []

    @property
    def epoch_done(self):
        return self._epoch_done

    def reset_epoch(self):
        self._epoch_done = False
        self._current_index = 0
        self._current_x_batch = []

    def get_points_from_indices(self, indices):
        """
        retrieves actual points from given index tensor
        """
        points, colours = self._data_loader.dataset[indices]
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
