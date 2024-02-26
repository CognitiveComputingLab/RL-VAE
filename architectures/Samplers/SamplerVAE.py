import torch
from architectures.Samplers.Sampler import Sampler


class SamplerVAE(Sampler):
    def __init__(self, device, data_loader):
        super().__init__(device, data_loader)

    def next_batch_indices(self):
        """
        get indices of next batch of points
        """
        # get indices tensor
        indices = [x for x in range(self.__current_index, min(self.__current_index + self.__data_loader.batch_size,
                                                              len(self.__data_loader.dataset)))]
        indices_tensor = torch.tensor([indices])
        self._current_x_batch = indices_tensor[0]

        # keep track of index
        self.__current_index += self.__data_loader.batch_size

        # reset
        if self.__current_index >= len(self.__data_loader.dataset):
            self.__epoch_done = True

        return indices_tensor.squeeze(0)

    def next_complementary_indices(self, property_calculator):
        """
        get indices for the points that are used as complementary points for the previous batch
        these points are used to compute and compare the low / high dimensional properties
        :param property_calculator: PropertyCalculator object with the relevant high / low dimensional properties saved
        """
        return
