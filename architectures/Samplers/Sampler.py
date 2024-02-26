import torch


class Sampler:
    def __init__(self, device):
        self.device = device
        self.__data_loader = None
        self.__complementary_available = False
        self.__epoch_done = True

        # keep track
        self.__current_index = 0

    @property
    def complementary_available(self):
        return self.__complementary_available

    @property
    def epoch_done(self):
        return self.__epoch_done

    def reset_epoch(self):
        self.__epoch_done = False

    @property
    def data_loader(self):
        return self.__data_loader

    @data_loader.setter
    def data_loader(self, new_loader):
        self.__data_loader = new_loader

    def next_batch_indices(self):
        """
        get next batch of points
        however, return only the indices in a pytorch tensor
        """
        # get indices tensor
        indices = [x for x in range(self.__current_index, min(self.__current_index + self.__data_loader.batch_size,
                                                              len(self.__data_loader.dataset)))]
        indices_tensor = torch.tensor([indices])

        # keep track of index
        self.__current_index += self.__data_loader.batch_size

        # reset
        if self.__current_index >= len(self.__data_loader.dataset):
            self.__current_index = 0
            self.__epoch_done = True

        return indices_tensor

    def get_points_from_indices(self, indices):
        """
        retrieves actual points from given index tensor
        """
        points, colours = self.__data_loader.dataset[indices]
        points = points.squeeze(0)
        colours = colours.squeeze(0)
        return points, colours
