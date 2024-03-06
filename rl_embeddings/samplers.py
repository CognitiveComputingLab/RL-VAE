import torch
import torch.nn as nn
import abc


class Sampler(nn.Module, abc.ABC):
    def __init__(self, device, data_loader):
        super().__init__()

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
        all points are automatically transferred to the device
        """
        points, colours = self._data_loader.dataset[indices]
        points = points.squeeze(0).to(self._device)
        colours = colours.squeeze(0).to(self._device)
        return points, colours

    @abc.abstractmethod
    def forward(self, high_dim_properties):
        """
        sample all points to train the embedding model
        includes regular and complementary points (if needed)
        :param high_dim_properties: high dimensional property matrix as calculated by PropertyCalculator object
        """
        raise NotImplementedError


class SamplerVAE(Sampler):
    def __init__(self, device, data_loader):
        super().__init__(device, data_loader)

    def get_batch_indices(self):
        """
        get indices of next batch of points by naively looping through the dataset
        :return: pytorch tensor of shape [x] with x amount of indices, the indices as a tensor of points
        """
        # get indices tensor
        indices = [x for x in range(self._current_index, min(self._current_index + self._data_loader.batch_size,
                                                             len(self._data_loader.dataset)))]
        indices_tensor = torch.tensor([indices])
        self._current_x_batch = indices_tensor[0]

        # keep track of index
        self._current_index += self._data_loader.batch_size

        # reset
        if self._current_index >= len(self._data_loader.dataset):
            self._epoch_done = True

        # remove first dimension from tensor
        return indices_tensor.squeeze(0)

    def forward(self, high_dim_properties):
        """
        get next batch of points by naively looping through the dataset
        gets the indices first but only returns the actual points
        :return: tuple of two pytorch tensors
            - point coordinates, shape: [batch_size, high_dim]
            - colours of points, shape: [batch_size, 3]
        """
        # get indices of batch
        indices_tensor = self.get_batch_indices()

        # get actual points
        points = self.get_points_from_indices(indices_tensor)

        return points


class SamplerTSNE(SamplerVAE):
    def __init__(self, device, data_loader):
        super().__init__(device, data_loader)

    def forward(self, high_dim_properties):
        # get indices of batch
        indices_tensor = self.get_batch_indices()

        # get actual points
        points = self.get_points_from_indices(indices_tensor)

        return points, indices_tensor


class SamplerUMAP(SamplerVAE):
    def __init__(self, device, data_loader):
        super().__init__(device, data_loader)

    def forward(self, high_dim_properties):
        """
        get indices of next batch of points by naively looping through the dataset
        additionally get complementary points
        :return: 4 tensors
            - p1: tuple of two pytorch tensors (coordinates, colours) for normal points
            - p2: tuple of two pytorch tensors (coordinates, colours) for complementary points
            - ind1: normal point indices tensor, shape: [batch_size]
            - ind2: complementary point indices tensor, shape: [batch_size]
        """
        # normal point indices
        ind1 = self.get_batch_indices()

        # normal points
        p1 = self.get_points_from_indices(ind1)

        # complementary point indices
        ind2 = self.next_complementary_indices(high_dim_properties)

        # complementary points
        p2 = self.get_points_from_indices(ind2)

        return p1, p2, ind1, ind2

    def next_complementary_indices(self, high_dim_properties):
        """
        Get a single complementary index for each point in the current batch,
        randomly choosing between a positive and negative sample with equal probability.
        :param high_dim_properties: high dimensional property matrix calculated by PropertyCalculator object
        :return: A tensor of shape [batch_size, 1] where each entry is the index of the complementary point.
        """
        complementary_indices = []

        for sample_index in self._current_x_batch:
            # get probability tensor from numpy
            probabilities = high_dim_properties[sample_index]

            # exclude the specified index
            adjusted_probabilities = probabilities.clone()
            adjusted_probabilities[sample_index] = 0

            # decide randomly between positive and negative sample
            if torch.rand(1) < 0.5:
                # positive sample: probabilities > 0
                positive_weights = adjusted_probabilities / adjusted_probabilities.sum()
                selected_index = torch.multinomial(positive_weights, 1, replacement=False)
            else:
                # negative sample: flip probabilities to prefer lower values, excluding the current index
                adjusted_probabilities[sample_index] = 1
                negative_weights = 1 - adjusted_probabilities
                negative_weights /= negative_weights.sum()
                selected_index = torch.multinomial(negative_weights, 1, replacement=False)

            complementary_indices.append(selected_index)

        complementary_indices = torch.stack(complementary_indices).squeeze(1)
        return complementary_indices
