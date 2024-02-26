import torch
from architectures.Samplers.Sampler import Sampler


class SamplerUMAP(Sampler):
    def __init__(self, device):
        super().__init__(device)

        # hyperparameters
        self.n_pos_samples = 10
        self.n_neg_samples = 10

    def next_complementary_indices(self, property_calculator):
        """
        get indices for the points that are used as complementary points for the previous batch
        these points are used to compute and compare the low / high dimensional properties
        :return: tensor of shape [batch_size, pos_samples + neg_samples] each point in the current batch gets a number
        of positive and negative sample indices
        """
        complementary_indices = []

        for sample_index in self._current_x_batch:
            # get probability tensor from numpy
            probabilities = torch.from_numpy(property_calculator.symmetric_probabilities[sample_index]).float()

            # exclude the specified index
            adjusted_probabilities = probabilities.clone()
            adjusted_probabilities[sample_index] = 0

            n_positive = torch.sum(adjusted_probabilities > 0).item()
            n_negative = torch.sum(adjusted_probabilities < 1).item() - 1

            # adjust n_samples based on available samples
            n_samples_positive = min(self.n_pos_samples, n_positive)
            n_samples_negative = min(self.n_neg_samples, n_negative)

            # sample positive (move towards point)
            positive_weights = adjusted_probabilities / adjusted_probabilities.sum()
            positive_indices = torch.multinomial(positive_weights, n_samples_positive, replacement=False)

            # sample negative (move away from point)
            adjusted_probabilities[sample_index] = 1
            negative_weights = 1 - adjusted_probabilities
            negative_weights /= negative_weights.sum()
            negative_indices = torch.multinomial(negative_weights, n_samples_negative, replacement=False)

            # combine negative and positive samples
            current_complementary_indices = torch.concat([positive_indices, negative_indices])
            complementary_indices.append(current_complementary_indices)

        complementary_indices = torch.stack(complementary_indices, dim=0)
        return complementary_indices
