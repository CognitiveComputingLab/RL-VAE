import torch
from architectures.Samplers.Sampler import Sampler
from architectures.Samplers.SamplerVAE import SamplerVAE


class SamplerUMAP(Sampler, SamplerVAE):
    def __init__(self, device, data_loader):
        super().__init__(device, data_loader)

    def next_complementary_indices(self, property_calculator):
        """
        Get a single complementary index for each point in the current batch,
        randomly choosing between a positive and negative sample with equal probability.
        :param property_calculator: PropertyCalculator Object
        :return: A tensor of shape [batch_size, 1] where each entry is the index of the complementary point.
        """
        complementary_indices = []

        for sample_index in self._current_x_batch:
            # Get probability tensor from numpy
            probabilities = property_calculator.high_dim_property[sample_index]

            # Exclude the specified index
            adjusted_probabilities = probabilities.clone()
            adjusted_probabilities[sample_index] = 0

            # Decide randomly between positive and negative sample
            if torch.rand(1) < 0.5:
                # Positive sample: probabilities > 0
                positive_weights = adjusted_probabilities / adjusted_probabilities.sum()
                selected_index = torch.multinomial(positive_weights, 1, replacement=False)
            else:
                # Negative sample: flip probabilities to prefer lower values, excluding the current index
                adjusted_probabilities[sample_index] = 1  # Ensure current index is not selected
                negative_weights = 1 - adjusted_probabilities
                negative_weights /= negative_weights.sum()
                selected_index = torch.multinomial(negative_weights, 1, replacement=False)

            complementary_indices.append(selected_index)

        complementary_indices = torch.stack(complementary_indices).squeeze(1)
        return complementary_indices
