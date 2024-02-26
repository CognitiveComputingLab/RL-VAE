from architectures.PropertyCalculators.PropertyCalculatorUMAP import PropertyCalculatorUMAP
from architectures.Samplers.Sampler import Sampler


class EmbeddingFramework:
    def __init__(self, device, input_dim, output_dim):
        # init general
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim

        # components
        self.property_calculator = PropertyCalculatorUMAP(device)
        self.sampler = Sampler(device)

    def train(self, training_data_loader, epochs=100):
        """
        train the encoder to produce an embedding for the given dataset
        :param training_data_loader: pytorch dataloader
        :param epochs: number of epochs to train for as int
        """
        # pass the data loader to the sampler
        self.sampler.data_loader = training_data_loader

        # distance calculation for high dimensional data
        self.property_calculator.calculate_high_dim_property(training_data_loader)

        for epoch in range(epochs):
            # tell the sampler that a new epoch is starting
            self.sampler.reset_epoch()

            while not self.sampler.epoch_done:
                # get batch of points
                ind = self.sampler.next_batch_indices()
                x, _ = self.sampler.get_points_from_indices(ind)
                x = x.to(self.device)


