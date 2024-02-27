import torch
from architectures.PropertyCalculators.PropertyCalculatorUMAP import PropertyCalculatorUMAP
from architectures.Samplers.SamplerUMAP import SamplerUMAP
from architectures.Explorers.ExplorerIdentity import ExplorerIdentity
from architectures.Encoders.EncoderSimple import EncoderSimple
from architectures.PropertyCalculators.PropertyCalculator import PropertyCalculator
from architectures.Samplers.Sampler import Sampler


class EmbeddingFramework:
    def __init__(self, device, input_dim, output_dim, data_loader):
        # init general
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim

        # components
        self.property_calculator = PropertyCalculatorUMAP(device, data_loader)
        self.sampler = SamplerUMAP(device, data_loader)
        self.encoder_agent = EncoderSimple(input_dim, output_dim).to(self.device)
        self.explorer = ExplorerIdentity(device)

    def train(self, epochs=100):
        """
        train the encoder to produce an embedding for the given dataset
        :param epochs: number of epochs to train for as int
        """
        # distance calculation for high dimensional data
        self.property_calculator.calculate_high_dim_property()

        for epoch in range(epochs):
            # tell the sampler that a new epoch is starting
            self.sampler.reset_epoch()

            while not self.sampler.epoch_done:
                # get batch of points
                ind = self.sampler.next_batch_indices()
                x, _ = self.sampler.get_points_from_indices(ind)
                x = x.to(self.device)

                # pass through encoder
                out1 = self.encoder_agent(x)
                out1 = self.explorer.get_point_from_output(out1)

                # get complementary indices corresponding to p1
                ind2 = self.sampler.next_complementary_indices(self.property_calculator)
                if ind2:
                    # get points
                    x2, _ = self.sampler.get_points_from_indices(ind2)
                    x2 = x2.to(self.device)

                    # pass through encoder
                    out2 = self.encoder_agent(x2)
                    out2 = self.explorer.get_point_from_output(out2)

                    # calculate low dim property
                    low_dim_property = self.property_calculator.get_low_dim_property(out1, out2)
                    high_prob = self.property_calculator.high_dim_property[ind, ind2]

                return