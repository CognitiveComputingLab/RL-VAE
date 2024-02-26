import torch
from architectures.Encoders.StandardModels import EncoderAgentUMAP
from architectures.PropertyCalculators.PropertyCalculatorUMAP import PropertyCalculatorUMAP
from architectures.Samplers.SamplerUMAP import SamplerUMAP
from architectures.Explorers.ExplorerIdentity import ExplorerIdentity
from architectures.Encoders.StandardModels import EncoderAgent
from architectures.PropertyCalculators.PropertyCalculator import PropertyCalculator
from architectures.Samplers.Sampler import Sampler


class EmbeddingFramework:
    def __init__(self, device, input_dim, output_dim):
        # init general
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim

        # components
        self.property_calculator = PropertyCalculatorUMAP(device)
        self.sampler = SamplerUMAP(device)
        self.encoder_agent = EncoderAgentUMAP(input_dim, output_dim).to(self.device)
        self.explorer = ExplorerIdentity(device)
        self.optimizer = torch.optim.Adam(list(self.encoder_agent.parameters()))

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

        # testing
        self.sampler.reset_epoch()
        ind = self.sampler.next_batch_indices()
        print("index shape: ", ind.shape)
        x, _ = self.sampler.get_points_from_indices(ind)
        print("point shape: ", x.shape)
        x = x.to(self.device)
        out1 = self.encoder_agent(x)
        out1 = self.explorer.get_point_from_output(out1)

        ind2 = self.sampler.next_complementary_indices(self.property_calculator)
        print("complementary index shape: ", ind2.shape)
        x2, _ = self.sampler.get_points_from_indices(ind2)
        x2 = x2.to(self.device)
        print("complementary point shape: ", x2.shape)

        out2 = self.encoder_agent(x2)
        out2 = self.explorer.get_point_from_output(out2)
        print("out2 shape: ", out2.shape)

        low_dim_property = self.property_calculator.get_low_dim_property(out1, out2)
        print("low dim property:", low_dim_property.shape)

        high_prob = self.property_calculator.symmetric_probabilities[ind, ind2]
        print("high prob: ", high_prob.shape)

        return
        for epoch in range(epochs):
            # tell the sampler that a new epoch is starting
            self.sampler.reset_epoch()

            while not self.sampler.epoch_done:
                # get batch of points
                ind = self.sampler.next_batch_indices()
                x, _ = self.sampler.get_points_from_indices(ind)
                x = x.to(self.device)

                # pass points through encoder
                out = self.encoder_agent(x)

                # get complementary sample for high/low dim comparison
                ind2 = self.sampler.next_complementary_indices()



