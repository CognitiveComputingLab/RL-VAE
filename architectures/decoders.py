import torch.nn as nn
from architectures.encoders import GeneralModel


class DecoderSimple(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(DecoderSimple, self).__init__()
        self.gm = GeneralModel(latent_dims, [512, 1024, 2024, 4096])
        self.linear = nn.Linear(4096, input_dim)

    def forward(self, transmitter_out):
        """
        take an input point
        :param transmitter_out: directly take input from transmitter output
            - can be structured in different ways
            - single point tensor
            - tuple
                -> first element in tuple will be taken as point tensor
        :return: decoded point tensor
        """
        z = transmitter_out
        if type(z) is tuple:
            z = transmitter_out[0]

        # pass through model
        z = self.gm(z)
        z = self.linear(z)

        # output decoded point tensor
        return z
