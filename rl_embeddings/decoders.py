import torch.nn as nn
from rl_embeddings.encoders import GeneralModel
from rl_embeddings.components import Component


class DecoderSimple(nn.Module, Component):
    def __init__(self, input_dim, latent_dims):
        super(DecoderSimple, self).__init__()
        Component.__init__(self)
        self._required_inputs = ["sample"]

        self.gm = GeneralModel(latent_dims, [512, 1024, 2024, 4096])
        self.linear = nn.Linear(4096, input_dim)

    def forward(self, **kwargs):
        """
        pass input through decoder pytorch network
        :return: decoded point tensor
        """
        # check required arguments
        self.check_required_input(**kwargs)

        # get information from arguments
        z = kwargs["sample"]

        # pass through model
        z = self.gm(z)
        z = self.linear(z)

        # output decoded point tensor
        return {"decoded_points": z}
