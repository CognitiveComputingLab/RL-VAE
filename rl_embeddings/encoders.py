import torch.nn as nn
import torch.nn.functional as f
from rl_embeddings.components import Component


class GeneralModel(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(GeneralModel, self).__init__()
        self.layers = nn.ModuleList()
        for h_dim in hidden_dims:
            self.layers.append(nn.Sequential(
                nn.Linear(input_dim, h_dim),
                nn.ReLU(),
            ))
            input_dim = h_dim

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConvGeneralModel(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(ConvGeneralModel, self).__init__()
        self.layers = nn.ModuleList()
        for h_dim in hidden_channels:
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, h_dim, 4, 2, 1),
                nn.ReLU(),
            ))
            in_channels = h_dim

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.flatten(1)
        return x


class EncoderSimple(nn.Module, Component):
    def __init__(self, input_dim, latent_dim):
        super(EncoderSimple, self).__init__()
        Component.__init__(self)
        self._required_inputs = ["points"]

        # assuming each point has the same dimension as input_dim
        self.gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])

        # output layers for each point
        self.linear1 = nn.Linear(4096, latent_dim)

    def forward(self, **kwargs):
        """
        pass point through general model
        """
        # check required arguments
        self.check_required_input(**kwargs)

        # sample out only contains x
        x, _ = kwargs["points"]

        # pass through general model
        x = self.gm(x)

        # compute outputs for each point
        mu1 = self.linear1(x)
        return {"encoded_points": mu1}


class EncoderUMAP(nn.Module, Component):
    def __init__(self, input_dim, latent_dim):
        super(EncoderUMAP, self).__init__()
        Component.__init__(self)
        self._required_inputs = ["points", "complementary_points"]

        # init network while assuming each point has the same dimension as input_dim
        self.gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])

        # output layers for each point
        self.linear1 = nn.Linear(4096, latent_dim)

    def forward(self, **kwargs):
        """
        pass regular points and complementary through general model
        """
        # check required arguments
        self.check_required_input(**kwargs)

        # get regular points from sample_out
        p1 = kwargs["points"]
        p2 = kwargs["complementary_points"]

        # get xs from points
        x1, _ = p1
        x2, _ = p2

        # pass regular and complementary points through network
        mu1 = self.linear1(f.relu(self.gm(x1)))
        mu2 = self.linear1(f.relu(self.gm(x2)))

        return {"encoded_points": mu1, "encoded_complementary_points": mu2}


class EncoderKHeadUMAP(nn.Module, Component):
    def __init__(self, input_dim, latent_dim, k=2):
        super(EncoderKHeadUMAP, self).__init__()
        Component.__init__(self)
        self._required_inputs = ["points", "complementary_points"]
        self.k = k

        # init network while assuming each point has the same dimension as input_dim
        self.gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])

        # output layers for each point
        self.linear1 = nn.Linear(4096, k * latent_dim)

        # head weights
        self.weight_gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])
        self.linear_weight = nn.Linear(4096, k)

    def forward(self, **kwargs):
        """
        pass regular points and complementary through general model for each head
        compute probability weights of choosing each head with softmax
        """
        # check required arguments
        self.check_required_input(**kwargs)

        # get regular points from sample_out
        p1 = kwargs["points"]
        p2 = kwargs["complementary_points"]

        # get xs from points
        x1, _ = p1
        x2, _ = p2

        # pass regular and complementary points through network
        mu1 = self.linear1(f.relu(self.gm(x1)))
        mu2 = self.linear1(f.relu(self.gm(x2)))

        # get the weights
        weights = self.weight_gm(x1)
        weights = self.linear_weight(weights)
        weights = nn.functional.softmax(weights, dim=1)

        return {"head_means": mu1, "head_complementary_means": mu2, "head_weights": weights}


class EncoderUMAPConv(EncoderUMAP):
    def __init__(self, input_dim, latent_dim):
        super(EncoderUMAPConv, self).__init__(input_dim, latent_dim)
        self.gm = ConvGeneralModel(input_dim, [16, 32, 64, 128, 256])


class EncoderVAE(nn.Module, Component):
    def __init__(self, input_dim, latent_dims):
        super(EncoderVAE, self).__init__()
        Component.__init__(self)
        self._required_inputs = ["points"]

        self.gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])
        self.linearM = nn.Linear(4096, latent_dims)
        self.linearS = nn.Linear(4096, latent_dims)

    def forward(self, **kwargs):
        """
        VAE model function
        """
        # check required arguments
        self.check_required_input(**kwargs)

        # get x from normal points
        x, _ = kwargs["points"]

        # get distribution parameters
        x = self.gm(x)
        x = f.relu(x)
        mu = self.linearM(x)
        log_var = self.linearS(x)

        # return parameters, assuming gaussian
        return {"means": mu, "log_vars": log_var}


class EncoderKHeadVAE(nn.Module, Component):
    def __init__(self, input_dim, latent_dims, k):
        super(EncoderKHeadVAE, self).__init__()
        Component.__init__(self)
        self._required_inputs = ["points"]

        self.k = k
        self.gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])
        self.linearM = nn.Linear(4096, k * latent_dims)
        self.linearS = nn.Linear(4096, k * latent_dims)
        self.weight_gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])
        self.linear_weight = nn.Linear(4096, k)

    def forward(self, **kwargs):
        """
        compute k different VAE heads simultaneously
        :param: only the points are needed
        :return: 3 tensors
            - mu: means for each head, shape: [batch_size, num_heads, low_dims]
            - logvar: logvars for each head, shape: [batch_size, num_heads, low_dims]
            - weights: weights that represent the predicted strength of embedding for each head
        """
        # check required arguments
        self.check_required_input(**kwargs)

        # get x from sample out
        x, _ = kwargs["points"]

        # pass through general model
        x2 = self.gm(x)

        # output the means
        mu = self.linearM(x2)
        batch_size = mu.size(0)
        mu = mu.view(batch_size, self.k, -1)

        # output the log-vars
        logvar = self.linearS(x2)
        batch_size = logvar.size(0)
        logvar = logvar.view(batch_size, self.k, -1)

        weights = self.weight_gm(x)
        weights = self.linear_weight(weights)
        # weights = nn.functional.softmax(weights, dim=1)

        return {"head_means": mu, "head_log_vars": logvar, "head_weights": weights}
