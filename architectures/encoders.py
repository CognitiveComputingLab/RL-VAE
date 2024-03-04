import torch.nn as nn


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


class EncoderSimple(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(EncoderSimple, self).__init__()
        # assuming each point has the same dimension as input_dim
        self.gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])

        # output layers for each point
        self.linear1 = nn.Linear(4096, latent_dim)

    def forward(self, sample_out):
        """
        pass point through general model
        :param sample_out: only contains x
        """
        # sample out only contains x
        x, _ = sample_out

        # pass through general model
        x = self.gm(x)

        # compute outputs for each point
        mu1 = self.linear1(x)
        return mu1


class EncoderUMAP(EncoderSimple):
    def __init__(self, input_dim, latent_dim):
        super().__init__(input_dim, latent_dim)

    def forward(self, sample_out):
        """
        pass point through general model
        :param sample_out: output of umap sampling, includes indices etc.
        """
        # get regular points from sample_out
        p1, p2, ind1, ind2 = sample_out

        # pass regular and complementary points through network
        mu1 = super().forward(p1)
        mu2 = super().forward(p2)

        return mu1, mu2, ind1, ind2


class EncoderVAE(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(EncoderVAE, self).__init__()
        self.gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])
        self.linearM = nn.Linear(4096, latent_dims)
        self.linearS = nn.Linear(4096, latent_dims)

    def forward(self, sample_out):
        """
        VAE model function
        :param sample_out: sample out containing only the simple points
        """
        # get x from normal points
        x, _ = sample_out

        # get distribution parameters
        x = self.gm(x)
        mu = self.linearM(x)
        log_var = self.linearS(x)

        # return parameters, assuming gaussian
        return mu, log_var


class EncoderKHeadVAE(nn.Module):
    def __init__(self, input_dim, latent_dims, k):
        super(EncoderKHeadVAE, self).__init__()
        self.k = k
        self.gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])
        self.linearM = nn.Linear(4096, k * latent_dims)
        self.linearS = nn.Linear(4096, k * latent_dims)
        self.weight_gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])
        self.linear_weight = nn.Linear(4096, k)

    def forward(self, sample_out):
        """
        compute k different VAE heads simultaneously
        :param sample_out: only the points are needed
        :return: 3 tensors
            - mu: means for each head, shape: [batch_size, num_heads, low_dims]
            - logvar: logvars for each head, shape: [batch_size, num_heads, low_dims]
            - weights: weights that represent the predicted strength of embedding for each head
        """
        # get x from sample out
        x, _ = sample_out

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
        weights = nn.functional.softmax(weights, dim=1)

        return mu, logvar, weights
