import random
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy import optimize
from tqdm import tqdm
import architectures.rl_vae as rl_vae
from architectures.rl_vae import GeneralModel
import warnings
import matplotlib.pyplot as plt
import torch.nn.functional as functional


# suppress warnings
warnings.filterwarnings('ignore')


# define the model
class EncoderAgentUMAP(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(EncoderAgentUMAP, self).__init__()
        # Assuming each point has the same dimension as input_dim
        self.gm = GeneralModel(input_dim, [1024, 2048, 2048, 4096])

        # Output layers for each point
        self.linear1 = nn.Linear(4096, latent_dims)

    def forward(self, x):
        # Concatenate the two input points
        x = self.gm(x)

        # Compute outputs for each point
        mu1 = self.linear1(x)
        return mu1


class UMAP_VAE(rl_vae.RlVae):
    def __init__(self, device, input_dim, latent_dimensions=2):
        super().__init__(device, input_dim, latent_dimensions)
        self.arch_name = "UMAP-VAE"
        self.device = device
        self.encoder_agent = EncoderAgentUMAP(input_dim, latent_dimensions).to(device)
        self.optimizer = torch.optim.AdamW(
            self.encoder_agent.parameters(),
            weight_decay=1e-2
        )

        # umap parameters
        self.k_neighbours = 15
        self.min_distance = 0.25
        self.a = 0
        self.b = 0
        self.symmetric_probabilities = None
        self.n_samples = self.k_neighbours

    @staticmethod
    def compute_n_neighbours(prob):
        """
        Compute n_neighbor = k (scalar) for each 1D array of high-dimensional probability
        """
        return np.power(2, np.sum(prob))

    @staticmethod
    def prob_high_dim(dist, rho, sigma, dist_row):
        """
        For each row of Euclidean distance matrix (dist_row) compute
        probability in high dimensions (1D array)
        """
        d = dist[dist_row] - rho[dist_row]
        d[d < 0] = 0
        return np.exp(- d / sigma)

    @staticmethod
    def compute_low_dim_distance(x, a, b):
        return 1 / (1 + a * x ** (2 * b))

    def f(self, x):
        y = []
        for i in range(len(x)):
            if x[i] <= self.min_distance:
                y.append(1)
            else:
                y.append(np.exp(- x[i] + self.min_distance))
        return y

    def compute_low_dim_probability(self, t1, t2):
        e_distance = torch.sqrt(torch.sum(torch.pow(torch.subtract(t1, t2), 2), dim=1))
        inv_distances = torch.pow(1 + self.a * torch.square(e_distance) ** self.b, -1)
        return inv_distances

    def sigma_binary_search(self, k_of_sigma):
        """
        Solve equation k_of_sigma(sigma) = fixed_k
        with respect to sigma by using the binary search algorithm
        """
        sigma_lower_limit = 0
        sigma_upper_limit = 1000
        approx_sigma = 0
        for i in range(20):
            approx_sigma = (sigma_lower_limit + sigma_upper_limit) / 2
            if k_of_sigma(approx_sigma) < self.k_neighbours:
                sigma_lower_limit = approx_sigma
            else:
                sigma_upper_limit = approx_sigma
            if np.abs(self.k_neighbours - k_of_sigma(approx_sigma)) <= 1e-5:
                break
        return approx_sigma

    def init_umap(self, dataset):
        """
        initialise the umap hyperparameters
        precompute distances / probabilities for high dimensional data
        """
        n = dataset.data.shape[0]

        # compute high dimensional distances
        self.console_log("Computing UMAP high-dimensional distances")
        dist = np.square(euclidean_distances(dataset.data, dataset.data))
        rho = [sorted(dist[i])[1] for i in range(dist.shape[0])]

        # compute high dimensional probabilities
        prob = np.zeros((n, n))
        sigma_array = []
        self.console_log("Computing UMAP variances")
        for dist_row in tqdm(range(n)):
            func = lambda sigma: self.compute_n_neighbours(self.prob_high_dim(dist, rho, sigma, dist_row))
            binary_search_result = self.sigma_binary_search(func)
            prob[dist_row] = self.prob_high_dim(dist, rho, binary_search_result, dist_row)
            sigma_array.append(binary_search_result)

        # get symmetric from both directions
        self.symmetric_probabilities = prob + np.transpose(prob) - np.multiply(prob, np.transpose(prob))

        # compute a and b parameters based on min distance
        self.console_log("computing UMAP parameters")
        x = np.linspace(0, 3, 300)
        p, _ = optimize.curve_fit(self.compute_low_dim_distance, x, self.f(x))
        self.a = p[0]
        self.b = p[1]

    def plot_latent(self, dataset, save_as, num_batches=100):
        """
        plot the umap projection of the points according to the nn umap
        """
        # Placeholder for collected points
        points = []
        colours = []

        # iterate over all and place objects
        for i in range(dataset.data.shape[0]):
            p_tensor = torch.from_numpy(dataset.data[i]).unsqueeze(0).to(self.device).float()
            colours.append(dataset.colors[i])
            p = self.encoder_agent(p_tensor)
            point = p
            points.append(point.detach().to('cpu').numpy())
        points = np.concatenate(points, axis=0)

        # Plotting
        plt.scatter(points[:, 0], points[:, 1], c=colours, cmap='tab10')
        plt.title('NN UMAP Projection')
        plt.savefig(save_as)
        plt.close()

    def sample(self, sample_index):
        """
        sample n_samples negative and positive samples for UMAP
        :param sample_index: the index of the point to sample neighbours for
        :return positive_indices: indices of positive samples (selected with positive probability weighting)
        :return negative_indices: indices of negative samples (selected with 1 - positive probability weighting)
        """
        # get probability tensor from numpy
        probabilities = torch.from_numpy(self.symmetric_probabilities[sample_index]).to(self.device).float()

        # exclude the specified index
        adjusted_probabilities = probabilities.clone()
        adjusted_probabilities[sample_index] = 0

        n_positive = torch.sum(adjusted_probabilities > 0).item()
        n_negative = torch.sum(adjusted_probabilities < 1).item() - 1

        # adjust n_samples based on available samples
        n_samples_positive = min(self.n_samples, n_positive)
        n_samples_negative = min(self.n_samples, n_negative)

        # sample positive
        positive_weights = adjusted_probabilities / adjusted_probabilities.sum()
        positive_indices = torch.multinomial(positive_weights, n_samples_positive, replacement=False)

        # sample negative
        adjusted_probabilities[sample_index] = 1
        negative_weights = 1 - adjusted_probabilities
        negative_weights /= negative_weights.sum()
        negative_indices = torch.multinomial(negative_weights, n_samples_negative, replacement=False)

        return positive_indices, negative_indices

    def train(self, dataset, epochs=10):
        """
        train the nn UMAP method on the given dataset
        :param dataset: ToyTorchDataset object and NOT a dataloader
        :param epochs: number of epochs to run for
        """
        loss_history = []
        for epoch in range(epochs):
            epoch_loss = 0

            # Initialize gradient accumulation
            self.optimizer.zero_grad()

            for ind in tqdm(range(dataset.data.shape[0])):
                # sample the second index
                pos_p, neg_p = self.sample(ind)
                samples = torch.concat([pos_p, neg_p])

                for ind2 in samples:
                    if ind2 == ind:
                        continue
                    ind2 = ind2.item()

                    p1_tensor = torch.from_numpy(dataset.data[ind]).unsqueeze(0).to(self.device).float()
                    p2_tensor = torch.from_numpy(dataset.data[ind2]).unsqueeze(0).to(self.device).float()

                    out1 = self.encoder_agent(p1_tensor)
                    out2 = self.encoder_agent(p2_tensor)

                    p1 = out1
                    p2 = out2
                    # p1 = y_t[ind] + out1
                    # p2 = y_t[ind2] + out2

                    distance = torch.norm(p1 - p2, p=2, dim=1)
                    out_prob = torch.pow(1 + self.a * distance.pow(2 * self.b), -1)

                    high_prob = torch.tensor([self.symmetric_probabilities[ind][ind2]]).float().to(self.device)

                    # Compute loss
                    loss = functional.binary_cross_entropy(out_prob, high_prob)
                    epoch_loss += loss.item()

                    # Backpropagation
                    loss.backward()  # Accumulate gradients for each pair

                self.optimizer.step()
                self.optimizer.zero_grad()

            avg_loss = epoch_loss / dataset.data.shape[0]
            loss_history.append(avg_loss)
            self.console_log(f"Average Cross-Entropy = {avg_loss} after {epoch} epochs")
            self.plot_latent(dataset, f'images/plot-epoch-{epoch}.png')
