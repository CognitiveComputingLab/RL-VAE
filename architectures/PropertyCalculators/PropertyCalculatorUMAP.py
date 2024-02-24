import torch
import numpy as np
from scipy import optimize
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from PropertyCalculator import PropertyCalculator


class PropertyCalculatorUMAP(PropertyCalculator):
    def __init__(self, device):
        super().__init__(device)
        # umap specific
        self.symmetric_probabilities = None

        # hyperparameters
        self.k_neighbours = 15
        self.a = None
        self.b = None

    ###################################################
    # overwriting main property calculation functions #
    ###################################################

    def calculate_high_dim_property(self, train_data_loader):
        """
        compute all properties required for comparing high dimensional points
        in this case, compute symmetric pairwise probabilities
        :param train_data_loader: pytorch dataloader
        """
        # recover real dataset from data loader
        dataset = train_data_loader.dataset
        n = dataset.data.shape[0]

        rho, dist = self.calculate_high_dim_distances(dataset)
        prob = self.calculate_high_dim_probabilities(dist, rho, n)

        # get symmetric from both directions
        self.symmetric_probabilities = self.symmetrize(prob)

        # compute a and b parameters based on min distance
        x = np.linspace(0, 3, 300)
        p, _ = optimize.curve_fit(self.compute_low_dim_distance, x, self.f(x))
        self.a = p[0]
        self.b = p[1]

    def get_high_dim_property(self, ind, ind2):
        high_prob = torch.tensor([self.symmetric_probabilities[ind][ind2]]).float().to(self.device)
        return high_prob

    def get_low_dim_property(self, p1, p2):
        distance = torch.norm(p1 - p2, p=2, dim=1)
        out_prob = torch.pow(1 + self.a * distance.pow(2 * self.b), -1)
        return out_prob

    ##################################
    # UMAP specific helper functions #
    ##################################

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

    @staticmethod
    def calculate_high_dim_distances(dataset):
        """
        compute the high dimensional distances between points
        :param dataset: training dataset recovered from dataloader
        :return dist: euclidian distance matrix between pairs of points
        :return rho: list of closest neighbour for each datapoint
        """
        # compute high dimensional distances and save in square matrix
        dist = np.square(euclidean_distances(dataset.data, dataset.data))
        rho = [sorted(dist[i])[1] for i in range(dist.shape[0])]
        return rho, dist

    @staticmethod
    def symmetrize(prob):
        return prob + np.transpose(prob) - np.multiply(prob, np.transpose(prob))

    def calculate_high_dim_probabilities(self, dist, rho, n):
        prob = np.zeros((n, n))
        sigma_array = []
        for dist_row in tqdm(range(n)):
            func = lambda sigma: self.compute_n_neighbours(self.prob_high_dim(dist, rho, sigma, dist_row))
            binary_search_result = self.sigma_binary_search(func)
            prob[dist_row] = self.prob_high_dim(dist, rho, binary_search_result, dist_row)
            sigma_array.append(binary_search_result)
        return prob

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
