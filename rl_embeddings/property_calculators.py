import abc
import torch
import torch.nn as nn
import numpy as np
from scipy import optimize
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances


class PropertyCalculator(nn.Module, abc.ABC):
    def __init__(self, device, data_loader):
        super().__init__()

        self._device = device
        self.__disable_tqdm = False

        self._data_loader = data_loader

    @property
    def disable_tqdm(self):
        return self.__disable_tqdm

    @disable_tqdm.setter
    def disable_tqdm(self, value):
        self.__disable_tqdm = value

    @abc.abstractmethod
    def symmetrize(self, prob):
        """
        symmetrize the high dimensional property
        """
        raise NotImplementedError

    ###############################
    # high dim property functions #
    ###############################

    @property
    @abc.abstractmethod
    def high_dim_property(self):
        raise NotImplementedError

    @abc.abstractmethod
    def calculate_high_dim_property(self):
        """
        compute all properties required for comparing high dimensional points
        """
        raise NotImplementedError

    ###################################
    # standard pytorch module methods #
    ###################################

    @abc.abstractmethod
    def forward(self, explorer_out):
        """
        compute properties to calculate
        :return: low and high dimensional properties according to the explorer output
        """
        raise NotImplementedError


class PropertyCalculatorNone(PropertyCalculator):
    def __init__(self, device, data_loader):
        super().__init__(device, data_loader)

    @property
    def high_dim_property(self):
        return

    def symmetrize(self, prob):
        return

    def calculate_high_dim_property(self):
        return

    def forward(self, explorer_out):
        return


class PropertyCalculatorUMAP(PropertyCalculator):
    def __init__(self, device, data_loader):
        super().__init__(device, data_loader)
        # umap specific
        self.__symmetric_probabilities = None

        # hyperparameters
        self.k_neighbours = 15
        self.min_distance = 0.25
        self.__a = None
        self.__b = None

    ##########################
    # pytorch module forward #
    ##########################

    def forward(self, explorer_out):
        """
        get low and high dimensional properties as defined in the UMAP paper
        high dimensional property should already be calculated, needs to be retrieved for explorer_out
        low dimensional property is calculated based on distributions with hyperparameters of high dim
        :return: tuple of tensors
            - low dimensional properties
            - high dimensional properties
        """
        # get points from exploration
        p1, p2, ind1, ind2 = explorer_out

        # get properties
        low_dim_property = self.get_low_dim_property(p1, p2)
        high_dim_property = self.high_dim_property[ind1, ind2].float().to(self._device)

        return low_dim_property, high_dim_property

    ##############################################
    # overwriting property calculation functions #
    ##############################################

    @property
    def high_dim_property(self):
        return self.__symmetric_probabilities

    def symmetrize(self, prob):
        return prob + np.transpose(prob) - np.multiply(prob, np.transpose(prob))

    def calculate_high_dim_property(self):
        """
        compute all properties required for comparing high dimensional points
        in this case, compute symmetric pairwise probabilities
        """
        # recover real dataset from data loader
        dataset = self._data_loader.dataset
        n = dataset.data.shape[0]

        rho, dist = self.calculate_high_dim_distances(dataset)
        prob = self.calculate_high_dim_probabilities(dist, rho, n)

        # get symmetric from both directions
        sym_probabilities = self.symmetrize(prob)
        self.__symmetric_probabilities = torch.tensor(sym_probabilities).float()

        # compute a and b parameters based on min distance
        x = np.linspace(0, 3, 300)
        p, _ = optimize.curve_fit(self.compute_low_dim_distance, x, self.f(x))
        self.__a = p[0]
        self.__b = p[1]

    def get_low_dim_property(self, p1, p2):
        """
        calculate low dimensional property
        in this case probability of being neighbours between low dimensional points
        :param p1: first point as pytorch Tensor
        :param p2: second point as pytorch Tensor
        """
        distance = torch.norm(p1 - p2, p=2, dim=1)
        out_prob = torch.pow(1 + self.__a * distance.pow(2 * self.__b), -1)
        return out_prob

    ##################################
    # UMAP specific helper functions #
    ##################################

    def f(self, x):
        y = []
        for i in range(len(x)):
            if x[i] <= self.min_distance:
                y.append(1)
            else:
                y.append(np.exp(- x[i] + self.min_distance))
        return y

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

    def calculate_high_dim_probabilities(self, dist, rho, n):
        prob = np.zeros((n, n))
        sigma_array = []
        for dist_row in tqdm(range(n), disable=self.disable_tqdm):
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