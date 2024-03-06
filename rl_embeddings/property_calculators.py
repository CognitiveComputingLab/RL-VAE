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
    def symmetrize(self, prob, n):
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

    def symmetrize(self, prob, n):
        return

    def calculate_high_dim_property(self):
        return

    def forward(self, explorer_out):
        return


class PropertyCalculatorUMAP(PropertyCalculator):
    # some code adapted from https://towardsdatascience.com/how-to-program-umap-from-scratch-e6eff67f55fe
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

    def symmetrize(self, prob, n):
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
        sym_probabilities = self.symmetrize(prob, n)
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


class PropertyCalculatorTSNE(PropertyCalculator):
    # some code adapted from https://towardsdatascience.com/understanding-t-sne-by-implementing-2baf3a987ab3
    def __init__(self, device, data_loader):
        super().__init__(device, data_loader)
        self.__symmetric_probabilities = None

        # hyperparameters
        self.perplexity = 15

    @property
    def high_dim_property(self):
        return self.__symmetric_probabilities

    def symmetrize(self, prob, n):
        return (prob + prob.T) / (2. * n)

    def calculate_high_dim_property(self):
        """
        calculate high dimensional distances according to T-SNE paper
        this is called only once before proper training starts
        later used for comparison with lower dimensional distances
        """
        # recover real dataset from data loader
        dataset = self._data_loader.dataset
        n = dataset.data.shape[0]

        # pairwise distances between points as matrix
        pwd = self.pairwise_distances(dataset.data)

        # binary search for sigma values according to perplexity hyperparameter
        sigmas = self.find_sigmas(pwd)

        # get conditional pairwise distances
        c_pwd = self.p_conditional(pwd, sigmas)

        self.__symmetric_probabilities = c_pwd

    def get_low_dim_property(self, p1):
        """
        low dim property based on encoded points
        """
        distances = self.pairwise_distances_torch(p1)
        nom = 1 / (1 + distances)
        nom.fill_diagonal_(0)
        return nom / torch.sum(torch.sum(nom))

    def forward(self, explorer_out):
        """
        get low and high dimensional properties as defined in the TSNE paper
        high dimensional property should already be calculated, needs to be retrieved for explorer_out
        low dimensional property is calculated based on distributions with hyperparameters of high dim
        :return: tuple of tensors
            - low dimensional properties
            - high dimensional properties
        """
        # get points from exploration
        p1, ind1 = explorer_out

        # low dim distances
        low_dist = self.get_low_dim_property(p1)

        # high dim distances for this specific batch
        high_dist = self.high_dim_property[ind1][:, ind1]
        # normalize according to batch
        high_dist = high_dist / high_dist.sum(axis=1).reshape([-1, 1])
        # make distances symmetric
        high_symmetric_distances = self.symmetrize(high_dist, len(ind1))
        # convert to tensor, do not need any grad
        high_symmetric_distances = torch.tensor(high_symmetric_distances).float().to(self._device)
        high_symmetric_distances.fill_diagonal_(0)

        return low_dist, high_symmetric_distances

    ##########################
    # T-SNE helper functions #
    ##########################

    @staticmethod
    def pairwise_distances(x):
        return np.sum((x[None, :] - x[:, None]) ** 2, 2)

    @staticmethod
    def pairwise_distances_torch(x):
        x = x.unsqueeze(1)
        diff = x - x.transpose(0, 1)
        dist_squared = torch.sum(diff ** 2, dim=2)
        return dist_squared

    @staticmethod
    def p_conditional(dists, sigmas):
        """
        get conditional probabilities between points in high dimension
        does NOT normalize, because this is done for the batch specifically
        """
        e = np.exp(-dists / (2 * np.square(sigmas.reshape((-1, 1)))))
        np.fill_diagonal(e, 0.)
        e += 1e-8
        return e

    @staticmethod
    def perp(conditional_matrix):
        ent = -np.sum(conditional_matrix * np.log2(conditional_matrix), 1)
        return 2 ** ent

    @staticmethod
    def binary_search(func, goal, tol=1e-10, max_iters=1000, lower_bound=1e-20, upper_bound=10000):
        """
        binary search to find sigmas according to perplexity hyperparameter
        """
        guess = 0
        for _ in range(max_iters):
            guess = (upper_bound + lower_bound) / 2.
            val = func(guess)

            if val > goal:
                upper_bound = guess
            else:
                lower_bound = guess

            if np.abs(val - goal) <= tol:
                return guess
        return guess

    def find_sigmas(self, dists):
        found_sigmas = np.zeros(dists.shape[0])
        for i in range(dists.shape[0]):
            func = lambda sig: self.perp(self.p_conditional(dists[i:i + 1, :], np.array([sig])))
            found_sigmas[i] = self.binary_search(func, self.perplexity)
        return found_sigmas
