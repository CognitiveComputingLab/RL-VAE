import abc
import torch
import torch.nn as nn
import numpy as np
from scipy import optimize
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from rl_embeddings.components import Component


class SimilarityCalculator(nn.Module, Component, abc.ABC):
    def __init__(self, device, data_loader):
        super().__init__()
        Component.__init__(self)

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
        symmetrize the high dimensional similarity
        """
        raise NotImplementedError

    ###############################
    # high dim similarity functions #
    ###############################

    @property
    @abc.abstractmethod
    def high_dim_similarity(self):
        raise NotImplementedError

    @abc.abstractmethod
    def calculate_high_dim_similarity(self):
        """
        compute all similarities required for comparing high dimensional points
        """
        raise NotImplementedError

    ###################################
    # standard pytorch module methods #
    ###################################

    @abc.abstractmethod
    def forward(self, **kwargs):
        """
        compute similarities to calculate
        :return: low and high dimensional similarities according to the explorer output
        """
        raise NotImplementedError


class SimilarityCalculatorUMAP(SimilarityCalculator):
    # some code adapted from https://towardsdatascience.com/how-to-program-umap-from-scratch-e6eff67f55fe
    def __init__(self, device, data_loader):
        super().__init__(device, data_loader)
        self._required_inputs = ["encoded_points", "encoded_complementary_points", "indices", "complementary_indices"]

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

    def forward(self, **kwargs):
        """
        get low and high dimensional similarities as defined in the UMAP paper
        high dimensional similarity should already be calculated, needs to be retrieved for explorer_out
        low dimensional similarity is calculated based on distributions with hyperparameters of high dim
        :return: tuple of tensors
            - low dimensional similarities
            - high dimensional similarities
        """
        # check required arguments
        self.check_required_input(**kwargs)

        # get points from exploration
        p1 = kwargs["encoded_points"]
        p2 = kwargs["encoded_complementary_points"]
        ind1 = kwargs["indices"]
        ind2 = kwargs["complementary_indices"]

        # get similarities
        low_dim_similarity = self.get_low_dim_similarity(p1, p2)
        high_dim_similarity = self.high_dim_similarity[ind1, ind2].float().to(self._device)

        return {"low_dim_similarity": low_dim_similarity, "high_dim_similarity": high_dim_similarity}

    ##############################################
    # overwriting similarity calculation functions #
    ##############################################

    @property
    def high_dim_similarity(self):
        return self.__symmetric_probabilities

    def symmetrize(self, prob, n):
        return prob + np.transpose(prob) - np.multiply(prob, np.transpose(prob))

    def calculate_high_dim_similarity(self):
        """
        compute all similarities required for comparing high dimensional points
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

    def get_low_dim_similarity(self, p1, p2):
        """
        calculate low dimensional similarity
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


class SimilarityCalculatorTSNE(SimilarityCalculator):
    # some code adapted from https://towardsdatascience.com/understanding-t-sne-by-implementing-2baf3a987ab3
    def __init__(self, device, data_loader):
        super().__init__(device, data_loader)
        self._required_inputs = ["encoded_points", "indices"]

        # init
        self.__symmetric_probabilities = None

        # hyperparameters
        self.perplexity = 3

    @property
    def high_dim_similarity(self):
        return self.__symmetric_probabilities

    def symmetrize(self, prob, n):
        sym_prob = (prob + prob.t()) / (2.0 * n)
        return sym_prob

    def calculate_high_dim_similarity(self):
        """
        calculate high dimensional distances according to T-SNE paper
        this is called only once before proper training starts
        later used for comparison with lower dimensional distances
        """
        # recover real dataset from data loader
        dataset = self._data_loader.dataset
        dataset_tensor = torch.tensor(dataset.data, requires_grad=False).float().to(self._device)

        # pairwise distances between points as matrix
        pwd = self.pairwise_distances_cdist(dataset_tensor)

        # binary search for sigma values according to perplexity hyperparameter
        sigmas = self.find_sigmas(pwd)

        # get conditional pairwise distances
        c_pwd = self.p_conditional(pwd, sigmas, normalize=False)

        self.__symmetric_probabilities = c_pwd

    def get_low_dim_similarity(self, p1):
        """
        low dim similarity based on encoded points
        """
        distances = self.pairwise_distances_cdist(p1)
        nom = 1 / (1 + distances)
        nom_div = torch.sum(nom.clone().fill_diagonal_(0))
        return nom / nom_div

    def forward(self, **kwargs):
        """
        get low and high dimensional similarities as defined in the TSNE paper
        high dimensional similarity should already be calculated, needs to be retrieved for explorer_out
        low dimensional similarity is calculated based on distributions with hyperparameters of high dim
        :return:
            - low dimensional similarities
            - high dimensional similarities
        """
        # check required arguments
        self.check_required_input(**kwargs)

        # get points from exploration
        p1 = kwargs["encoded_points"]
        ind1 = kwargs["indices"]

        # low dim distances
        low_dist = self.get_low_dim_similarity(p1)
        low_dist.fill_diagonal_(0)

        # high dim distances for this specific batch
        high_dist = self.high_dim_similarity[ind1][:, ind1]
        # normalize according to batch
        high_dist = high_dist / high_dist.sum(axis=1).reshape([-1, 1])
        high_dist.fill_diagonal_(0)
        # make distances symmetric
        high_symmetric_distances = self.symmetrize(high_dist, len(ind1))

        return {"low_dim_similarity": low_dist, "high_dim_similarity": high_symmetric_distances}

    ##########################
    # T-SNE helper functions #
    ##########################

    @staticmethod
    def pairwise_distances(x):
        return np.sum((x[None, :] - x[:, None]) ** 2, 2)

    @staticmethod
    def pairwise_distances_torch(x):
        # expand for easy computation
        p_i = x.unsqueeze(0)
        p_j = x.unsqueeze(1)

        # calculate squared euclidian norm
        diff = p_i - p_j
        diff = diff ** 2
        diff = diff.sum(-1)
        return diff

    @staticmethod
    def pairwise_distances_cdist(x):
        return torch.cdist(x, x).pow(2)

    @staticmethod
    def p_conditional(dists, sigmas, normalize=True):
        """
        Get conditional probabilities between points in high dimension using PyTorch.
        :param dists: pairwise distances between points in 2d tensor
        :param sigmas: compute sigmas for gaussian distributions
        :param normalize:
            if set to False: does NOT normalize, because this is done for the batch specifically.
        """
        # reshape sigmas for broadcasting
        sigmas = sigmas.reshape(-1, 1)

        # main calculation
        e = torch.exp(-dists / (2 * torch.square(sigmas)))

        # avoid division by zero or log of zero in subsequent operations
        e.fill_diagonal_(0)
        e += 1e-8

        # normalize
        if normalize:
            e = e / e.sum(dim=1, keepdim=True)

        return e

    @staticmethod
    def perp(conditional_matrix):
        ent = -torch.sum(conditional_matrix * torch.log2(conditional_matrix), dim=1)
        return 2 ** ent

    @staticmethod
    def binary_search(func, goal, tol=1e-10, max_iters=20, lower_bound=1e-20, upper_bound=10000):
        """
        binary search to find sigmas according to perplexity hyperparameter
        """
        guess = torch.tensor((upper_bound + lower_bound) / 2.0)
        for _ in range(max_iters):
            guess = (upper_bound + lower_bound) / 2.0
            val = func(guess)

            if val > goal:
                upper_bound = guess
            else:
                lower_bound = guess

            if abs(val - goal) <= tol:
                return guess
        return guess

    def find_sigmas(self, dists):
        # tensor for the sigma values, on the same device as dists
        found_sigmas = torch.zeros(dists.shape[0], device=dists.device)

        for i in tqdm(range(dists.shape[0])):
            # lambda function for the current row of dists
            func = lambda sig: self.perp(
                self.p_conditional(dists[i:i + 1, :], torch.tensor([sig], device=dists.device)))
            # binary search to find the sigma that matches the perplexity for the row
            found_sigmas[i] = self.binary_search(func, self.perplexity)
        return found_sigmas
