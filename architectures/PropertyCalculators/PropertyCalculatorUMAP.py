import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from PropertyCalculator import PropertyCalculator


class PropertyCalculatorUMAP(PropertyCalculator):
    def __init__(self):
        super().__init__()

    def compute_high_dim_distances(self, train_data_loader):
        """
        compute the high dimensional distances between points
        :param train_data_loader: pytorch dataloader
        """
        # recover real dataset from data loader
        dataset = train_data_loader.dataset
        n = dataset.data.shape[0]

        # compute high dimensional distances and save in square matrix
        dist = np.square(euclidean_distances(dataset.data, dataset.data))
        rho = [sorted(dist[i])[1] for i in range(dist.shape[0])]