"""
file contains helper functions for the RL-VAE etc.
"""
import torch
import numpy as np


class ToyTorchDataset(torch.utils.data.Dataset):
    """
    Converts a toy dataset (from toy_data)
    into a PyTorch dataset that can be loaded into a data loader
    """

    def __init__(self, dataset):
        self.data = dataset.data
        self.colors = dataset.colors

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        color = torch.tensor(self.colors[idx], dtype=torch.float32)
        return sample, color


def symmetric_kl_divergence_2d(mus, logvar):
    """
    Compute the distance between all given distributions and return the average over the entire batch
    mus and logvar shape must be: [batch_size, num_heads, num_dimensions]
    IMPORTANT: num_dimensions is only handled for 2 right now
    return: average over entire batch (average of averages)
    """
    average_distances = torch.tensor([], dtype=torch.float32)
    for b in range(mus.shape[0]):
        cumulative_distance = 0.0
        counter = 0
        for i in range(mus.shape[1]):
            for j in range(i + 1, mus.shape[1]):
                mu1 = mus[b][i]
                mu2 = mus[b][j]
                logvar1 = logvar[b][i]
                logvar2 = logvar[b][j]

                # Convert logvar to covariance matrices (assuming diagonal covariance)
                sigma1 = torch.exp(0.5 * logvar1)  # standard deviation for first Gaussian
                sigma2 = torch.exp(0.5 * logvar2)  # standard deviation for second Gaussian

                # Construct diagonal covariance matrices
                cov1 = torch.diag(sigma1.pow(2))
                cov2 = torch.diag(sigma2.pow(2))

                # Inverse and determinant of covariance matrices
                inv_cov1 = torch.inverse(cov1)
                inv_cov2 = torch.inverse(cov2)
                det_cov1 = torch.det(cov1)
                det_cov2 = torch.det(cov2)

                # Mean difference
                mu_diff = mu1 - mu2

                # KL divergence from Gaussian 1 to Gaussian 2
                kl_12 = 0.5 * (torch.trace(torch.matmul(inv_cov2, cov1))
                               + torch.dot(mu_diff, torch.matmul(inv_cov2, mu_diff))
                               - 2
                               + torch.log(det_cov2 / det_cov1))

                # KL divergence from Gaussian 2 to Gaussian 1
                kl_21 = 0.5 * (torch.trace(torch.matmul(inv_cov1, cov2))
                               + torch.dot(mu_diff, torch.matmul(inv_cov1, mu_diff))
                               - 2
                               + torch.log(det_cov1 / det_cov2))

                # Symmetric KL divergence
                kl_symmetric = kl_12 + kl_21
                cumulative_distance += kl_symmetric.item()  # Convert to Python float
                counter += 1

        current_average = cumulative_distance / counter
        average_distances = torch.cat((average_distances, torch.tensor([current_average], dtype=torch.float32)))

    return torch.mean(average_distances)


def compute_euclidian_distance(mus):
    """
    compute the euclidian distances between the means distributions
    mus: tensor with shape [batch_size, heads, dimensions]
    """
    average_distances = torch.tensor([], dtype=torch.float32)
    for b in range(mus.shape[0]):
        cumulative_distance = 0.0
        counter = 0
        for i in range(mus.shape[1]):
            for j in range(i + 1, mus.shape[1]):
                mu1 = mus[b][i]
                mu2 = mus[b][j]

                distance = torch.sqrt(torch.square(mu1[0] - mu2[0]) + torch.square(mu1[1] - mu2[1]))
                cumulative_distance += distance.item()  # Convert to Python float
                counter += 1

        current_average = cumulative_distance / counter
        average_distances = torch.cat((average_distances, torch.tensor([current_average], dtype=torch.float32)))

    return torch.mean(average_distances)


def scale_to_01(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


if __name__ == "__main__":
    input_mu = torch.Tensor([
        [[0.0115, -0.0214],
         [0.0194, 0.0226],
         [0.0042, -0.0058],
         [0.0083, 0.0015],
         [0.0235, 0.0034]],

        [[0.5, 0.5],
         [0.5, 0.5],
         [0.5, 0.5],
         [0.5, 0.5],
         [0.5, 0.5]],

        [[0.5, 0.5],
         [0.5, 0.5],
         [0.5, 0.5],
         [0.5, 0.5],
         [0.5, 0.5]]
    ])

    input_logvar = torch.Tensor([
        [[-0.0036, -0.0109],
         [0.0209, 0.0153],
         [0.0063, -0.0077],
         [-0.0310, 0.0057],
         [0.0183, 0.0011]],

        [[0.5, 0.5],
         [0.5, 0.5],
         [0.5, 0.5],
         [0.5, 0.5],
         [0.5, 0.5]],

        [[0.5, 0.5],
         [0.5, 0.5],
         [0.5, 0.5],
         [0.5, 0.5],
         [0.5, 0.5]]
    ])
    print(input_mu.shape)
    test = compute_euclidian_distance(input_mu)

    data = {t: np.random.rand(5, 2) for t in range(100)}
    print(data)

    print(test[0].shape)
