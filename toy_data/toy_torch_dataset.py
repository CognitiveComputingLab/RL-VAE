"""
file contains helper functions for the RL-VAE etc.
"""
import torch


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

