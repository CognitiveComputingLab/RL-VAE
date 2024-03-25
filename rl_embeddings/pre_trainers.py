import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt


class PreTrainerSpectral:
    def __init__(self, emb_model, device, data_loader):
        self._emb_model = emb_model
        self._device = device
        self._data_loader = data_loader

        self.batch_size = 100

        # compute spectral embedding
        sp_embedding = SpectralEmbedding(n_components=2)
        self.data = torch.from_numpy(data_loader.dataset.data).to(self._device).float()
        self.embedded = sp_embedding.fit_transform(data_loader.dataset.data)
        self.embedded = torch.from_numpy(self.embedded).to(self._device).float()
        self.colors = torch.from_numpy(data_loader.dataset.colors).to(device).float()

    def plot_spectral(self, path):
        colors = self.colors[:, :3].to('cpu').detach().numpy()
        embedded = self.embedded.to('cpu').detach().numpy()
        plt.scatter(embedded[:, 0], embedded[:, 1], c=colors)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(f'latent projection')
        plt.savefig(path)
        plt.close()

    def pre_train(self, epochs=10):
        """
        pre-train the embedding method networks on spectral embedding
        """
        if not hasattr(self._emb_model, "encoder"):
            raise NotImplementedError("Trying to pre-train, but this model does not have an encoder.")

        # pre-train the encoder first
        self.pre_train_encoder(epochs)

        # pre-train decoder based on encoder pre-training
        if hasattr(self._emb_model, "decoder"):
            self.pre_train_decoder(epochs)

    def pre_train_encoder(self, epochs):
        """
        pre-train the encoder on spectral embedding for given data
        """
        optimizer = torch.optim.Adam(list(self._emb_model.encoder.parameters()))
        loss_fn = nn.MSELoss()

        for epoch in tqdm(range(epochs), disable=False):

            # run through epoch
            for i in range(0, len(self.data), self.batch_size):
                # get batch of datapoints
                batch_points = self.data[i:i+self.batch_size]
                batch_embedded = self.embedded[i:i+self.batch_size]

                # embed with model
                sampler_out = {"points": (batch_points, None), "complementary_points": (batch_points, None)}
                out = self._emb_model.encoder(**sampler_out)
                if hasattr(self._emb_model, "explorer"):
                    out["epoch"] = epoch
                    out = self._emb_model.explorer(**out)

                # compare embedding with spectral
                encoded_points = out["encoded_points"]
                loss = loss_fn(encoded_points, batch_embedded)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def pre_train_decoder(self, epochs):
        """
        pre-train the decoder on spectral embedding for given data
        """
        return
