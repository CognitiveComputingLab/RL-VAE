import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.manifold import SpectralEmbedding


class SpectralPreTrainer:
    def __init__(self, emb_model, device, data_loader):
        self._emb_model = emb_model
        self._device = device
        self._data_loader = data_loader

        self.batch_size = 100

        # compute spectral embedding
        sp_embedding = SpectralEmbedding(n_components=2)
        self.data = torch.from_numpy(data_loader.dataset.data).to(self._device).float()
        self.colors = torch.from_numpy(data_loader.dataset.colors).to(device).float()
        self.embedded = sp_embedding.fit_transform(data_loader.dataset.data)
        self.embedded = torch.from_numpy(self.embedded).to(self._device).float()

    def pre_train_encoder(self, epochs=10):
        """
        pre-train the encoder on spectral embedding for given data
        """
        # check for encoder
        if not hasattr(self._emb_model, "encoder"):
            raise NotImplementedError("Trying to pre-train encoder, but this model does not have an encoder.")

        optimizer = torch.optim.Adam(list(self._emb_model.parameters()))
        loss_fn = nn.MSELoss()

        for epoch in tqdm(range(epochs), disable=False):

            # run through epoch
            epoch_done = False
            while not epoch_done:
                # embed with model
                sampler_out = {"points": (self.data, self.colors)}
                out = self._emb_model.encoder(**sampler_out)
                if hasattr(self._emb_model, "explorer"):
                    out["epoch"] = epoch
                    out = self._emb_model.explorer(**out)

                # compare embedding with spectral
                encoded_points = out["encoded_points"]
                loss = loss_fn(encoded_points, self.embedded)
                print("loss ", loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break

    def pre_train_decoder(self):
        """
        pre-train the decoder on spectral embedding for given data
        """
        # check for decoder
        if not hasattr(self._emb_model, "encoder"):
            raise NotImplementedError("Trying to pre-train decoder, but this model does not have a decoder.")
