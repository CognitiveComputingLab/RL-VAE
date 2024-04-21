import abc
import numpy as np

from toy_data.util import DynamicImporter

plt = DynamicImporter('matplotlib.pyplot')
umap = DynamicImporter('umap')


class Embedding(abc.ABC):

    def __init__(self, dataset):
        self.dataset = dataset

    @abc.abstractmethod
    def fit(self, **kwargs):
        pass

    @abc.abstractmethod
    def plot(self):
        pass


class UMAP(Embedding):

    def __init__(self, dataset):
        super().__init__(dataset=dataset)
        self._embedding = None

    def fit(self, verbose=False, n_neighbors=500, **kwargs):
        kwargs = dict(n_neighbors=n_neighbors, verbose=verbose) | kwargs
        self._embedding = umap.UMAP(**kwargs).fit_transform(self.dataset.data)
        # self._embedding = umap.ParametricUMAP(**kwargs).fit_transform(self.dataset.data)
        return self

    def plot(self):
        # init
        # plt.figure(figsize=(10, 8))
        point_size = 12

        try:
            colors = self.dataset.colors
        except AttributeError:
            plt.scatter(self._embedding[:, 0], self._embedding[:, 1], s=point_size)
        else:
            plt.scatter(self._embedding[:, 0], self._embedding[:, 1], s=point_size, c=colors)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection')
        plt.savefig("images/umap-projection.png")
        plt.close()

    def save_raw(self):
        save_embedding = np.array([self._embedding])
        save_colors = np.array([self.dataset.colors])
        save_labels = np.array([self.dataset.labels])
        np.savez('images/raw-data.npz', embeddings=save_embedding, colors=save_colors, labels=save_labels,
                 high_dim_data=self.dataset.data)
