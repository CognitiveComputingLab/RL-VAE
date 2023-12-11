import abc

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
        return self

    def plot(self):
        try:
            colors = self.dataset.colors
        except AttributeError:
            plt.scatter(self._embedding[:, 0], self._embedding[:, 1])
        else:
            plt.scatter(self._embedding[:, 0], self._embedding[:, 1], c=colors)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection')
        plt.show()
