import abc
import numpy as np

from toy_data.plotting import scatter3d, mpl_2d_plot
from toy_data.util import DynamicImporter

plt = DynamicImporter('matplotlib.pyplot')
cm = DynamicImporter('matplotlib.cm')


class ToyData(abc.ABC):

    def __init__(self, n):
        self.n = n
        self._data = None
        self._labels = None

    @abc.abstractmethod
    def generate(self):
        """
        Generate the data. This should set the _data attribute. This method should return `self` to allow for chained
        calls.
        """
        return self

    def add_noise(self, scale=0.1):
        """
        Add Gaussian noise to data (i.e. the _data attribute) and should return `self` for chained calls.
        :return:
        """
        self._data += np.random.normal(scale=scale, size=self._data.shape)
        return self

    @property
    def data(self):
        """
        Returns the data as a read-only property. It assumes the _data attribute has been set and raises an error
        otherwise. If your generate() method does not set the _data attribute, you should override this property
        accordingly.
        """
        if self._data is None:
            raise RuntimeError("Data has not been generated. Call generate() method first.")
        else:
            return self._data

    @property
    def labels(self):
        if self._labels is None:
            raise AttributeError("No labels defined for this dataset")
        else:
            return self._labels

    @property
    def colors(self):
        raise AttributeError("No colours defined for this dataset")

    def _label_color_mapper(self, labels=None, cmap=None):
        if labels is None:
            labels = self.labels
        if cmap is not None:
            kw = dict(cmap='hsv')
        else:
            kw = dict()
        return cm.ScalarMappable(**kw).to_rgba(labels)

    def plot(self):
        raise NotImplementedError("No plotting function defined for this dataset")


class Circle2D(ToyData):

    def generate(self):
        self._data = np.random.normal(size=(self.n, 2))
        self._data /= np.sqrt((self._data ** 2).sum(axis=1, keepdims=True))
        return self

    @property
    def labels(self):
        return np.arctan2(self.data[:, 1], self.data[:, 0])

    @property
    def colors(self):
        return self._label_color_mapper(cmap='hsv')

    def plot(self):
        mpl_2d_plot(self)


class CircleND(ToyData):

    def __init__(self, n=500, width=5, wraps=None, dim=12):
        super().__init__(n=n)
        self.width = width
        if wraps is None:
            self.wraps = max(3, int(10 * np.ceil(width / dim)))
        else:
            self.wraps = wraps
        self.dim = dim

    def generate(self):
        # fill linear space symmetric around zero to later wrap around the circle
        self._data = np.linspace(-self.dim * self.wraps / 2,
                                 self.dim * self.wraps / 2,
                                 self.dim * self.wraps,
                                 endpoint=False)
        # add a random offset (also use as labels)
        self._labels = np.random.uniform(-self.dim / 2, self.dim / 2, self.n)
        # self._labels = np.linspace(-self.dim / 2, self.dim / 2, self.n, endpoint=False)
        self._data = self._data[:, None] - self._labels[None, :]
        # Gaussian shape
        self._data = np.exp(-self._data ** 2 / self.width ** 2)
        # wrap around circle and sum up overlapping points (this depends on underlying memory layout!)
        self._data = self._data.reshape(-1, self.dim, self.n).sum(axis=0)
        # transpose to have one data point per row
        self._data = self._data.T
        # normalise
        self._data /= self._data.sum(axis=1, keepdims=True)
        return self

    def add_noise(self, scale=None):
        if scale is None:
            scale = 0.1 / self.dim
        self._data += np.abs(np.random.normal(scale=scale, size=self._data.shape))
        self._data /= self._data.sum(axis=1, keepdims=True)
        return self

    @property
    def colors(self):
        return self._label_color_mapper(cmap='hsv')

    def plot(self):
        x = np.arange(self.data.shape[1])
        for y, c in zip(self.data, self.colors):
            plt.plot(x, y, color=c)
        plt.show()


class MoebiusStrip(ToyData):

    def __init__(self, n=500, width=1, turns=1/2, color_scale=0.8):
        super().__init__(n=n)
        self.width = width
        self.turns = turns
        if int(2 * turns) != 2 * turns:
            raise ValueError(f"'turns' should be a multiple of 1/2 but is {turns}")
        self.color_scale = color_scale

    def generate(self):
        unit_main_angle = np.random.uniform(0, 1, self.n)
        main_angle = unit_main_angle * 2 * np.pi
        self._data = np.zeros((self.n, 3))
        # main circle
        self._data[:, 0] = np.cos(main_angle)
        self._data[:, 1] = np.sin(main_angle)
        # strip offset
        twist_angle = main_angle * self.turns + 3 / 2 * np.pi
        unit_offset = np.random.uniform(0, 1, self.n)
        offset = (unit_offset - 1 / 2) * self.width
        self._data[:, 0] += offset * np.sin(twist_angle) * np.cos(main_angle)
        self._data[:, 1] += offset * np.sin(twist_angle) * np.sin(main_angle)
        self._data[:, 2] += offset * np.cos(twist_angle)
        # labels
        if int(self.turns) == self.turns:
            self._labels = np.concatenate([unit_main_angle[:, None], unit_offset[:, None]], axis=1)
        else:
            self._labels = np.concatenate([unit_main_angle[:, None], (2 * np.abs(unit_offset - 0.5))[:, None]], axis=1)
        return self

    @property
    def colors(self):
        colors = self._label_color_mapper(labels=self.labels[:, 0], cmap='hsv')
        t = self.labels[:, 1]
        pos = t > 0.5
        neg = t <= 0.5
        t_pos = 2 * (t[pos] - 0.5) * self.color_scale
        t_neg = (1 - 2 * t[neg]) * self.color_scale
        rgb = colors[:, 0:3]
        rgb[pos] = (1 - t_pos)[:, None] * rgb[pos] + t_pos[:, None] * np.zeros_like(rgb[pos])
        rgb[neg] = (1 - t_neg)[:, None] * rgb[neg] + t_neg[:, None] * np.ones_like(rgb[neg])
        return colors

    def plot(self):
        scatter3d(self).show()
