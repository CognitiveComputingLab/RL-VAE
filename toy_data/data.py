import abc
import os
import re

import numpy as np
import pandas as pd
from PIL import Image

from toy_data.plotting import scatter3d, mpl_2d_plot, change_lightness
from toy_data.util import DynamicImporter

from pitchscapes.keyfinding import KeyEstimator
from pitchscapes.plotting import key_scores_to_color

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

    @property
    def labels(self):
        if self._labels is None:
            return []
        return self._labels

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


class FashionMNIST(ToyData):
    def __init__(self, n):
        super().__init__(n)
        self._colors = None

    @property
    def colors(self):
        return self._label_color_mapper(cmap='hsv')

    def generate(self):
        # get data from csv file
        data_df = pd.read_csv("data/fashion_mnist/fashion-mnist_train.csv")

        # get labels
        labels = data_df['label'].to_numpy()
        data_df = data_df.drop('label', axis=1)

        # get rest of data as numpy array
        data_np = np.array([row.to_numpy() for index, row in data_df.iterrows()])
        data_np = data_np / 255

        # update class variables
        self._data = data_np[:self.n]
        self._labels = labels[:self.n]

        return self


class Coil100(ToyData):
    def __init__(self, n):
        super().__init__(n)

    @property
    def colors(self):
        return self._label_color_mapper(cmap='hsv')

    def generate(self):
        directory = "data/coil-100"
        image_files = [f for f in os.listdir(directory) if f.endswith(".png")]

        # keep track of data
        images_list = []
        labels_list = []

        for filename in image_files:
            file_path = os.path.join(directory, filename)
            label = int(re.search(r'obj(\d+)_', filename).group(1))

            # open the image file
            with Image.open(file_path) as img:
                # normalize and flatten the image
                img_normalized = np.array(img) / 255.0
                img_flattened = img_normalized.reshape(-1)

                # add to loaded images
                images_list.append(img_flattened)
                labels_list.append(label)

        # convert lists to numpy arrays
        images_array = np.stack(images_list)
        labels_array = np.stack(labels_list)

        # No need to reorder the dimensions since we're flattening the images
        self._data = images_array[:self.n]
        self._labels = labels_array[:self.n]

        return self


class Coil20(ToyData):
    def __init__(self, n):
        super().__init__(n)

    @property
    def colors(self):
        return self._label_color_mapper(cmap='hsv')

    def generate(self):
        directory = "data/coil-20"
        image_files = [f for f in os.listdir(directory) if f.endswith(".png")]

        # Keep track of data
        images_list = []
        labels_list = []

        for filename in image_files:
            file_path = os.path.join(directory, filename)
            label = int(re.search(r'obj(\d+)_', filename).group(1))

            # Open the image file
            with Image.open(file_path).convert('L') as img:  # 'L' mode for grayscale
                # Normalize and flatten the image
                img_normalized = np.array(img) / 255.0
                img_flattened = img_normalized.reshape(-1)

                # Add to loaded images
                images_list.append(img_flattened)
                labels_list.append(label)

        # Convert lists to numpy arrays
        images_array = np.stack(images_list)
        labels_array = np.stack(labels_list)

        self._data = images_array[:self.n]
        self._labels = labels_array[:self.n]

        return self


class MusicData(ToyData):
    def __init__(self, n):
        super().__init__(n)

    def generate(self):
        data_path = "data/data_WTK_r200_p0.0_s0.03_c0_sparse_corpus.npy"
        self._data = np.load(data_path)
        if self.n < len(self._data):
            self._data = self._data[:self.n, :]
        return self

    @property
    def colors(self):
        k = KeyEstimator()
        scores = k.get_score(self.data)
        colors = key_scores_to_color(scores, circle_of_fifths=True)
        return colors


class SphereND(ToyData):

    def __init__(self, n, d):
        super().__init__(n=n)
        self.d = d

    def generate(self):
        self._data = np.random.normal(size=(self.n, self.d))
        self._data /= np.sqrt((self._data ** 2).sum(axis=1, keepdims=True))
        return self


class Sphere3D(SphereND):

    def __init__(self, n):
        super().__init__(n=n, d=3)

    @property
    def labels(self):
        return np.concatenate([
            ((np.arctan2(self.data[:, 1], self.data[:, 0]) / np.pi + 1) / 2)[:, None],
            (np.arctan2(self.data[:, 2], np.sqrt(self.data[:, 1] ** 2 + self.data[:, 0] ** 2)) / np.pi + 0.5)[:, None],
        ],
            axis=1)

    @property
    def colors(self):
        colors = self._label_color_mapper(labels=self.labels[:, 0], cmap='hsv')
        colors = change_lightness(colors=colors, t=self.labels[:, 1])
        return colors

    def plot(self):
        scatter3d(self).show()


class Circle2D(SphereND):

    def __init__(self, n):
        super().__init__(n=n, d=2)

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

    def __init__(self, n=500, width=1, turns=1/2, color_scale=None):
        super().__init__(n=n)
        self.width = width
        self.turns = turns
        if int(2 * turns) != 2 * turns:
            raise ValueError(f"'turns' should be a multiple of 1/2 but is {turns}")
        if color_scale is None:
            if int(turns) == turns:
                color_scale = 0.8
            else:
                color_scale = 1.
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
            pos = unit_offset > 0.5
            unit_main_angle_half = unit_main_angle / 2
            unit_main_angle_half[pos] += 0.5
            self._labels = np.concatenate([unit_main_angle_half[:, None], (2 * np.abs(unit_offset - 0.5))[:, None]], axis=1)
        return self

    @property
    def colors(self):
        if self.color_scale == "3D":
            colors = self.data.copy()
            for i in range(3):
                colors[:, i] -= np.min(colors[:, i])
                colors[:, i] /= np.max(colors[:, i])
        else:
            colors = self._label_color_mapper(labels=self.labels[:, 0], cmap='hsv')
            colors = change_lightness(colors=colors, t=1 - self.labels[:, 1], scale=self.color_scale)
        return colors

    def plot(self):
        scatter3d(self).show()
