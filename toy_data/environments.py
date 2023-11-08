import abc

import numpy as np

from data import ToyData, Circle2D
from plotting import scatter3d
from util import DynamicImporter

plt = DynamicImporter('matplotlib.pyplot')


class Environment(ToyData):

    def __init__(self, n, length):
        super().__init__(n=n)
        self.length = length

    @abc.abstractmethod
    def flatten(self):
        """
        Flatten the sampled trajectories to a set of observations
        """
        return self


class RandomWalker(Environment):

    def __init__(self, n=100, length=100, dim=2, step_scale=0.1, init='normal', drift='centre'):
        super().__init__(n=n, length=length)
        self.dim = dim
        self.step_scale = step_scale
        if init == 'normal':
            self.init = lambda n, dim: np.random.normal(size=(n, dim))
        elif init == 'circle':
            if self.dim < 2:
                raise ValueError("Circular initialisation only possible in two or more dimensions")

            def init(n, dim):
                init_2d = Circle2D(n).generate().data
                if dim > 2:
                    init_nd = np.zeros((n, dim))
                    init_nd[:, 0:2] = init_2d
                    return init_nd
                else:
                    return init_2d

            self.init = init
        elif init == 'sphere':
            def init(n, dim):
                init = np.random.normal(size=(n, dim))
                init /= np.sqrt((init ** 2).sum(axis=1, keepdims=True))
                return init

            self.init = init
        else:
            if callable(init):
                self.init = init
            else:
                raise TypeError("'init' must be callable")
        if drift == 'centre':
            self.drift = lambda x: -x / 100
        elif drift == 'circle':
            if self.dim < 2:
                raise ValueError("Circular drift only possible in two or more dimensions")

            def drift(x):
                distance = np.sqrt((x[:, 0:2] ** 2).sum(axis=1, keepdims=True))
                unit_force = x[:, 0:2] / distance
                force = np.zeros_like(x)
                force[:, 0:2] = unit_force * (1 - distance) / 10
                force[:, 2:] = -x[:, 2:] / 10
                return force

            self.drift = drift
        elif drift == 'sphere':
            def drift(x):
                distance = np.sqrt((x ** 2).sum(axis=1, keepdims=True))
                unit_force = x / distance
                return unit_force * (1 - distance) / 10

            self.drift = drift
        else:
            if callable(drift):
                self.drift = drift
            else:
                raise TypeError("'drift' must be callable")

    def generate(self):
        init_location = self.init(self.n, self.dim)
        steps = np.random.normal(scale=self.step_scale, size=(self.n, self.length, self.dim))
        if self.drift is None:
            self._data = np.cumsum(steps, axis=1)
            self._data += init_location[:, None, :]
        else:
            self._data = np.zeros((self.n, self.length, self.dim))
            for i in range(self.length):
                if i == 0:
                    self._data[:, i, :] = init_location
                else:
                    self._data[:, i, :] = self._data[:, i - 1, :]
                self._data[:, i, :] += steps[:, i, :]
                self._data[:, i, :] += self.drift(self._data[:, i, :])
        return self

    def flatten(self):
        self._data = self._data.reshape(-1, self.dim)
        return self

    def plot(self):
        if self.dim > 3:
            raise RuntimeError("Cannot plot in more than 3 dimensions")
        elif self.dim == 2:
            if len(self.data.shape) == 2:
                plt.plot(self.data[:, 0], self.data[:, 1], 'o', ms=1, alpha=0.2)
            else:
                plt.plot(self.data[:, :, 0].T, self.data[:, :, 1].T, '-o', lw=1, ms=1, alpha=0.2)
            plt.gca().set_aspect('equal', 'datalim')
            plt.show()
        else:
            if len(self.data.shape) == 2:
                fig = scatter3d(self)
            else:
                fig = None
                for i in range(self.n):
                    fig = scatter3d(data=self.data[i], fig=fig, mode='markers+lines')
            fig.show()
