import numpy as np

from Orange.projection import _som


class SOM:
    def __init__(self, dim_x, dim_y, hexagonal=False):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.weights = None
        self.hexagonal = hexagonal

    def init_weights(self, x):
        self.weights = np.random.random((self.dim_y, self.dim_x, x.shape[1]))
        norms = np.sum(self.weights ** 2, axis=2)
        self.weights /= norms[:, :, None]

    def fit(self, x, n_iterations, learning_rate=0.5, sigma=1.0, callback=None):
        self.init_weights(x)
        update = _som.update_hex if self.hexagonal else _som.update
        for iteration in range(n_iterations):
            decay = 1 + iteration / (n_iterations / 2)
            update(self.weights, x, sigma / decay, learning_rate / decay)
            if callback is not None:
                callback(iteration / n_iterations)

    def winner(self, row):
        winner = np.empty(2, dtype=np.int16)
        _som.get_winner(self.weights, row, winner, int(self.hexagonal))
        return tuple(winner)
