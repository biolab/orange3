import numpy as np
import scipy.sparse as sp

from Orange.projection import _som


class SOM:
    def __init__(self, dim_x, dim_y, hexagonal=False):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.weights = self.ssum_weights = None
        self.hexagonal = hexagonal

    def init_weights(self, x):
        self.weights = np.random.random((self.dim_y, self.dim_x, x.shape[1]))
        norms = np.sum(self.weights ** 2, axis=2)
        self.ssum_weights = np.ones((self.dim_y, self.dim_x))
        self.weights /= norms[:, :, None]

    def fit(self, x, n_iterations, learning_rate=0.5, sigma=1.0, callback=None):
        if sp.issparse(x):
            f = _som.update_sparse_hex if self.hexagonal else _som.update_sparse

            def update(decay):
                f(self.weights, self.ssum_weights, x,
                  sigma / decay, learning_rate / decay)
        else:
            f = _som.update_hex if self.hexagonal else _som.update

            def update(decay):
                f(self.weights, x,
                  sigma / decay, learning_rate / decay)

        self.init_weights(x)
        for iteration in range(n_iterations):
            update(1 + iteration / (n_iterations / 2))
            if callback is not None and not callback(iteration / n_iterations):
                break

    def winner(self, row):
        winner = np.empty(2, dtype=np.int16)
        if sp.issparse(row):
            assert row.shape[0] == 1
            _som.get_winner_sparse(self.weights, self.ssum_weights,
                                   row.indices, row.data,
                                   winner, int(self.hexagonal))
        else:
            _som.get_winner(self.weights, row, winner, int(self.hexagonal))
        return tuple(winner)
