import numpy as np
import scipy.sparse as sp

from Orange.projection import _som


class SOM:
    def __init__(self, dim_x, dim_y,
                 hexagonal=False, pca_init=True, random_seed=None):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.weights = self.ssum_weights = None
        self.hexagonal = hexagonal
        self.pca_init = pca_init
        self.random_seed = random_seed

    def init_weights_random(self, x):
        random = (np.random if self.random_seed is None
                  else np.random.RandomState(self.random_seed))
        self.weights = random.rand(self.dim_y, self.dim_x, x.shape[1])
        norms = np.sum(self.weights ** 2, axis=2)
        self.weights /= norms[:, :, None]
        self.ssum_weights = np.ones((self.dim_y, self.dim_x))

    def init_weights_pca(self, x):
        pc_length, pc = np.linalg.eig(np.cov(x.T))
        c0, c1, *_ = np.argsort(pc_length)
        pc0, pc1 = pc[c0], pc[c1]
        self.weights = np.empty((self.dim_y, self.dim_x, x.shape[1]))
        for i, c1 in enumerate(np.linspace(-1, 1, self.dim_y)):
            for j, c2 in enumerate(np.linspace(-1, 1, self.dim_x)):
                self.weights[i, j] = c1 * pc0 + c2 * pc1
        norms = np.sum(self.weights ** 2, axis=2)
        self.weights /= norms[:, :, None]
        self.ssum_weights = np.ones((self.dim_y, self.dim_x))

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

        if self.pca_init and not sp.issparse(x):
            self.init_weights_pca(x)
        else:
            self.init_weights_random(x)

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
