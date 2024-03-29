from typing import Union, Optional

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

    @staticmethod
    def prepare_data(x: Union[np.ndarray, sp.spmatrix],
                     offsets: Optional[np.ndarray] = None,
                     scales: Optional[np.ndarray] = None) \
            -> (Union[np.ndarray, sp.spmatrix],
                np.ndarray,
                Union[np.ndarray, None],
                Union[np.ndarray, None]):
        if sp.issparse(x) and offsets is not None:
            # This is used in compute_value, by any widget, hence there is no
            # way to prevent it or report an error. We go dense...
            x = x.todense()
        if sp.issparse(x):
            cont_x = x.tocsr()
            mask = np.ones(cont_x.shape[0], bool)
        else:
            mask = np.all(np.isfinite(x), axis=1)
            useful = np.sum(mask)
            if useful == 0:
                return x, mask, offsets, scales
            if useful == len(mask):
                cont_x = x.copy()
            else:
                cont_x = x[mask]
            if offsets is None:
                offsets = np.min(cont_x, axis=0)
            cont_x -= offsets[None, :]
            if scales is None:
                scales = np.max(cont_x, axis=0)
                scales[scales == 0] = 1
            cont_x /= scales[None, :]
        return cont_x, mask, offsets, scales

    def init_weights_random(self, x):
        random = (np.random if self.random_seed is None
                  else np.random.RandomState(self.random_seed))
        self.weights = random.rand(self.dim_y, self.dim_x, x.shape[1])
        norms = np.sum(self.weights ** 2, axis=2)
        norms[norms == 0] = 1
        self.weights /= norms[:, :, None]
        self.ssum_weights = np.ones((self.dim_y, self.dim_x))

    def init_weights_pca(self, x):
        pc_length, pc = np.linalg.eig(np.cov(x.T))
        c0, c1, *_ = np.argsort(pc_length)
        pc0, pc1 = np.real(pc[c0]), np.real(pc[c1])
        self.weights = np.empty((self.dim_y, self.dim_x, x.shape[1]))
        for i, c1 in enumerate(np.linspace(-1, 1, self.dim_y)):
            for j, c2 in enumerate(np.linspace(-1, 1, self.dim_x)):
                self.weights[i, j] = c1 * pc0 + c2 * pc1
        norms = np.sum(self.weights ** 2, axis=2)
        norms[norms == 0] = 1
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

        if self.pca_init and not sp.issparse(x) and x.shape[1] > 1:
            self.init_weights_pca(x)
        else:
            self.init_weights_random(x)

        for iteration in range(n_iterations):
            update(1 + iteration / (n_iterations / 2))
            if callback is not None and not callback(iteration / n_iterations):
                break

    def winners(self, x):
        return self.winner_from_weights(
            x, self.weights, self.ssum_weights, self.hexagonal)

    @staticmethod
    def winner_from_weights(x, weights, ssum_weights, hexagonal):
        if sp.issparse(x):
            return _som.get_winners_sparse(
                weights, ssum_weights, x, int(hexagonal))
        else:
            return _som.get_winners(weights, x, int(hexagonal))
