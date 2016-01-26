import numbers

import numpy as np
import scipy.sparse.linalg as sla

import Orange.data
from Orange.projection import Projector, Projection

__all__ = ["CUR"]


class CUR(Projector):
    """CUR matrix decomposition

    Parameters
    ----------
    rank : boolean, optional, default: True
        number of most significant right singular vectors considered
        for computing feature statistical leverage scores

    max_error : float, optional, default: 1
        relative error w.r.t. optimal `rank`-rank SVD approximation

    compute_U : boolean, optional, default: False
        Compute matrix U in the CUR decomposition or set it to None.

        If True matrix U is computed from C and R through Moore-Penrose
        generalized inverse as U = pinv(C) * X * pin(R).

    random_state : integer or numpy.RandomState, optional
        The generator used in importance sampling. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    preprocessors : list, optional (default="[]")
        An ordered list of preprocessors applied to data before
        training or testing.

    Attributes
    ----------
    lev_features_ : array-like, shape [n_features]
        Stores normalized statistical leverage scores of features

    features_ : array-like, shape [O(k log(k) / max_error^2)]
        Stores indices of features selected by the CUR algorithm

    lev_samples_ : array-like, shape [n_samples]
        Stores normalized statistical leverage scores of samples

    samples_ : array-like, shape [O(k log(k) / max_error^2)]
        Stores indices of samples selected by the CUR algorithm

    C_ : array-like, shape [n_samples, O(k log(k) / max_error^2)]
        Stores matrix C as defined in the CUR, a small number of
        columns from the data

    U_ : array-like, shape [O(k log(k) / max_error^2), O(k log(k) / max_error^2)]
        Stores matrix U as defined in the CUR

    R__ : array-like, shape [O(k log(k) / max_error^2), n_features]
        Stores matrix R as defined in the CUR, a small number of rows from the
        data

    References
    ----------
    "CUR matrix decompositions for improved data analysis" Mahoney, M.W;
    Drineas P. PNAS (2009)

    """

    name = 'cur'

    def __init__(self, rank=3, max_error=1, compute_U=False,
                 random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.rank = rank
        self.compute_U = compute_U
        self.max_error = max_error
        if isinstance(random_state, numbers.Integral):
            self.random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.mtrand._rand

    def fit(self, X, Y=None):
        U, s, V = sla.svds(X, self.rank)
        self.lev_features_, self.features_ = self._select_columns(X, [U.T, s, V.T])
        self.lev_samples_, self.samples_ = self._select_columns(X.T, [V.T, s, U])
        self.C_ = X[:, self.features_]
        self.R_ = X.T[:, self.samples_].T
        if self.compute_U:
            pinvC = np.linalg.pinv(self.C_)
            pinvR = np.linalg.pinv(self.R_)
            self.U_ = np.dot(np.dot(pinvC, X), pinvR)
        else:
            self.U_ = None
        return CURModel(self)

    def transform(self, X, axis):
        if axis == 0:
            return X[:, self.features_]
        else:
            return X[self.samples_, :]

    def _select_columns(self, X, UsV):
        U, s, V = UsV
        lev = self._score_leverage(V)
        c = self.rank * np.log(self.rank) / self.max_error**2
        prob = np.minimum(1., c * lev)
        rnd = self.random_state.rand(X.shape[1])
        cols = np.nonzero(rnd < prob)[0]
        return lev, cols

    def _score_leverage(self, V):
        return np.array(1. / self.rank * np.sum(np.power(V, 2), 1))


class CURModel(Projection):
    def __init__(self, proj):
        super().__init__(proj=proj)

    def __call__(self, data, axis=0):
        if data.domain is not self.domain:
            data = Orange.data.Table(self.domain, data)
        Xt = self.proj.transform(data.X, axis)

        if axis == 0:
            def cur_variable(i):
                var = data.domain[i]
                return var.copy(compute_value=Projector(self, i))

            domain = Orange.data.Domain(
                [cur_variable(org_idx) for org_idx in self.features_],
                class_vars=data.domain.class_vars)
            transformed_data = Orange.data.Table(domain, Xt, data.Y)
        elif axis == 1:
            Y = data.Y[self.proj.samples_]
            metas = data.metas[self.proj.samples_]
            transformed_data = Orange.data.Table(data.domain, Xt, Y, metas=metas)
        else:
            raise TypeError('CUR can select either columns '
                            '(axis = 0) or rows (axis = 1).')

        return transformed_data


class Projector:
    def __init__(self, projection, feature):
        self.projection = projection
        self.feature = feature
        self.transformed = None

    def __call__(self, data):
        if data is not self.transformed:
            self.transformed = self.projection.transform(data.X)
        return self.transformed[:, self.feature]

    def __getstate__(self):
        d = dict(self.__dict__)
        d['transformed'] = None
        return d


if __name__ == '__main__':
    import numpy as np
    import scipy.sparse.linalg as sla

    import Orange.data
    import Orange.projection

    np.random.seed(42)
    X = np.random.rand(60, 100)
    rank = 5
    max_error = 1

    data = Orange.data.Table(X)
    cur = Orange.projection.CUR(rank=rank, max_error=max_error, compute_U=True)
    cur_model = cur(data)

    transformed_data = cur_model(data, axis=0)
    np.testing.assert_array_equal(transformed_data.X, cur_model.C_)

    U, s, V = sla.svds(X, rank)
    S = np.diag(s)
    X_k = np.dot(U, np.dot(S, V))
    err_svd = np.linalg.norm(X - X_k, 'fro')
    print('Fro. error (optimal SVD): %5.4f' % err_svd)

    X_hat = np.dot(cur_model.C_, np.dot(cur_model.U_, cur_model.R_))
    err_cur = np.linalg.norm(X - X_hat, 'fro')
    print('Fro. error (CUR): %5.4f' % err_cur)
    # CUR guarantees with high probability err_cur <= (2+eps) err_svd
    assert err_cur < (3 + cur_model.max_error) * err_svd
