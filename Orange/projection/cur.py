import numbers

import numpy as np
import scipy.sparse.linalg as sla

import Orange.data
from Orange.projection import Projection, ProjectionModel

__all__ = ["CUR"]


class CUR(Projection):
    def __init__(self, rank=3, max_error=1, random_state=None,
                 compute_U=False, preprocessors=None):
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
        self.lev_cols_, self.cols_ = self._select_columns(X, [U.T, s, V.T])
        self.lev_rows_, self.rows_ = self._select_columns(X.T, [V.T, s, U])
        self.C_ = X[:, self.cols_]
        self.R_ = X.T[:, self.rows_].T
        if self.compute_U:
            pinvC = np.linalg.pinv(self.C_)
            pinvR = np.linalg.pinv(self.R_)
            self.U_ = np.dot(np.dot(pinvC, X), pinvR)
        else:
            self.U_ = None
        return CURModel(self, self.preprocessors)

    def transform(self, X, axis):
        if axis == 0:
            return X[:, self.cols_]
        else:
            return X[self.rows_, :]

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


class CURModel(ProjectionModel):
    def __init__(self, proj, preprocessors=None):
        super().__init__(proj=proj, preprocessors=preprocessors)

    def __call__(self, data, axis=0):
        data = self.preprocess(data)
        Xt = self.proj.transform(data.X, axis)

        if axis == 0:
            def cur_variable(i):
                v = data.domain.variables[i]
                v.compute_value = Projector(self, i)
                return v

            domain = Orange.data.Domain(
                [cur_variable(org_idx) for org_idx in self.cols_],
                class_vars=data.domain.class_vars)
            transformed_data = Orange.data.Table(domain, Xt, data.Y)
        elif axis == 1:
            Y = data.Y[self.proj.rows_]
            transformed_data = Orange.data.Table(data.domain, Xt, Y)
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
