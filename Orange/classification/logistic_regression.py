import numpy
from scipy import sparse
from sklearn import linear_model

from Orange import classification


def _np_replace_nan(A, value=0.0):
    """
    Replace NaN values in a numpy array `A` with `value`.
    """
    A = numpy.asanyarray(A)
    mask = numpy.isnan(A)
    return numpy.where(mask, value, A)


def _sp_replace_nan(A, value=0.0):
    """
    Replace NaN values in a sparse matrix `A` with `value.
    """
    A_csr = A.tocsr()

    # Ensure we have a new copy to modify at will
    if A_csr is A:
        A_csr = A_csr.copy()

    mask = numpy.isnan(A_csr.data)
    if mask.any():
        A_csr.data[mask] = value

        if value == 0.0:
            A_csr.eliminate_zeros()

    return A_csr


def _np_drop_nan(A, axis=0):
    """
    Drop rows or columns from `A` which contain NaNs.
    """
    # assert 0 <= axis <= 1
    if axis not in [0, 1]:
        raise ValueError("axis out of bounds")

    axis_mask = _np_contains_nan(A, axis)
    if axis == 0:
        return A[~axis_mask, :]
    else:
        return A[:, ~axis_mask]


def _sp_drop_nan(A, axis=0):
    """
    Drop rows or columns from `A` which contain NaNs.
    """
    if axis == 0:
        A_c = A.tocsr()
    elif axis == 1:
        A_c = A.tocsc()
    else:
        raise ValueError("axis out of bounds")

    if A_c is A:
        A_c = A_c.copy()

    axis_mask = _sp_contains_nan(A_c, axis)
    axis_indices = numpy.flatnonzero(~axis_mask)
    if axis == 0:
        return A_c[axis_indices, :]
    else:
        return A_c[:, axis_indices]


def _np_contains_nan(A, axis=0):
    if axis not in [0, 1]:
        raise ValueError("axis out of bounds")
    A = numpy.asarray(A)
    mask = numpy.isnan(A)
    return numpy.sum(mask, 1 - axis, dtype=bool).ravel()


def _sp_contains_nan(A, axis=0):
    if axis not in [0, 1]:
        raise ValueError("axis out of bounds")

    return numpy.isnan(numpy.asarray(A.sum(axis=1 - axis)).ravel())


def replace_nan(A, value=0.0):
    if sparse.issparse(A):
        return _sp_replace_nan(A, value=value)
    else:
        return _np_replace_nan(A, value=value)


def drop_nan(A, axis=0):
    if sparse.issparse(A):
        return _sp_drop_nan(A, axis)
    else:
        return _np_drop_nan(A, axis)


def contains_nan(A, axis=0):
    if sparse.issparse(A):
        return _sp_contains_nan(A, axis=axis)
    else:
        return _np_contains_nan(A, axis=axis)


class LogisticRegressionLearner(classification.Fitter):
    def __init__(self, penalty="l2", dual=False, tol=0.0001, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.random_state = random_state

    def fit(self, X, Y, W):
        X = replace_nan(X, value=0.0)

        lr = linear_model.LogisticRegression(
            penalty=self.penalty,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            random_state=self.random_state
        )
        clsf = lr.fit(X, Y.ravel())

        return LogisticRegressionClassifier(clsf)


class LogisticRegressionClassifier(classification.Model):
    def __init__(self, clf):
        self.clf = clf

    def predict(self, X):
        X = replace_nan(X, value=0.0)

        value = self.clf.predict(X)
        prob = self.clf.predict_proba(X)
        return value, prob
