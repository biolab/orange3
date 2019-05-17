import numbers
import six
import numpy as np
import scipy.sparse as sp
from scipy.linalg import lu, qr, svd

from sklearn import decomposition as skl_decomposition
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import svd_flip, safe_sparse_dot
from sklearn.utils.validation import check_is_fitted

import Orange.data
from Orange.statistics import util as ut
from Orange.data import Variable
from Orange.data.util import get_unique_names
from Orange.misc.wrapper_meta import WrapperMeta
from Orange.preprocess.score import LearnerScorer
from Orange.projection import SklProjector, DomainProjection

__all__ = ["PCA", "SparsePCA", "IncrementalPCA", "TruncatedSVD"]


def randomized_pca(A, n_components, n_oversamples=10, n_iter="auto",
                   flip_sign=True, random_state=0):
    """Compute the randomized PCA decomposition of a given matrix.

    This method differs from the scikit-learn implementation in that it supports
    and handles sparse matrices well.

    """
    if n_iter == "auto":
        # Checks if the number of iterations is explicitly specified
        # Adjust n_iter. 7 was found a good compromise for PCA. See sklearn #5299
        n_iter = 7 if n_components < .1 * min(A.shape) else 4

    n_samples, n_features = A.shape

    c = np.atleast_2d(ut.nanmean(A, axis=0))

    if n_samples >= n_features:
        Q = random_state.normal(size=(n_features, n_components + n_oversamples))
        if A.dtype.kind == "f":
            Q = Q.astype(A.dtype, copy=False)

        Q = safe_sparse_dot(A, Q) - safe_sparse_dot(c, Q)

        # Normalized power iterations
        for _ in range(n_iter):
            Q = safe_sparse_dot(A.T, Q) - safe_sparse_dot(c.T, Q.sum(axis=0)[None, :])
            Q, _ = lu(Q, permute_l=True)
            Q = safe_sparse_dot(A, Q) - safe_sparse_dot(c, Q)
            Q, _ = lu(Q, permute_l=True)

        Q, _ = qr(Q, mode="economic")

        QA = safe_sparse_dot(A.T, Q) - safe_sparse_dot(c.T, Q.sum(axis=0)[None, :])
        R, s, V = svd(QA.T, full_matrices=False)
        U = Q.dot(R)

    else:  # n_features > n_samples
        Q = random_state.normal(size=(n_samples, n_components + n_oversamples))
        if A.dtype.kind == "f":
            Q = Q.astype(A.dtype, copy=False)

        Q = safe_sparse_dot(A.T, Q) - safe_sparse_dot(c.T, Q.sum(axis=0)[None, :])

        # Normalized power iterations
        for _ in range(n_iter):
            Q = safe_sparse_dot(A, Q) - safe_sparse_dot(c, Q)
            Q, _ = lu(Q, permute_l=True)
            Q = safe_sparse_dot(A.T, Q) - safe_sparse_dot(c.T, Q.sum(axis=0)[None, :])
            Q, _ = lu(Q, permute_l=True)

        Q, _ = qr(Q, mode="economic")

        QA = safe_sparse_dot(A, Q) - safe_sparse_dot(c, Q)
        U, s, R = svd(QA, full_matrices=False)
        V = R.dot(Q.T)

    if flip_sign:
        U, V = svd_flip(U, V)

    return U[:, :n_components], s[:n_components], V[:n_components, :]


class ImprovedPCA(skl_decomposition.PCA):
    """Patch sklearn PCA learner to include randomized PCA for sparse matrices.

    Scikit-learn does not currently support sparse matrices at all, even though
    efficient methods exist for PCA. This class patches the default scikit-learn
    implementation to properly handle sparse matrices.

    Notes
    -----
      - This should be removed once scikit-learn releases a version which
        implements this functionality.

    """
    # pylint: disable=too-many-branches
    def _fit(self, X):
        """Dispatch to the right submethod depending on the chosen solver."""
        X = check_array(
            X,
            accept_sparse=["csr", "csc"],
            dtype=[np.float64, np.float32],
            ensure_2d=True,
            copy=self.copy,
        )

        # Handle n_components==None
        if self.n_components is None:
            if self.svd_solver != "arpack":
                n_components = min(X.shape)
            else:
                n_components = min(X.shape) - 1
        else:
            n_components = self.n_components

        # Handle svd_solver
        self._fit_svd_solver = self.svd_solver
        if self._fit_svd_solver == "auto":
            # Sparse data can only be handled with the randomized solver
            if sp.issparse(X):
                self._fit_svd_solver = "randomized"
            # Small problem or n_components == 'mle', just call full PCA
            elif max(X.shape) <= 500 or n_components == "mle":
                self._fit_svd_solver = "full"
            elif 1 <= n_components < .8 * min(X.shape):
                self._fit_svd_solver = "randomized"
            # This is also the case of n_components in (0,1)
            else:
                self._fit_svd_solver = "full"

        # Ensure we don't try call arpack or full on a sparse matrix
        if sp.issparse(X) and self._fit_svd_solver != "randomized":
            raise ValueError("only the randomized solver supports sparse matrices")

        # Call different fits for either full or truncated SVD
        if self._fit_svd_solver == "full":
            return self._fit_full(X, n_components)
        elif self._fit_svd_solver in ["arpack", "randomized"]:
            return self._fit_truncated(X, n_components, self._fit_svd_solver)
        else:
            raise ValueError(
                "Unrecognized svd_solver='{0}'".format(self._fit_svd_solver)
            )

    def _fit_truncated(self, X, n_components, svd_solver):
        """Fit the model by computing truncated SVD (by ARPACK or randomized) on X"""
        n_samples, n_features = X.shape

        if isinstance(n_components, six.string_types):
            raise ValueError(
                "n_components=%r cannot be a string with svd_solver='%s'" %
                (n_components, svd_solver)
            )
        if not 1 <= n_components <= min(n_samples, n_features):
            raise ValueError(
                "n_components=%r must be between 1 and min(n_samples, "
                "n_features)=%r with svd_solver='%s'" % (
                    n_components, min(n_samples, n_features), svd_solver
                )
            )
        if not isinstance(n_components, (numbers.Integral, np.integer)):
            raise ValueError(
                "n_components=%r must be of type int when greater than or "
                "equal to 1, was of type=%r" % (n_components, type(n_components))
            )
        if svd_solver == "arpack" and n_components == min(n_samples, n_features):
            raise ValueError(
                "n_components=%r must be strictly less than min(n_samples, "
                "n_features)=%r with svd_solver='%s'" % (
                    n_components, min(n_samples, n_features), svd_solver
                )
            )

        random_state = check_random_state(self.random_state)

        self.mean_ = X.mean(axis=0)
        total_var = ut.var(X, axis=0, ddof=1)

        if svd_solver == "arpack":
            # Center data
            X -= self.mean_
            # random init solution, as ARPACK does it internally
            v0 = random_state.uniform(-1, 1, size=min(X.shape))
            U, S, V = sp.linalg.svds(X, k=n_components, tol=self.tol, v0=v0)
            # svds doesn't abide by scipy.linalg.svd/randomized_svd
            # conventions, so reverse its outputs.
            S = S[::-1]
            # flip eigenvectors' sign to enforce deterministic output
            U, V = svd_flip(U[:, ::-1], V[::-1])

        elif svd_solver == "randomized":
            # sign flipping is done inside
            U, S, V = randomized_pca(
                X,
                n_components=n_components,
                n_iter=self.iterated_power,
                flip_sign=True,
                random_state=random_state,
            )

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = V
        self.n_components_ = n_components

        # Get variance explained by singular values
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var.sum()
        self.singular_values_ = S.copy()  # Store the singular values.

        if self.n_components_ < min(n_features, n_samples):
            self.noise_variance_ = (total_var.sum() - self.explained_variance_.sum())
            self.noise_variance_ /= min(n_features, n_samples) - n_components
        else:
            self.noise_variance_ = 0

        return U, S, V

    def transform(self, X):
        check_is_fitted(self, ["mean_", "components_"], all_or_any=all)

        X = check_array(
            X,
            accept_sparse=["csr", "csc"],
            dtype=[np.float64, np.float32],
            ensure_2d=True,
            copy=self.copy,
        )

        if self.mean_ is not None:
            X = X - self.mean_
        X_transformed = np.dot(X, self.components_.T)
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)
        return X_transformed


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    component = 0

    def score(self, data):
        model = self(data)
        return (
            np.abs(model.components_[:self.component]) if self.component
            else np.abs(model.components_),
            model.orig_domain.attributes)


class PCA(SklProjector, _FeatureScorerMixin):
    __wraps__ = ImprovedPCA
    name = 'PCA'
    supports_sparse = True

    def __init__(self, n_components=None, copy=True, whiten=False,
                 svd_solver='auto', tol=0.0, iterated_power='auto',
                 random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y=None):
        params = self.params.copy()
        if params["n_components"] is not None:
            params["n_components"] = min(min(X.shape), params["n_components"])
        proj = self.__wraps__(**params)
        proj = proj.fit(X, Y)
        return PCAModel(proj, self.domain, len(proj.components_))


class SparsePCA(SklProjector):
    __wraps__ = skl_decomposition.SparsePCA
    name = 'Sparse PCA'
    supports_sparse = False

    def __init__(self, n_components=None, alpha=1, ridge_alpha=0.01,
                 max_iter=1000, tol=1e-8, method='lars', n_jobs=1, U_init=None,
                 V_init=None, verbose=False, random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y=None):
        proj = self.__wraps__(**self.params)
        proj = proj.fit(X, Y)
        return PCAModel(proj, self.domain, len(proj.components_))


class PCAModel(DomainProjection, metaclass=WrapperMeta):
    var_prefix = "PC"

    def _get_var_names(self, n):
        names = [f"{self.var_prefix}{postfix}" for postfix in range(1, n + 1)]
        return get_unique_names(self.orig_domain, names)


class IncrementalPCA(SklProjector):
    __wraps__ = skl_decomposition.IncrementalPCA
    name = 'Incremental PCA'
    supports_sparse = False

    def __init__(self, n_components=None, whiten=False, copy=True,
                 batch_size=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y=None):
        proj = self.__wraps__(**self.params)
        proj = proj.fit(X, Y)
        return IncrementalPCAModel(proj, self.domain, len(proj.components_))

    def partial_fit(self, data):
        return self(data)


class IncrementalPCAModel(PCAModel):
    def partial_fit(self, data):
        if isinstance(data, Orange.data.Storage):
            if data.domain != self.pre_domain:
                data = data.from_table(self.pre_domain, data)
            self.proj.partial_fit(data.X)
        else:
            self.proj.partial_fit(data)
        self.__dict__.update(self.proj.__dict__)
        return self


class TruncatedSVD(SklProjector, _FeatureScorerMixin):
    __wraps__ = skl_decomposition.TruncatedSVD
    name = 'Truncated SVD'
    supports_sparse = True

    def __init__(self, n_components=2, algorithm='randomized', n_iter=5,
                 random_state=None, tol=0.0, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y=None):
        params = self.params.copy()
        # strict requirement in scikit fit_transform:
        # n_components must be < n_features
        params["n_components"] = min(min(X.shape)-1, params["n_components"])

        proj = self.__wraps__(**params)
        proj = proj.fit(X, Y)
        return PCAModel(proj, self.domain, len(proj.components_))
