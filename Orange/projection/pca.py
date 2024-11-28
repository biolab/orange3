import numpy as np
import scipy.sparse as sp
from sklearn import decomposition as skl_decomposition

import Orange.data
from Orange.data import Variable
from Orange.data.util import get_unique_names
from Orange.misc.wrapper_meta import WrapperMeta
from Orange.preprocess.score import LearnerScorer
from Orange.projection import SklProjector, DomainProjection

__all__ = ["PCA", "SparsePCA", "IncrementalPCA", "TruncatedSVD"]


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
    __wraps__ = skl_decomposition.PCA
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

        # scikit-learn doesn't support requesting the same number of PCs as
        # there are columns when the data is sparse. In this case, densify the
        # data. Since we're essentially requesting back a PC matrix of the same
        # size as the original data, we will assume the matrix is small enough
        # to densify as well
        if sp.issparse(X) and params["n_components"] == min(X.shape):
            X = X.toarray()

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
        params["n_components"] = min(min(X.shape) - 1, params["n_components"])

        proj = self.__wraps__(**params)
        proj = proj.fit(X, Y)
        return PCAModel(proj, self.domain, len(proj.components_))
