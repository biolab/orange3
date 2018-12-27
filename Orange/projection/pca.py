import numpy as np
import sklearn.decomposition as skl_decomposition

try:
    from orangecontrib.remote import aborted, save_state
except ImportError:
    def aborted():
        return False

    def save_state(_):
        pass

import Orange.data
from Orange.data import Variable
from Orange.data.util import get_unique_names
from Orange.misc.wrapper_meta import WrapperMeta
from Orange.preprocess import Continuize
from Orange.preprocess.score import LearnerScorer
from Orange.projection import SklProjector, DomainProjection

__all__ = ["PCA", "SparsePCA", "IncrementalPCA", "TruncatedSVD"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    component = 0

    def score(self, data):
        model = self(data)
        return np.abs(model.components_[:self.component]) \
            if self.component else np.abs(model.components_)


class PCA(SklProjector, _FeatureScorerMixin):
    __wraps__ = skl_decomposition.PCA
    name = 'PCA'
    supports_sparse = False

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
        domain = self.orig_domain.variables + self.orig_domain.metas
        return get_unique_names([v.name for v in domain], names)


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


class RemotePCA:
    def __new__(cls, data, batch=100, max_iter=100):
        cont = Continuize(multinomial_treatment=Continuize.Remove)
        data = cont(data)
        model = Orange.projection.IncrementalPCA()
        n = data.approx_len()
        percent = batch / n * 100 if n else 100
        for i in range(max_iter):
            data_sample = data.sample_percentage(percent, no_cache=True)
            if not data_sample:
                continue
            data_sample.download_data(1000000)
            data_sample = Orange.data.Table.from_numpy(
                Orange.data.Domain(data_sample.domain.attributes),
                data_sample.X)
            model = model.partial_fit(data_sample)
            model.iteration = i
            save_state(model)
            if aborted() or data_sample is data:
                break
        return model
