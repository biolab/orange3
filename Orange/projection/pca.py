import sklearn.decomposition as skl_decomposition

try:
    from orangecontrib.remote import aborted, save_state
except ImportError:
    def aborted():
        return False

    def save_state(_):
        pass

import Orange.data
from Orange.misc.wrapper_meta import WrapperMeta
from Orange.preprocess import Continuize
from Orange.projection import SklProjector, Projection

__all__ = ["PCA", "SparsePCA", "RandomizedPCA", "IncrementalPCA"]


class PCA(SklProjector):
    __wraps__ = skl_decomposition.PCA
    name = 'pca'

    def __init__(self, n_components=None, copy=True, whiten=False, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y=None):
        proj = self.__wraps__(**self.params)
        proj = proj.fit(X, Y)
        return PCAModel(proj, self.domain)


class SparsePCA(SklProjector):
    __wraps__ = skl_decomposition.SparsePCA
    name = 'sparse pca'

    def __init__(self, n_components=None, alpha=1, ridge_alpha=0.01,
                 max_iter=1000, tol=1e-8, method='lars', n_jobs=1, U_init=None,
                 V_init=None, verbose=False, random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y=None):
        proj = self.__wraps__(**self.params)
        proj = proj.fit(X, Y)
        return PCAModel(proj, self.domain)


class RandomizedPCA(SklProjector):
    __wraps__ = skl_decomposition.RandomizedPCA
    name = 'randomized pca'

    def __init__(self, n_components=None, copy=True, iterated_power=3,
                 whiten=False, random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y=None):
        proj = self.__wraps__(**self.params)
        proj = proj.fit(X, Y)
        return PCAModel(proj, self.domain)


class _LinearCombination:
    def __init__(self, attrs, weights, mean=None):
        self.attrs = attrs
        self.weights = weights
        self.mean = mean

    def __call__(self):
        if self.mean is None:
            return ' + '.join('{} * {}'.format(w, a.to_sql())
                              for a, w in zip(self.attrs, self.weights))
        return ' + '.join('{} * ({} - {})'.format(w, a.to_sql(), m, w)
            for a, m, w in zip(self.attrs, self.mean, self.weights))


class PCAModel(Projection, metaclass=WrapperMeta):
    def __init__(self, proj, domain):
        def pca_variable(i):
            v = Orange.data.ContinuousVariable(
                'PC%d' % (i + 1), compute_value=Projector(self, i))
            v.to_sql = _LinearCombination(
                domain.attributes, self.components_[i, :],
                getattr(self, 'mean_', None))
            return v

        super().__init__(proj=proj)
        self.orig_domain = domain
        self.n_components = self.components_.shape[0]
        self.domain = Orange.data.Domain(
            [pca_variable(i) for i in range(self.n_components)],
             domain.class_vars, domain.metas)


class IncrementalPCA(SklProjector):
    __wraps__ = skl_decomposition.IncrementalPCA
    name = 'incremental pca'

    def __init__(self, n_components=None, whiten=False, copy=True,
                 batch_size=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y=None):
        proj = self.__wraps__(**self.params)
        proj = proj.fit(X, Y)
        return IncrementalPCAModel(proj, self.domain)

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


class Projector:
    def __init__(self, projection, feature):
        self.projection = projection
        self.feature = feature
        self.transformed = None

    def __call__(self, data):
        if data.domain != self.projection.pre_domain:
            data = data.from_table(self.projection.pre_domain, data)
        self.transformed = self.projection.transform(data.X)
        return self.transformed[:, self.feature]

    def __getstate__(self):
        d = dict(self.__dict__)
        d['transformed'] = None
        return d


class RemotePCA:
    def __new__(cls, data, batch=100, max_iter=100):
        cont = Continuize(multinomial_treatment=Continuize.Remove)
        data = cont(data)
        model = Orange.projection.IncrementalPCA()
        percent = batch / data.approx_len() * 100
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
