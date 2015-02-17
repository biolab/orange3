import sklearn.decomposition as skl_decomposition

import Orange.data
from Orange.misc.wrapper_meta import WrapperMeta
from Orange.projection import SklProjection, ProjectionModel

__all__ = ["PCA", "SparsePCA", "RandomizedPCA"]


class PCA(SklProjection):
    __wraps__ = skl_decomposition.PCA
    name = 'pca'

    def __init__(self, n_components=None, copy=True, whiten=False, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y=None):
        proj = self.__wraps__(**self.params)
        proj = proj.fit(X, Y)
        return PCAModel(proj, self.preprocessors)


class SparsePCA(SklProjection):
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
        return PCAModel(proj, self.preprocessors)


class RandomizedPCA(SklProjection):
    __wraps__ = skl_decomposition.RandomizedPCA
    name = 'randomized pca'

    def __init__(self, n_components=None, copy=True, iterated_power=3,
                 whiten=False, random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y=None):
        proj = self.__wraps__(**self.params)
        proj = proj.fit(X, Y)
        return PCAModel(proj, self.preprocessors)


class PCAModel(ProjectionModel, metaclass=WrapperMeta):
    def __init__(self, proj, preprocessors=None):
        super().__init__(proj=proj, preprocessors=preprocessors)
        self.n_components = self.components_.shape[0]

    def __call__(self, data):
        data = self.preprocess(data)
        Xt = self.transform(data.X)

        def pca_variable(i):
            v = Orange.data.ContinuousVariable('PC %d' % i)
            v.compute_value = Projector(self, i)
            return v

        domain = Orange.data.Domain(
            [pca_variable(i) for i in range(self.n_components)],
            data.domain.class_vars, data.domain.metas)
        transformed = Orange.data.Table.from_numpy(
                domain, Xt, Y=data.Y, metas=data.metas)
        return transformed


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
