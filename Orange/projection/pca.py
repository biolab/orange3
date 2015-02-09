import sklearn.decomposition as skl_decomposition

import Orange.data
from Orange.projection import SklProjection, ProjectionModel

__all__ = ["PCA", "SparsePCA", "RandomizedPCA"]


class PCA(SklProjection):

    __wraps__ = skl_decomposition.PCA

    def __init__(self, n_components=None, whiten=False, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y=None):
        proj = self.__wraps__(**self.params)
        proj = proj.fit(X, Y)
        return PCAModel(proj, self.preprocessors)


class SparsePCA(SklProjection):

    __wraps__ = skl_decomposition.SparsePCA

    def __init__(self, n_components=None, alpha=1, ridge_alpha=0.01,
                 max_iter=1000, tol=1e-8, method='lars', n_jobs=1, U_init=None,
                 V_init=None, random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y=None):
        proj = self.__wraps__(**self.params)
        proj = proj.fit(X, Y)
        return PCAModel(proj, self.preprocessors)


class RandomizedPCA(SklProjection):

    __wraps__ = skl_decomposition.RandomizedPCA

    def __init__(self, n_components=None, iterated_power=3, whiten=False,
                 random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y=None):
        proj = self.__wraps__(**self.params)
        proj = proj.fit(X, Y)
        return PCAModel(proj, self.preprocessors)


class PCAModel(ProjectionModel):
    def __init__(self, proj, preprocessors=None):
        super().__init__(proj=proj, preprocessors=preprocessors)
        self.n_components = self.components_.shape[0]

    def __call__(self, data):
        data = self.preprocess(data)
        X, Y = data.X, data.Y
        Xt = self.transform(X)
        feat = [Orange.data.ContinuousVariable('PC %d' % i)
                for i in range(self.n_components)]
        domain = Orange.data.Domain(feat, class_vars=data.domain.class_vars)
        return Orange.data.Table(domain, Xt, Y)
