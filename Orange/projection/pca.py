import sklearn.decomposition as skl_decomposition

from Orange.projection import SklProjection

__all__ = ["PCA", "SparsePCA", "RandomizedPCA"]


class PCA(SklProjection):

    __wraps__ = skl_decomposition.PCA

    def __init__(self, n_components=None, whiten=False, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class SparsePCA(SklProjection):

    __wraps__ = skl_decomposition.SparsePCA

    def __init__(self, n_components=None, alpha=1, ridge_alpha=0.01,
                 max_iter=1000, tol=1e-8, method='lars', n_jobs=1, U_init=None,
                 V_init=None, random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class RandomizedPCA(SklProjection):

    __wraps__ = skl_decomposition.RandomizedPCA

    def __init__(self, n_components=None, iterated_power=3, whiten=False,
                 random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
