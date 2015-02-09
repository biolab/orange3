import sklearn.manifold as skl_manifold

import Orange
from Orange.projection import SklProjection

__all__ = ["MDS", "Isomap", "LocallyLinearEmbedding"]


class MDS(SklProjection):

    __wraps__ = skl_manifold.MDS

    def __init__(self, n_components=2, metric=True, n_init=4, max_iter=300,
                 eps=0.001, n_jobs=1, random_state=None,
                 dissimilarity=Orange.distance.Euclidean,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
        self._metric = dissimilarity

    def __call__(self, data):
        if self._metric is not 'precomputed':
            data = self.preprocess(data)
            self.domain = data.domain
            X, Y, domain = data.X, data.Y, data.domain
            dist_matrix = self._metric(X).X
        else:
            dist_matrix, Y, domain = data.X, None, None
        self.params['dissimilarity'] = 'precomputed'
        clf = self.fit(dist_matrix, Y)
        clf.domain = domain
        return clf

    def fit(self, X, Y=None):
        proj = self.__wraps__(**self.params)
        return proj.fit(X, y=Y)


class Isomap(SklProjection):

    __wraps__ = skl_manifold.Isomap

    def __init__(self, n_neighbors=5, n_components=2, eigen_solver='auto',
                 max_iter=None, path_method='auto',
                 neighbors_algorithm='auto', preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class LocallyLinearEmbedding(SklProjection):

    __wraps__ = skl_manifold.LocallyLinearEmbedding

    def __init__(self, n_neighbors=5, n_components=2, reg=0.001,
                 eigen_solver='auto', tol=1e-06 , max_iter=100,
                 method='standard', hessian_tol=0.0001,
                 modified_tol=1e-12, neighbors_algorithm='auto',
                 random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()