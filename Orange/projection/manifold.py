import sklearn.manifold as skl_manifold

from Orange.distance import SklDistance, SpearmanDistance, PearsonDistance
from Orange.projection import SklProjector

__all__ = ["MDS", "Isomap", "LocallyLinearEmbedding"]


class MDS(SklProjector):
    __wraps__ = skl_manifold.MDS
    name = 'mds'

    def __init__(self, n_components=2, metric=True, n_init=4, max_iter=300,
                 eps=0.001, n_jobs=1, random_state=None,
                 dissimilarity='euclidean',
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
        self._metric = dissimilarity

    def __call__(self, data):
        distances = SklDistance, SpearmanDistance, PearsonDistance
        if isinstance(self._metric, distances):
            data = self.preprocess(data)
            X, Y, domain = data.X, data.Y, data.domain
            dist_matrix = self._metric(X).X
            self.params['dissimilarity'] = 'precomputed'
            clf = self.fit(dist_matrix, Y=Y)
        elif self._metric is 'precomputed':
            dist_matrix, Y, domain = data.X, None, None
            clf = self.fit(dist_matrix, Y=Y)
        else:
            data = self.preprocess(data)
            X, Y, domain = data.X, data.Y, data.domain
            clf = self.fit(X, Y=Y)
        clf.domain = domain
        return clf

    def fit(self, X, init=None, Y=None):
        proj = self.__wraps__(**self.params)
        return proj.fit(X, init=init, y=Y)


class Isomap(SklProjector):
    __wraps__ = skl_manifold.Isomap
    name = 'isomap'

    def __init__(self, n_neighbors=5, n_components=2, eigen_solver='auto',
                 max_iter=None, path_method='auto',
                 neighbors_algorithm='auto', preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class LocallyLinearEmbedding(SklProjector):
    __wraps__ = skl_manifold.LocallyLinearEmbedding
    name = 'lle'

    def __init__(self, n_neighbors=5, n_components=2, reg=0.001,
                 eigen_solver='auto', tol=1e-06 , max_iter=100,
                 method='standard', hessian_tol=0.0001,
                 modified_tol=1e-12, neighbors_algorithm='auto',
                 random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
