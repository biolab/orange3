import sklearn.manifold as skl_manifold

from Orange.distance import SklDistance, SpearmanDistance, PearsonDistance
from Orange.projection import SklProjector

__all__ = ["MDS", "Isomap", "LocallyLinearEmbedding", "SpectralEmbedding",
           "TSNE"]


class MDS(SklProjector):
    __wraps__ = skl_manifold.MDS
    name = 'MDS'

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
            dist_matrix = self._metric(X)
            self.params['dissimilarity'] = 'precomputed'
            clf = self.fit(dist_matrix, Y=Y)
        elif self._metric is 'precomputed':
            dist_matrix, Y, domain = data, None, None
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
                 tol=0, max_iter=None, path_method='auto',
                 neighbors_algorithm='auto', preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class LocallyLinearEmbedding(SklProjector):
    __wraps__ = skl_manifold.LocallyLinearEmbedding
    name = 'lle'

    def __init__(self, n_neighbors=5, n_components=2, reg=0.001,
                 eigen_solver='auto', tol=1e-06, max_iter=100,
                 method='standard', hessian_tol=0.0001,
                 modified_tol=1e-12, neighbors_algorithm='auto',
                 random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class SpectralEmbedding(SklProjector):
    __wraps__ = skl_manifold.SpectralEmbedding
    name = 'Spectral Embedding'

    def __init__(self, n_components=2, affinity='nearest_neighbors', gamma=None,
                 random_state=None, eigen_solver=None, n_neighbors=None,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class TSNE(SklProjector):
    __wraps__ = skl_manifold.TSNE
    name = 't-SNE'

    def __init__(self, n_components=2, perplexity=30.0, early_exaggeration=4.0,
                 learning_rate=1000.0, n_iter=1000, n_iter_without_progress=30,
                 min_grad_norm=1e-07, metric='euclidean', init='random',
                 random_state=None, method='barnes_hut', angle=0.5,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def __call__(self, data):
        if self.params['metric'] is 'precomputed':
            X, Y, domain = data, None, None
        else:
            data = self.preprocess(data)
            X, Y, domain = data.X, data.Y, data.domain
            distances = SklDistance, SpearmanDistance, PearsonDistance
            if isinstance(self.params['metric'], distances):
                X = self.params['metric'](X)
                self.params['metric'] = 'precomputed'
        clf = self.fit(X, Y=Y)
        clf.domain = domain
        return clf
