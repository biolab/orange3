import warnings

import numpy as np

from scipy.sparse.linalg import eigsh as arpack_eigh
from scipy.linalg import eigh as lapack_eigh

import sklearn.manifold as skl_manifold

from Orange.distance import Distance, DistanceModel, Euclidean
from Orange.projection import SklProjector

__all__ = ["MDS", "Isomap", "LocallyLinearEmbedding", "SpectralEmbedding",
           "TSNE"]


def torgerson(distances, n_components=2, eigen_solver="auto"):
    """
    Perform classical mds (Torgerson-Gower scaling).

    ..note ::
        If the distances are euclidean then this is equivalent to projecting
        the original data points to the first `n` principal components.

    Parameters
    ----------
    distances : (N, N) ndarray
        Input distance (dissimilarity) matrix.
    n_components : int
        Number of components to return
    eigen_solver : str
        One of `lapack`, `arpack` or `'auto'`. The later chooses between the
        former based on the input.

    See Also
    --------
    `cmdscale` in R
    """
    distances = np.asarray(distances)
    assert distances.shape[0] == distances.shape[1]
    N = distances.shape[0]
    # O ^ 2
    D_sq = distances ** 2

    # double center the D_sq
    rsum = np.sum(D_sq, axis=1, keepdims=True)
    csum = np.sum(D_sq, axis=0, keepdims=True)
    total = np.sum(csum)
    D_sq -= rsum / N
    D_sq -= csum / N
    D_sq += total / (N ** 2)

    B = np.multiply(D_sq, -0.5, out=D_sq)

    if eigen_solver == 'auto':
        if N > 200 and n_components < 10:  # arbitrary - follow skl KernelPCA
            eigen_solver = 'arpack'
        else:
            eigen_solver = 'lapack'

    if eigen_solver == "arpack":
        v0 = np.random.RandomState(0xD06).uniform(-1, 1, B.shape[0])
        w, v = arpack_eigh(B, k=n_components, v0=v0)
        assert np.all(np.diff(w) >= 0), "w was not in ascending order"
        U, L = v[:, ::-1], w[::-1]
    elif eigen_solver == "lapack":  # lapack (d|s)syevr
        w, v = lapack_eigh(B, overwrite_a=True,
                           eigvals=(max(N - n_components, 0), N - 1))
        assert np.all(np.diff(w) >= 0), "w was not in ascending order"
        U, L = v[:, ::-1], w[::-1]
    else:
        raise ValueError(eigen_solver)

    assert L.shape == (min(n_components, N),)
    assert U.shape == (N, min(n_components, N))

    # Warn for (sufficiently) negative eig values ...
    neg = L < -5 * np.finfo(L.dtype).eps
    if np.any(neg):
        warnings.warn(
            ("{} of the {} eigenvalues were negative."
             .format(np.sum(neg), L.size)),
            UserWarning, stacklevel=2,
        )
    # ... and clamp them all to 0
    L[L < 0] = 0

    if n_components > N:
        U = np.hstack((U, np.zeros((N, n_components - N))))
        L = np.hstack((L, np.zeros((n_components - N))))
    U = U[:, :n_components]
    L = L[:n_components]
    return U * np.sqrt(L.reshape((1, n_components)))


class MDS(SklProjector):
    __wraps__ = skl_manifold.MDS
    name = 'MDS'

    def __init__(self, n_components=2, metric=True, n_init=4, max_iter=300,
                 eps=0.001, n_jobs=1, random_state=None,
                 dissimilarity='euclidean', init_type="random", init_data=None,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
        self._metric = dissimilarity
        self.init_type = init_type
        self.init_data = init_data

    def __call__(self, data):
        params = self.params.copy()
        dissimilarity = params['dissimilarity']
        if isinstance(self._metric, DistanceModel) \
                or (isinstance(self._metric, type)
                    and issubclass(self._metric, Distance)):
            data = self.preprocess(data)
            _X, Y, domain = data.X, data.Y, data.domain
            X = dist_matrix = self._metric(_X)
            dissimilarity = 'precomputed'
        elif self._metric is 'precomputed':
            dist_matrix, Y, domain = data, None, None
            X = dist_matrix
            dissimilarity = 'precomputed'
        else:
            data = self.preprocess(data)
            X, Y, domain = data.X, data.Y, data.domain
            if self.init_type == "PCA":
                dist_matrix = Euclidean(X)

        if self.init_type == "PCA" and self.init_data is None:
            init_data = torgerson(dist_matrix, params['n_components'])
        elif self.init_data is not None:
            init_data = self.init_data
        else:
            init_data = None

        params["dissimilarity"] = dissimilarity
        mds = self.__wraps__(**params)
        mds.fit(X, y=Y, init=init_data)
        mds.domain = domain
        return mds


class Isomap(SklProjector):
    __wraps__ = skl_manifold.Isomap
    name = 'Isomap'

    def __init__(self, n_neighbors=5, n_components=2, eigen_solver='auto',
                 tol=0, max_iter=None, path_method='auto',
                 neighbors_algorithm='auto', n_jobs=1,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class LocallyLinearEmbedding(SklProjector):
    __wraps__ = skl_manifold.LocallyLinearEmbedding
    name = 'Locally Linear Embedding'

    def __init__(self, n_neighbors=5, n_components=2, reg=0.001,
                 eigen_solver='auto', tol=1e-06, max_iter=100,
                 method='standard', hessian_tol=0.0001,
                 modified_tol=1e-12, neighbors_algorithm='auto',
                 random_state=None, n_jobs=1,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class SpectralEmbedding(SklProjector):
    __wraps__ = skl_manifold.SpectralEmbedding
    name = 'Spectral Embedding'

    def __init__(self, n_components=2, affinity='nearest_neighbors', gamma=None,
                 random_state=None, eigen_solver=None, n_neighbors=None, n_jobs=1,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class TSNE(SklProjector):
    __wraps__ = skl_manifold.TSNE
    name = 't-SNE'

    def __init__(self, n_components=2, perplexity=30.0, early_exaggeration=4.0,
                 learning_rate=1000.0, n_iter=1000, n_iter_without_progress=30,
                 min_grad_norm=1e-07, metric='euclidean', init='random',
                 random_state=None, method='barnes_hut', angle=0.5, n_jobs=1,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def __call__(self, data):
        params = self.params.copy()
        metric = params["metric"]
        if metric == 'precomputed':
            X, Y, domain = data, None, None
        else:
            data = self.preprocess(data)
            X, Y, domain = data.X, data.Y, data.domain
            if isinstance(metric, Distance):
                X = metric(X)
                params['metric'] = 'precomputed'

        tsne = self.__wraps__(**params)
        tsne.fit(X, y=Y)
        tsne.domain = domain
        return tsne
