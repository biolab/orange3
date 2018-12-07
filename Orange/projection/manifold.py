import logging
import warnings

import numpy as np
import fastTSNE
import scipy.sparse as sp
from scipy.linalg import eigh as lapack_eigh
from scipy.sparse.linalg import eigsh as arpack_eigh
import sklearn.manifold as skl_manifold

import Orange
from Orange.data import Table, Domain, ContinuousVariable
from Orange.distance import Distance, DistanceModel, Euclidean
from Orange.projection import SklProjector, Projector, Projection
from Orange.projection.base import TransformDomain, ComputeValueProjector

__all__ = ["MDS", "Isomap", "LocallyLinearEmbedding", "SpectralEmbedding",
           "TSNE"]

# Disable t-SNE user warnings
fastTSNE.tsne.log.setLevel(logging.ERROR)
fastTSNE.affinity.log.setLevel(logging.ERROR)


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
        if isinstance(self._metric, DistanceModel) or (
                isinstance(self._metric, type)
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


class TSNEModel(Projection):
    """A tSNE embedding object. Supports further optimization as well as
    adding new data into the existing embedding.

    Attributes
    ----------
    embedding_ : fastTSNE.TSNEEmbedding
        The embedding object which takes care of subsequent optimizations of
        transforms.
    embedding : Table
        The embedding in an Orange table, easily accessible.
    pre_domain : Domain
        Original data domain
    """
    def __init__(self, embedding: fastTSNE.TSNEEmbedding, table: Table,
                 pre_domain: Domain):
        transformer = TransformDomain(self)

        def proj_variable(i):
            return self.embedding.domain[i].copy(
                compute_value=ComputeValueProjector(self, i, transformer))

        self.embedding_ = embedding
        self.embedding = table
        self.pre_domain = pre_domain
        self.domain = Domain(
            [proj_variable(attr) for attr in range(self.embedding.X.shape[1])],
            class_vars=table.domain.class_vars,
            metas=table.domain.metas)

    def transform(self, X: np.ndarray, **kwargs) -> fastTSNE.PartialTSNEEmbedding:
        if sp.issparse(X):
            raise TypeError(
                'A sparse matrix was passed, but dense data is required. Use '
                'X.toarray() to convert to a dense numpy array.'
            )
        return self.embedding_.transform(X, **kwargs)

    def __call__(self, data: Table, **kwargs) -> Table:
        # If we want to transform new data, ensure that we use correct domain
        if data.domain != self.pre_domain:
            data = data.transform(self.pre_domain)

        embedding = self.transform(data.X, **kwargs)
        return Table(self.domain, embedding.view(), data.Y, data.metas)

    def optimize(self, n_iter, inplace=False, propagate_exception=False, **kwargs):
        """Resume optimization for the current embedding."""
        kwargs = {'n_iter': n_iter, 'inplace': inplace,
                  'propagate_exception': propagate_exception, **kwargs}
        if inplace:
            self.embedding_.optimize(**kwargs)
            return self

        # If not inplace, we return a new TSNEModel object
        new_embedding = self.embedding_.optimize(**kwargs)
        table = Table(self.embedding.domain, new_embedding.view(np.ndarray),
                      self.embedding.Y, self.embedding.metas)
        return TSNEModel(new_embedding, table, self.pre_domain)


class TSNE(Projector):
    """t-distributed stochastic neighbor embedding (tSNE).

    Parameters
    ----------
    n_components : int
        The number of embedding that the embedding should contain. Note that
        only up to two dimensions are supported as otherwise the process can
        become prohibitively expensive.
    perplexity : float
        The desired perplexity of the probability distribution.
    learning_rate : float
        The learning rate for t-SNE. Typical values range from 1 to 1000.
        Setting the learning rate too high will result in the crowding problem
        where all the points form a ball in the center of the space.
    early_exaggeration_iter : int
        The number of iterations that the early exaggeration phase will be run
        for. Early exaggeration helps better separate clusters by increasing
        attractive forces between similar points.
    early_exaggeration : float
        The exaggeration term is used to increase the attractive forces during
        the first steps of the optimization. This enables points to move more
        easily through others, helping find their true neighbors quicker.
    n_iter : int
        The number of iterations to run the optimization after the early
        exaggeration phase.
    theta : float
        This is the trade-off parameter between speed and accuracy of the
        Barnes-Hut approximation of the negative forces. Setting a lower value
        will produce more accurate results, while setting a higher value will
        search through less of the space providing a rougher approximation.
        Scikit-learn recommends values between 0.2-0.8. This value is ignored
        unless the Barnes-Hut algorithm is used to compute negative gradients.
    min_num_intervals : int
        The minimum number of intervals into which we split our embedding. A
        larger value will produce better embeddings at the cost of performance.
        This value is ignored unless the interpolation based algorithm is used
        to compute negative gradients.
    ints_in_interval : float
        Since the coordinate range of the embedding will certainly change
        during optimization, this value tells us how many integer values should
        appear in a single interval. This number of intervals affect the
        embedding quality at the cost of performance. Less ints per interval
        will incur a larger number of intervals. This value is ignored unless
        the interpolation based algorithm is used to compute negative gradients.
    initialization : Optional[Union[np.ndarray, str]]
        An initial embedding strategy can be provided. A precomputed array with
        coordinates can be passed in, or optionally "random" or "pca"
        initializations are available. Note that while PCA can sometimes lead
        to faster convergence times, it can sometimes also lead to poor
        embeddings. Random initialization is typically a safe bet.
    metric : str
        The metric which will be used to evaluate the similarities between the
        input data points in the high dimensional space.
    n_jobs : int
        Parts of the algorithm can be in parallel and thus - faster.
    neighbors : str
        The method used to compute the nearest neighbors in the original, high
        dimensional data set. Possible values are "exact" or "approx" or any
        instance inheriting from `fastTSNE.nearest_neighbors.KNNIndex`. When
        dealing with larger data sets, approximate NN search is faster, when
        dealing with smaller data sets, exact NN search is typically faster.
    negative_gradient_method : str
        The method used to evaluate negative gradients (repulsive forces) in
        the embedding. Possible values are "bh" for Barnes-Hut or "fft" for
        Fast Fourier Accelerated Interpolation based tSNE or FItSNE for short.
        BH tends to be faster for smaller data sets but scales as O(n log n)
        while FItSNE is faster for larger data sets and scales linearly in the
        number of points.
    callbacks : Callable[[int, float, np.ndarray] -> bool]
        The callback should accept three parameters, the first is the current
        iteration, the second is the current KL divergence error and the last
        is the current embedding. The callback should return a boolean value
        indicating whether or not to stop optimization i.e. True to stop.
        This is convenient because returning `None` is falsey and helps avoid
        potential bugs if forgetting to return. Optionally, a list of callbacks
        is also supported.
    callbacks_every_iters : int
        How often should the callback be called.
    random_state: Optional[Union[int, RandomState]]
        The random state parameter follows the convention used in scikit-learn.
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.
    preprocessors

    """
    name = 't-SNE'
    preprocessors = [
        Orange.preprocess.Continuize(),
        Orange.preprocess.SklImpute(),
    ]

    def __init__(self, n_components=2, perplexity=30, learning_rate=200,
                 early_exaggeration_iter=250, early_exaggeration=12,
                 n_iter=750, exaggeration=None, theta=0.5, min_num_intervals=10,
                 ints_in_interval=1, initialization='random', metric='euclidean',
                 n_jobs=1, neighbors='exact', negative_gradient_method='bh',
                 callbacks=None, callbacks_every_iters=50,
                 random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.tsne = fastTSNE.TSNE(
            n_components=n_components, perplexity=perplexity,
            learning_rate=learning_rate, early_exaggeration=early_exaggeration,
            early_exaggeration_iter=early_exaggeration_iter, n_iter=n_iter,
            exaggeration=exaggeration, theta=theta, min_num_intervals=min_num_intervals,
            ints_in_interval=ints_in_interval, initialization=initialization,
            metric=metric, n_jobs=n_jobs, neighbors=neighbors,
            negative_gradient_method=negative_gradient_method,
            callbacks=callbacks, callbacks_every_iters=callbacks_every_iters,
            random_state=random_state
        )

    def fit(self, X: np.ndarray, Y: np.ndarray = None) -> fastTSNE.TSNEEmbedding:
        if sp.issparse(X):
            raise TypeError(
                'A sparse matrix was passed, but dense data is required. Use '
                'X.toarray() to convert to a dense numpy array.'
            )
        return self.tsne.fit(X)

    def __call__(self, data: Table) -> TSNEModel:
        # Preprocess the data - convert discrete to continuous
        data = self.preprocess(data)

        # Run tSNE optimization
        embedding = self.fit(data.X, data.Y)

        # The results should be accessible in an Orange table, which doesn't
        # need the full embedding attributes and is cast into a regular array
        n = self.tsne.n_components
        postfixes = ['x', 'y'] if n == 2 else list(range(1, n + 1))
        tsne_cols = [ContinuousVariable(f't-SNE-{p}') for p in postfixes]
        embedding_domain = Domain(tsne_cols, data.domain.class_vars, data.domain.metas)
        embedding_table = Table(embedding_domain, embedding.view(np.ndarray), data.Y, data.metas)

        # Create a model object which will be capable of transforming new data
        # into the existing embedding
        model = TSNEModel(embedding, embedding_table, data.domain)
        model.name = self.name

        return model
