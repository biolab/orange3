import sklearn.manifold as skl_manifold

import Orange
from Orange.projection import SklProjection

__all__ = ["MDS", "Isomap", "LocallyLinearEmbedding"]


class MDS(SklProjection):
    """Multidimensional scaling

    A wrapper for `sklearn.manifold.MDS`. The following is the documentation
    from `scikit-learn <http://scikit-learn.org>`_.

    Additional Orange parameters:

    preprocessors : list, optional (default="[]")
        An ordered list of preprocessors applied to data before
        training or testing.

    Parameters
    ----------
    metric : boolean, optional, default: True
        compute metric or nonmetric SMACOF (Scaling by Majorizing a
        Complicated Function) algorithm

    n_components : int, optional, default: 2
        number of dimension in which to immerse the similarities
        overridden if initial array is provided.

    n_init : int, optional, default: 4
        Number of time the smacof algorithm will be run with different
        initialisation. The final results will be the best output of the
        n_init consecutive runs in terms of stress.

    max_iter : int, optional, default: 300
        Maximum number of iterations of the SMACOF algorithm for a single run

    verbose : int, optional, default: 0
        level of verbosity

    eps : float, optional, default: 1e-6
        relative tolerance w.r.t stress to declare converge

    n_jobs : int, optional, default: 1
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    dissimilarity : `Orange.distance`, default: Orange.distance.Euclidean
        Which dissimilarity measure to use.
        Supported are distance from `Orange.distance` and 'precomputed'.


    Attributes
    ----------
    embedding_ : array-like, shape [n_components, n_samples]
        Stores the position of the dataset in the embedding space

    stress_ : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points)


    References
    ----------
    "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
    Groenen P. Springer Series in Statistics (1997)

    "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
    Psychometrika, 29 (1964)

    "Multidimensional scaling by optimizing goodness of fit to a nonmetric
    hypothesis" Kruskal, J. Psychometrika, 29, (1964)

    """

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
        clf = self.fit(dist_matrix, Y=Y)
        clf.domain = domain
        return clf

    def fit(self, X, init=None, Y=None):
        proj = self.__wraps__(**self.params)
        return proj.fit(X, init=init, y=Y)


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
