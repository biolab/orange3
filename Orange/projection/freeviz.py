
import numpy as np
import scipy.spatial

from Orange.preprocess.preprocess import RemoveNaNRows, Continuize, Scale
from Orange.projection import LinearProjector, DomainProjection

__all__ = ["FreeViz"]


class FreeVizModel(DomainProjection):
    var_prefix = "freeviz"


class FreeViz(LinearProjector):
    name = 'FreeViz'
    supports_sparse = False
    preprocessors = [RemoveNaNRows(),
                     Continuize(multinomial_treatment=Continuize.FirstAsBase),
                     Scale(scale=Scale.Span)]
    projection = FreeVizModel

    def __init__(self, weights=None, center=True, scale=True, dim=2, p=1,
                 initial=None, maxiter=500, alpha=0.1,
                 atol=1e-5, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.weights = weights
        self.center = center
        self.scale = scale
        self.dim = dim
        self.p = p
        self.initial = initial
        self.maxiter = maxiter
        self.alpha = alpha
        self.atol = atol
        self.is_class_discrete = False
        self.components_ = None

    def __call__(self, data):
        if data is not None:
            self.is_class_discrete = data.domain.class_var.is_discrete
            if len([attr for attr in data.domain.attributes
                    if attr.is_discrete and len(attr.values) > 2]):
                raise ValueError("Can not handle discrete variables"
                                 " with more than two values")
        return super().__call__(data)

    def get_components(self, X, Y):
        return self.freeviz(
            X, Y, weights=self.weights, center=self.center, scale=self.scale,
            dim=self.dim, p=self.p, initial=self.initial,
            maxiter=self.maxiter, alpha=self.alpha, atol=self.atol,
            is_class_discrete=self.is_class_discrete)[1].T

    @classmethod
    def squareform(cls, d):
        """
        Parameters
        ----------
        d : (N * (N - 1) // 2, ) ndarray
            A hollow symmetric square array in condensed form

        Returns
        -------
        D : (N, N) ndarray
            A symmetric square array in redundant form.

        See also
        --------
        scipy.spatial.distance.squareform
        """
        assert d.ndim == 1
        return scipy.spatial.distance.squareform(d, checks=False)

    @classmethod
    def row_v(cls, a):
        """
        Return a view of `a` as a row vector.
        """
        return a.reshape((1, -1))

    @classmethod
    def col_v(cls, a):
        """
        Return a view of `a` as a column vector.
        """
        return a.reshape((-1, 1))

    @classmethod
    def allclose(cls, a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
        # same as np.allclose in numpy==1.10
        return np.all(np.isclose(a, b, rtol, atol, equal_nan=equal_nan))

    @classmethod
    def forces_regression(cls, distances, y, p=1):
        y = np.asarray(y)
        ydist = scipy.spatial.distance.pdist(y.reshape(-1, 1), "sqeuclidean")
        mask = distances > np.finfo(distances.dtype).eps * 100
        F = ydist
        if p == 1:
            F[mask] /= distances[mask]
        else:
            F[mask] /= distances[mask] ** p
        return F

    @classmethod
    def forces_classification(cls, distances, y, p=1):
        diffclass = scipy.spatial.distance.pdist(y.reshape(-1, 1), "hamming") != 0
        # handle attractive force
        if p == 1:
            F = -distances
        else:
            F = -(distances ** p)

        # handle repulsive force
        mask = (diffclass &
                (distances > np.finfo(distances.dtype).eps * 100))
        assert mask.shape == F.shape and mask.dtype == bool
        if p == 1:
            F[mask] = 1 / distances[mask]
        else:
            F[mask] = 1 / (distances[mask] ** p)
        return F

    @classmethod
    def gradient(cls, X, embeddings, forces, embedding_dist=None, weights=None):
        X = np.asarray(X)
        embeddings = np.asarray(embeddings)

        if weights is not None:
            weights = np.asarray(weights)
            if weights.ndim != 1:
                raise ValueError("weights.ndim != 1 ({})".format(weights.ndim))

        N, P = X.shape
        _, dim = embeddings.shape

        if not N == embeddings.shape[0]:
            raise ValueError("X and embeddings must have the same length ({}!={})"
                             .format(X.shape[0], embeddings.shape[0]))

        if weights is not None and X.shape[0] != weights.shape[0]:
            raise ValueError("X.shape[0] != weights.shape[0] ({}!={})"
                             .format(X.shape[0], weights.shape[0]))

        # all pairwise vector differences between embeddings
        embedding_diff = (embeddings[:, np.newaxis, :] -
                          embeddings[np.newaxis, :, :])
        assert embedding_diff.shape == (N, N, dim)
        assert cls.allclose(embedding_diff[0, 1], embeddings[0] - embeddings[1])
        assert cls.allclose(embedding_diff[1, 0], -embedding_diff[0, 1])

        # normalize the direction vectors to unit direction vectors
        if embedding_dist is not None:
            # use supplied precomputed distances
            diff_norm = cls.squareform(embedding_dist)
        else:
            diff_norm = np.linalg.norm(embedding_diff, axis=2)

        mask = diff_norm > np.finfo(diff_norm.dtype).eps * 100
        embedding_diff[mask] /= diff_norm[mask][:, np.newaxis]

        forces = cls.squareform(forces)

        if weights is not None:
            # multiply in the instance weights
            forces *= cls.row_v(weights)
            forces *= cls.col_v(weights)

        # multiply unit direction vectors with the force magnitude
        F = embedding_diff * forces[:, :, np.newaxis]
        assert F.shape == (N, N, dim)
        # sum all the forces acting on a particle
        F = np.sum(F, axis=0)
        assert F.shape == (N, dim)
        # Transfer forces to the 'anchors'
        # (P, dim) array of gradients
        G = X.T.dot(F)
        assert G.shape == (P, dim)
        return G

    @classmethod
    def freeviz_gradient(cls, X, y, embedding, p=1, weights=None, is_class_discrete=False):
        """
        Return the gradient for the FreeViz [1]_ projection.

        Parameters
        ----------
        X : (N, P) ndarray
            The data instance coordinates
        y : (N,) ndarray
            The instance target/class values
        embedding : (N, dim) ndarray
            The current FreeViz point embeddings.
        p : positive number
            The force 'power', e.g. if p=1 (default) the attractive/repulsive
            forces follow linear/inverse linear law, for p=2 the forces follow
            square/inverse square law, ...
        weights : (N, ) ndarray, optional
            Optional vector of sample weights.

        Returns
        -------
        G : (P, dim) ndarray
            The projection gradient.

        .. [1] Janez Demsar, Gregor Leban, Blaz Zupan
               FreeViz - An Intelligent Visualization Approach for Class-Labeled
               Multidimensional Data Sets, Proceedings of IDAMAP 2005, Edinburgh.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        embedding = np.asarray(embedding)
        assert X.ndim == 2 and X.shape[0] == y.shape[0] == embedding.shape[0]
        D = scipy.spatial.distance.pdist(embedding)
        if is_class_discrete:
            forces = cls.forces_classification(D, y, p=p)
        else:
            forces = cls.forces_regression(D, y, p=p)
        G = cls.gradient(X, embedding, forces, embedding_dist=D, weights=weights)
        return G

    @classmethod
    def _rotate(cls, A):
        """
        Rotate a 2D projection A so the first axis (row in A) is aligned with
        vector (1, 0).
        """
        assert A.ndim == 2 and A.shape[1] == 2
        phi = np.arctan2(A[0, 1], A[0, 0])
        R = [[np.cos(-phi), np.sin(-phi)],
             [-np.sin(-phi), np.cos(-phi)]]
        return np.dot(A, R)

    @classmethod
    def freeviz(cls, X, y, weights=None, center=True, scale=True, dim=2, p=1,
                initial=None, maxiter=500, alpha=0.1, atol=1e-5, is_class_discrete=False):
        """
        FreeViz

        Compute a linear lower dimensional projection to optimize separation
        between classes ([1]_).

        Parameters
        ----------
        X : (N, P) ndarray
            The input data instances
        y : (N, ) ndarray
            The instance class labels
        weights : (N, ) ndarray, optional
            Instance weights
        center : bool or (P,) ndarray
            If `True` then X will have mean subtracted out, if False no
            centering is performed. Alternatively can be a P vector to subtract
            from X.
        scale : bool or (P,) ndarray
            If `True` the X's column will be scaled by 1/SD, if False no scaling
            is performed. Alternatively can be a P vector to divide X by.
        dim : int
            The dimension of the projected points/embedding.
        p : positive number
            The force 'power', e.g. if p=1 (default) the attractive/repulsive
            forces follow linear/inverse linear law, for p=2 the forces follow
            square/inverse square law, ...
        initial : (P, dim) ndarray, optional
            Initial projection matrix
        maxiter : int
            Maximum number of iterations.
        alpha : float
            The step size ('learning rate')
        atol : float
            Terminating numerical tolerance (absolute).

        Returns
        -------
        embeddings : (N, dim) ndarray
            The point projections (`= X.dot(P)`)
        projection : (P, dim)
            The projection matrix.
        center : (P,) ndarray or None
            The translation applied to X (if any).
        scale : (P,) ndarray or None
            The scaling applied to X (if any).

        .. [1] Janez Demsar, Gregor Leban, Blaz Zupan
               FreeViz - An Intelligent Visualization Approach for Class-Labeled
               Multidimensional Data Sets, Proceedings of IDAMAP 2005, Edinburgh.
        """
        needcopy = center is not False or scale is not False
        X = np.array(X, copy=needcopy)
        y = np.asarray(y)
        N, P = X.shape
        _N, = y.shape
        if N != _N:
            raise ValueError("X and y must have the same length")

        if weights is not None:
            weights = np.asarray(weights)

        if isinstance(center, bool):
            if center:
                center = np.mean(X, axis=0)
            else:
                center = None
        else:
            center = np.asarray(center, dtype=X.dtype)
            if center.shape != (P, ):
                raise ValueError("center.shape != (X.shape[1], ) ({} != {})"
                                 .format(center.shape, (X.shape[1], )))

        if isinstance(scale, bool):
            if scale:
                scale = np.std(X, axis=0)
            else:
                scale = None
        else:
            scale = np.asarray(scale, dtype=X.dtype)
            if scale.shape != (P, ):
                raise ValueError("scale.shape != (X.shape[1],) ({} != {}))"
                                 .format(scale.shape, (P, )))

        if initial is not None:
            initial = np.asarray(initial)
            if initial.ndim != 2 or initial.shape != (P, dim):
                raise ValueError
        else:
            initial = cls.init_random(P, dim)
            # initial = np.random.random((P, dim)) * 2 - 1

        # Center/scale X if requested
        if center is not None:
            X -= center

        if scale is not None:
            scalenonzero = np.abs(scale) > np.finfo(scale.dtype).eps
            X[:, scalenonzero] /= scale[scalenonzero]

        A = initial
        embeddings = np.dot(X, A)

        step_i = 0
        while step_i < maxiter:
            G = cls.freeviz_gradient(X, y, embeddings, p=p, weights=weights,
                                     is_class_discrete=is_class_discrete)

            # Scale the changes (the largest anchor move is alpha * radius)
            with np.errstate(divide="ignore"):  # inf's will be ignored by min
                step = np.min(np.linalg.norm(A, axis=1)
                              / np.linalg.norm(G, axis=1))
                if not np.isfinite(step):
                    break
            step = alpha * step
            Anew = A - step * G

            # Center anchors (?? This does not seem right; it changes the
            # projection axes direction somewhat arbitrarily)
            Anew = Anew - np.mean(Anew, axis=0)

            # Scale (so that the largest radius is 1)
            maxr = np.max(np.linalg.norm(Anew, axis=1))
            if maxr >= 0.001:
                Anew /= maxr

            change = np.linalg.norm(Anew - A, axis=1)
            if cls.allclose(change, 0, atol=atol):
                break

            A = Anew
            embeddings = np.dot(X, A)
            step_i = step_i + 1

        if dim == 2:
            A = cls._rotate(A)

        return embeddings, A, center, scale

    @staticmethod
    def init_radial(p):
        """
        Return a 2D projection with a circular anchor placement.
        """
        assert p > 0
        if p == 1:
            axes_angle = [0]
        elif p == 2:
            axes_angle = [0, np.pi / 2]
        else:
            axes_angle = np.linspace(0, 2 * np.pi, p, endpoint=False)

        A = np.c_[np.cos(axes_angle), np.sin(axes_angle)]
        return A

    @staticmethod
    def init_random(p, dim, rstate=None):
        if not isinstance(rstate, np.random.RandomState):
            rstate = np.random.RandomState(rstate if rstate is not None else 0)
        return rstate.rand(p, dim) * 2 - 1
