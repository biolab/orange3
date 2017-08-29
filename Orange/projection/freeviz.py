
import numpy
import scipy.spatial


def squareform(d):
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


def row_v(a):
    """
    Return a view of `a` as a row vector.
    """
    return a.reshape((1, -1))


def col_v(a):
    """
    Return a view of `a` as a column vector.
    """
    return a.reshape((-1, 1))


def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    # same as numpy.allclose in numpy==1.10
    return numpy.all(numpy.isclose(a, b, rtol, atol, equal_nan=equal_nan))


def forces_regression(distances, y, p=1):
    y = numpy.asarray(y)
    ydist = scipy.spatial.distance.pdist(y.reshape(-1, 1), "sqeuclidean")
    mask = distances > numpy.finfo(distances.dtype).eps * 100
    F = ydist
    if p == 1:
        F[mask] /= distances[mask]
    else:
        F[mask] /= distances[mask] ** p
    return F


def forces_classification(distances, y, p=1):
    diffclass = scipy.spatial.distance.pdist(y.reshape(-1, 1), "hamming") != 0
    # handle attractive force
    if p == 1:
        F = -distances
    else:
        F = -(distances ** p)

    # handle repulsive force
    mask = (diffclass &
            (distances > numpy.finfo(distances.dtype).eps * 100))
    assert mask.shape == F.shape and mask.dtype == numpy.bool
    if p == 1:
        F[mask] = 1 / distances[mask]
    else:
        F[mask] = 1 / (distances[mask] ** p)
    return F


def gradient(X, embeddings, forces, embedding_dist=None, weights=None):
    X = numpy.asarray(X)
    embeddings = numpy.asarray(embeddings)

    if weights is not None:
        weights = numpy.asarray(weights)
        if weights.ndim != 1:
            raise ValueError("weights.ndim != 1 ({})".format(weights.ndim))

    N, P = X.shape
    _, dim = embeddings.shape

    if not N == embeddings.shape[0]:
        raise ValueError("X and embeddings must have the same length ({}!={})"
                         .format(X.shape[0] != embeddings.shape[0]))

    if weights is not None and X.shape[0] != weights.shape[0]:
        raise ValueError("X.shape[0] != weights.shape[0] ({}!={})"
                         .format(X.shape[0], weights.shape[0]))

    # all pairwise vector differences between embeddings
    embedding_diff = (embeddings[:, numpy.newaxis, :] -
                      embeddings[numpy.newaxis, :, :])
    assert embedding_diff.shape == (N, N, dim)
    assert allclose(embedding_diff[0, 1], embeddings[0] - embeddings[1])
    assert allclose(embedding_diff[1, 0], -embedding_diff[0, 1])

    # normalize the direction vectors to unit direction vectors
    if embedding_dist is not None:
        # use supplied precomputed distances
        diff_norm = squareform(embedding_dist)
    else:
        diff_norm = numpy.linalg.norm(embedding_diff, axis=2)

    mask = diff_norm > numpy.finfo(diff_norm.dtype).eps * 100
    embedding_diff[mask] /= diff_norm[mask][:, numpy.newaxis]

    forces = squareform(forces)

    if weights is not None:
        # multiply in the instance weights
        forces *= row_v(weights)
        forces *= col_v(weights)

    # multiply unit direction vectors with the force magnitude
    F = embedding_diff * forces[:, :, numpy.newaxis]
    assert F.shape == (N, N, dim)
    # sum all the forces acting on a particle
    F = numpy.sum(F, axis=0)
    assert F.shape == (N, dim)
    # Transfer forces to the 'anchors'
    # (P, dim) array of gradients
    G = X.T.dot(F)
    assert G.shape == (P, dim)
    return G


def freeviz_gradient(X, y, embedding, p=1, weights=None):
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
    X = numpy.asarray(X)
    y = numpy.asarray(y)
    embedding = numpy.asarray(embedding)
    assert X.ndim == 2 and X.shape[0] == y.shape[0] == embedding.shape[0]
    D = scipy.spatial.distance.pdist(embedding)
    if y.dtype.kind == "i":
        forces = forces_classification(D, y, p=p)
    elif y.dtype.kind == "f":
        forces = forces_regression(D, y, p=p)
    else:
        raise TypeError
    G = gradient(X, embedding, forces, embedding_dist=D, weights=weights)
    return G


def _rotate(A):
    """
    Rotate a 2D projection A so the first axis (row in A) is aligned with
    vector (1, 0).
    """
    assert A.ndim == 2 and A.shape[1] == 2
    phi = numpy.arctan2(A[0, 1], A[0, 0])
    R = [[numpy.cos(-phi), numpy.sin(-phi)],
         [-numpy.sin(-phi), numpy.cos(-phi)]]
    return numpy.dot(A, R)


def freeviz(X, y, weights=None, center=True, scale=True, dim=2, p=1,
            initial=None, maxiter=500, alpha=0.1, atol=1e-5):
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
    X = numpy.array(X, copy=needcopy)
    y = numpy.asarray(y)
    N, P = X.shape
    _N, = y.shape
    if N != _N:
        raise ValueError("X and y must have the same length")

    if weights is not None:
        weights = numpy.asarray(weights)

    if isinstance(center, bool):
        if center:
            center = numpy.mean(X, axis=0)
        else:
            center = None
    else:
        center = numpy.asarray(center, dtype=X.dtype)
        if center.shape != (P, ):
            raise ValueError("center.shape != (X.shape[1], ) ({} != {})"
                             .format(center.shape, (X.shape[1], )))

    if isinstance(scale, bool):
        if scale:
            scale = numpy.std(X, axis=0)
        else:
            scale = None
    else:
        scale = numpy.asarray(scale, dtype=X.dtype)
        if scale.shape != (P, ):
            raise ValueError("scale.shape != (X.shape[1],) ({} != {))"
                             .format(scale.shape, (P, )))

    if initial is not None:
        initial = numpy.asarray(initial)
        if initial.ndim != 2 or initial.shape != (P, dim):
            raise ValueError
    else:
        initial = init_random(P, dim)
        # initial = numpy.random.random((P, dim)) * 2 - 1

    # Center/scale X if requested
    if center is not None:
        X -= center

    if scale is not None:
        scalenonzero = numpy.abs(scale) > numpy.finfo(scale.dtype).eps
        X[:, scalenonzero] /= scale[scalenonzero]

    A = initial
    embeddings = numpy.dot(X, A)

    step_i = 0
    while step_i < maxiter:
        G = freeviz_gradient(X, y, embeddings, p=p, weights=weights)

        # Scale the changes (the largest anchor move is alpha * radius)
        step = numpy.min(numpy.linalg.norm(A, axis=1) /
                         numpy.linalg.norm(G, axis=1))
        step = alpha * step
        Anew = A - step * G

        # Center anchors (?? This does not seem right; it changes the
        # projection axes direction somewhat arbitrarily)
        Anew = Anew - numpy.mean(Anew, axis=0)

        # Scale (so that the largest radius is 1)
        maxr = numpy.max(numpy.linalg.norm(Anew, axis=1))
        if maxr >= 0.001:
            Anew /= maxr

        change = numpy.linalg.norm(Anew - A, axis=1)
        if allclose(change, 0, atol=atol):
            break

        A = Anew
        embeddings = numpy.dot(X, A)
        step_i = step_i + 1

    if dim == 2:
        A = _rotate(A)

    return embeddings, A, center, scale


def init_radial(p):
    """
    Return a 2D projection with a circular anchor placement.
    """
    assert p > 0
    if p == 1:
        axes_angle = [0]
    elif p == 2:
        axes_angle = [0, numpy.pi / 2]
    else:
        axes_angle = numpy.linspace(0, 2 * numpy.pi, p, endpoint=False)

    A = numpy.c_[numpy.cos(axes_angle), numpy.sin(axes_angle)]
    return A


def init_random(p, dim, rstate=None):
    if rstate is None:
        rstate = numpy.random
    elif not isinstance(rstate, numpy.random.RandomState):
        rstate = numpy.random.RandomState(rstate)

    return rstate.random((p, dim)) * 2 - 1
