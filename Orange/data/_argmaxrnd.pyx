from hashlib import sha1
from cython import boundscheck, cdivision
import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, srand, RAND_MAX


cdef inline bint isnan(np.float_t x) nogil:
    return x != x


def argmaxrnd(np.ndarray vec, object random_seed=None):

    """
    Returns the index of the maximum value for a given 1D array.
    In case of multiple indices corresponding to the maximum value,
    the result is chosen randomly among those. The random number
    generator, used by the C lang function 'rand', can be seeded
    by forwarding a hashable python object. If -1 is passed, the
    input array is hashed instead.

    :param vec: 1D input array (vector) of real numbers
    :type vec: np.ndarray

    :param random_seed: used to initialize the random number generator
    :type random_seed: hashable python object

    :return: index of the maximum value
    """

    if vec.ndim != 1:
        raise ValueError("1D array of shape (n,) is expected.")

    if random_seed is not None:
        if random_seed == -1:
            srand(int(sha1(bytes(vec)).hexdigest(), base=16) & 0xffffffff)
        else:
            srand(hash(random_seed) & 0xffffffff)

    cdef:
        np.dtype dtype = vec.dtype
    if dtype == np.float_:
        return argmaxrnd_float(vec)
    elif dtype == np.int_:
        return argmaxrnd_int(vec)
    else:
        raise ValueError("dtype {} not supported.".format(dtype))


@boundscheck(False)
@cdivision(True)
cdef Py_ssize_t argmaxrnd_float(np.ndarray[np.float_t, ndim=1] vec):
    cdef:
        Py_ssize_t i, m_i
        int c = 0
        np.float_t curr, m

    for i in range(vec.shape[0]):
        curr = vec[i]
        if not isnan(curr):
            if curr > m or c == 0:
                m = curr
                c = 1
                m_i = i
            elif curr == m:
                c += 1
                if 1 + <int>(1.0 * c * rand() / RAND_MAX) == c:
                    m_i = i
    return m_i


@boundscheck(False)
@cdivision(True)
cdef Py_ssize_t argmaxrnd_int(np.ndarray[np.int_t, ndim=1] vec):
    cdef:
        Py_ssize_t i, m_i = 0
        int c = 1
        np.int_t curr, m = vec[0]

    for i in range(1, vec.shape[0]):
        curr = vec[i]
        if curr > m:
            m = curr
            c = 1
            m_i = i
        elif curr == m:
            c += 1
            if 1 + <int>(1.0 * c * rand() / RAND_MAX) == c:
                m_i = i
    return m_i
