import numpy as np

def argmaxrnd(vec, random_seed=None):

    """
    Returns the index of the maximum value for a given 1D array.
    In case of multiple indices corresponding to the maximum value,
    the result is chosen randomly among those. The random number
    generator can be seeded by forwarding a hashable python object.
    If -1 is passed, the input array is hashed instead.

    :param vec: 1D input array (vector) of real numbers
    :type vec: np.ndarray

    :param random_seed: used to initialize the random number generator
    :type random_seed: hashable python object

    :return: index of the maximum value
    """

    if vec.ndim != 1:
        raise ValueError("1D array of shape (n,) is expected.")

    if random_seed is None:
        return np.random.choice((vec == np.nanmax(vec)).nonzero()[0])

    if random_seed == -1:
        hsh = vec.astype(np.int)
    else:
        hsh = hash(random_seed)
    prng = np.random.RandomState(hsh)
    return prng.choice((vec == np.nanmax(vec)).nonzero()[0])
