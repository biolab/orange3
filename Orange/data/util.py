"""
Data-manipulation utilities.
"""
import numpy as np


def one_hot(values, dtype=float):
    """Return a one-hot transform of values

    Parameters
    ----------
    values : 1d array
        Integer values (hopefully 0-max).

    Returns
    -------
    result
        2d array with ones in respective indicator columns.
    """
    return np.eye(np.max(values) + 1, dtype=dtype)[np.asanyarray(values, dtype=int)]
