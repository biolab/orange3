"""
Data-manipulation utilities.
"""
import numpy as np
import bottleneck as bn


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


def scale(values, min=0, max=1):
    """Return values scaled to [min, max]"""
    minval = np.float_(bn.nanmin(values))
    ptp = bn.nanmax(values) - minval
    if ptp == 0:
        return np.clip(values, min, max)
    return (-minval + values) / ptp * (max - min) + min
