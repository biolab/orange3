import numpy as np

from Orange.data import Domain


def radviz(data, attrs, points=None):
    x = data.transform(domain=Domain(attrs)).X
    mask = ~np.isnan(x).any(axis=1)
    x = x[mask]

    n = len(x)
    if not n:
        return None, None, mask
    x = normalize(x)

    r_x = np.zeros(n)
    r_y = np.zeros(n)

    m = x.shape[1]
    if points is not None:
        s = points[:, :2]
    else:
        s = np.array([(np.cos(t), np.sin(t))
                      for t in [2.0 * np.pi * (i / float(m))
                                for i in range(m)]])
    for i in range(n):
        row = x[i]
        row_ = np.repeat(np.expand_dims(row, axis=1), 2, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            a = (s * row_).sum(axis=0)
            b = row.sum()
            y = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        r_x[i] = y[0]
        r_y[i] = y[1]

    return np.stack((r_x, r_y), axis=1), np.column_stack((s, attrs)), mask


def normalize(x):
    """
    MinMax normalization to fit a matrix in the space [0,1] by column.
    """
    a = x.min(axis=0)
    b = x.max(axis=0)
    return (x - a[np.newaxis, :]) / ((b - a)[np.newaxis, :])
