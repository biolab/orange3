import numpy as np

class DistMatrix():
    """
    Distance matrix.

    .. attribute:: dim

        Matrix dimension.

    .. attribute:: X

        Matrix data
    """

    def __init__(self, data):
        """Construct a new distance matrix containing the given data.

        :param data: Distance matrix
        :type data: numpy array or Python list containing lists or tuples.
        """
        self.dim = data.shape
        self.X = np.array(data)

    def get_KNN(self, i, k):
        """Return k columns with the lowest value in the i-th row.

        :param i: i-th row
        :type i: int
        :param k: number of neighbors
        :type k: int
        """
        idxs = np.argsort(self.X[i, :])[:]
        return self.X[:, idxs]

    def invert(self, typ):
        """Invert values in the distance matrix.

        :param type: 0 (-X), 1 (1 - X), 2 (max - X), 3 (1 / X)
        :type type: int
        """
        if typ == 0:
            return -self.X
        elif typ == 1:
            return 1.-self.X
        elif typ == 2:
            return 1./self.X
        else:
            raise ValueError('Unknown option for typ of matrix inversion.')