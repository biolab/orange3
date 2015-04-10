import numpy as np


class DistMatrix():
    """
    Distance matrix.

    .. attribute:: dim

        Matrix dimension.

    .. attribute:: X

        Matrix data.

    .. attribute:: row_items

        Items corresponding to matrix rows.

    .. attribute:: col_items

        Items corresponding to matrix columns.

    .. attribute:: axis

        If axis=1 we calculate distances between rows,
        if axis=0 we calculate distances between columns.
    """
    def __init__(self, data, row_items=None, col_items=None, axis=1):
        """Construct a new distance matrix containing the given data.

        :param data: Distance matrix
        :type data: numpy array
        :param row_items: Items in matrix rows
        :type row_items: `Orange.data.Table` or `Orange.data.Instance`
        :param col_items: Items in matrix columns
        :type col_items: `Orange.data.Table` or `Orange.data.Instance`
        :param axis: The axis along which the distances are calculated
        :type axis: int

        """
        self.dim = data.shape
        self.X = np.array(data)
        self.row_items = row_items
        self.col_items = col_items
        self.axis = axis

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