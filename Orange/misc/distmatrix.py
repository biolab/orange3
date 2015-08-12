import numpy as np
from Orange.util import deprecated


class DistMatrix(np.ndarray):
    """
    Distance matrix. Extends ``numpy.ndarray``.

    .. attribute:: row_items

        Items corresponding to matrix rows.

    .. attribute:: col_items

        Items corresponding to matrix columns.

    .. attribute:: axis

        If axis=1 we calculate distances between rows,
        if axis=0 we calculate distances between columns.
    """
    def __new__(cls, data, row_items=None, col_items=None, axis=1):
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
        obj = np.asarray(data).view(cls)
        obj.row_items = row_items
        obj.col_items = col_items
        obj.axis = axis
        return obj

    def __array_finalize__(self, obj):
        """See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html"""
        if obj is None: return
        self.row_items = getattr(obj, 'row_items', None)
        self.col_items = getattr(obj, 'col_items', None)
        self.axis = getattr(obj, 'axis', 1)

    def __array_wrap__(self, out_arr, context=None):
        if out_arr.ndim == 0:  # a single scalar
            return out_arr.item()
        return np.ndarray.__array_wrap__(self, out_arr, context)

    """
    __reduce__() and __setstate__() ensure DistMatrix is picklable.
    """
    def __reduce__(self):
        state = super().__reduce__()
        newstate = state[2] + (self.row_items, self.col_items, self.axis)
        return state[0], state[1], newstate

    def __setstate__(self, state):
        self.row_items = state[-3]
        self.col_items = state[-2]
        self.axis = state[-1]
        super().__setstate__(state[0:-3])

    @property
    @deprecated
    def dim(self):
        """Returns the single dimension of the symmetric square matrix."""
        return self.shape[0]

    @property
    @deprecated
    def X(self):
        return self

    @property
    def flat(self):
        return self[np.triu_indices(self.shape[0], 1)]

    def get_KNN(self, i, k):
        """Return k columns with the lowest value in the i-th row.

        :param i: i-th row
        :type i: int
        :param k: number of neighbors
        :type k: int
        """
        idxs = np.argsort(self[i, :])[:]
        return self[:, idxs]

    def invert(self, typ):
        """Invert values in the distance matrix.

        :param type: 0 (-X), 1 (1 - X), 2 (max - X), 3 (1 / X)
        :type type: int
        """
        if typ == 0:
            return -self
        elif typ == 1:
            return 1.-self
        elif typ == 2:
            return 1./self
        else:
            raise ValueError('Unknown option for typ of matrix inversion.')

    def submatrix(self, row_items, col_items=None):
        """Return a submatrix of self, describing only distances between items"""
        if not col_items:
            col_items = row_items
        obj = self[np.ix_(row_items, col_items)]
        if obj.row_items:
            obj.row_items = self.row_items[row_items]
        if obj.col_items:
            obj.col_items = self.col_items[col_items]
        return obj
