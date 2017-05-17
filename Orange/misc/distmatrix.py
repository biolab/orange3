import numpy as np

from Orange.data import Table, StringVariable, Domain
from Orange.data.io import detect_encoding
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
        if obj is None:
            return
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

    # noinspection PyMethodOverriding,PyArgumentList
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

    # noinspection PyPep8Naming
    @property
    @deprecated
    def X(self):
        return self

    @property
    def flat(self):
        return self[np.triu_indices(self.shape[0], 1)]

    def submatrix(self, row_items, col_items=None):
        """
        Return a submatrix

        Args:
            row_items: indices of rows
            col_items: incides of columns
        """
        if not col_items:
            col_items = row_items
        obj = self[np.ix_(row_items, col_items)]
        if self.row_items is not None:
            obj.row_items = self.row_items[row_items]
        if self.col_items is not None:
            if self.col_items is self.row_items and row_items is col_items:
                obj.col_items = obj.row_items
            else:
                obj.col_items = self.col_items[col_items]
        return obj

    @classmethod
    def from_file(cls, filename):
        """
        Load distance matrix from a file

        The file should be preferrably encoded in ascii/utf-8. White space at
        the beginning and end of lines is ignored.

        The first line of the file starts with the matrix dimension. It
        can be followed by a list flags

        - *axis=<number>*: the axis number
        - *symmetric*: the matrix is symmetric; when reading the element (i, j)
          it's value is also assigned to (j, i)
        - *asymmetric*: the matrix is asymmetric
        - *row_labels*: the file contains row labels
        - *col_labels*: the file contains column labels

        By default, matrices are symmetric, have axis 1 and no labels are given.
        Flags *labeled* and *labelled* are obsolete aliases for *row_labels*.

        If the file has column labels, they follow in the second line.
        Row labels appear at the beginning of each row.
        Labels are arbitrary strings that cannot contain newlines and
        tabulators. Labels are stored as instances of `Table` with a single
        meta attribute named "label".

        The remaining lines contain tab-separated numbers, preceded with labels,
        if present. Lines are padded with zeros if necessary. If the matrix is
        symmetric, the file contains the lower triangle; any data above the
        diagonal is ignored.

        Args:
            filename: file name
        """
        with open(filename, encoding=detect_encoding(filename)) as fle:
            line = fle.readline()
            if not line:
                raise ValueError("empty file")
            data = line.strip().split()
            if not data[0].strip().isdigit():
                raise ValueError("distance file must begin with dimension")
            n = int(data.pop(0))
            symmetric = True
            axis = 1
            col_labels = row_labels = None
            for flag in data:
                if flag in ("labelled", "labeled", "row_labels"):
                    row_labels = []
                elif flag == "col_labels":
                    col_labels = []
                elif flag == "symmetric":
                    symmetric = True
                elif flag == "asymmetric":
                    symmetric = False
                else:
                    flag_data = flag.split("=")
                    if len(flag_data) == 2:
                        name, value = map(str.strip, flag_data)
                    else:
                        name, value = "", None
                    if name == "axis" and value.isdigit():
                        axis = int(value)
                    else:
                        raise ValueError("invalid flag '{}'".format(
                            flag, filename))
            if col_labels is not None:
                col_labels = [x.strip()
                              for x in fle.readline().strip().split("\t")]
                if len(col_labels) != n:
                    raise ValueError("mismatching number of column labels")

            matrix = np.zeros((n, n))
            for i, line in enumerate(fle):
                if i >= n:
                    raise ValueError("too many rows".format(filename))
                line = line.strip().split("\t")
                if row_labels is not None:
                    row_labels.append(line.pop(0).strip())
                if len(line) > n:
                    raise ValueError("too many columns in matrix row {}".
                                     format("'{}'".format(row_labels[i])
                                            if row_labels else i + 1))
                for j, e in enumerate(line[:i + 1 if symmetric else n]):
                    try:
                        matrix[i, j] = float(e)
                    except ValueError as exc:
                        raise ValueError(
                            "invalid element at row {}, column {}".format(
                                "'{}'".format(row_labels[i])
                                if row_labels else i + 1,
                                "'{}'".format(col_labels[j])
                                if col_labels else j + 1)) from exc
                    if symmetric:
                        matrix[j, i] = matrix[i, j]
        if col_labels:
            col_labels = Table.from_list(
                Domain([], metas=[StringVariable("label")]),
                [[item] for item in col_labels])
        if row_labels:
            row_labels = Table.from_list(
                Domain([], metas=[StringVariable("label")]),
                [[item] for item in row_labels])
        return cls(matrix, row_labels, col_labels, axis)

    @staticmethod
    def _trivial_labels(items):
        return items and \
               isinstance(items, Table) and \
               len(items.domain.metas) == 1 and \
               isinstance(items.domain.metas[0], StringVariable)

    def has_row_labels(self):
        """
        Returns `True` if row labels can be automatically determined from data

        For this, the `row_items` must be an instance of `Orange.data.Table`
        whose domain contains a single meta attribute, which has to be a string.
        The domain may contain other variables, but not meta attributes.
        """
        return self._trivial_labels(self.row_items)

    def has_col_labels(self):
        """
        Returns `True` if column labels can be automatically determined from
        data

        For this, the `col_items` must be an instance of `Orange.data.Table`
        whose domain contains a single meta attribute, which has to be a string.
        The domain may contain other variables, but not meta attributes.
        """
        return self._trivial_labels(self.col_items)

    def save(self, filename):
        """
        Save the distance matrix to a file in the file format described at
        :obj:`~Orange.misc.distmatrix.DistMatrix.from_file`.

        Args:
            filename: file name
        """
        n = len(self)
        data = "{}\taxis={}".format(n, self.axis)
        row_labels = col_labels = None
        if self.has_col_labels():
            data += "\tcol_labels"
            col_labels = self.col_items
        if self.has_row_labels():
            data += "\trow_labels"
            row_labels = self.row_items
        symmetric = np.allclose(self, self.T)
        if not symmetric:
            data += "\tasymmetric"
        with open(filename, "wt") as fle:
            fle.write(data + "\n")
            if col_labels is not None:
                fle.write("\t".join(str(e.metas[0]) for e in col_labels) + "\n")
            for i, row in enumerate(self):
                if row_labels is not None:
                    fle.write(str(row_labels[i].metas[0]) + "\t")
                if symmetric:
                    fle.write("\t".join(map(str, row[:i + 1])) + "\n")
                else:
                    fle.write("\t".join(map(str, row)) + "\n")
