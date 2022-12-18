import os.path

import numpy as np

from Orange.misc import _distmatrix_xlsx
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
        # defined in __new___, pylint: disable=attribute-defined-outside-init
        """See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html"""
        if obj is None:
            return
        self.row_items = getattr(obj, 'row_items', None)
        self.col_items = getattr(obj, 'col_items', None)
        self.axis = getattr(obj, 'axis', 1)

    def __array_wrap__(self, out_arr, context=None):
        if out_arr.ndim == 0:  # a single scalar
            return out_arr[()]
        return np.ndarray.__array_wrap__(self, out_arr, context)

    # __reduce__() and __setstate__() ensure DistMatrix is picklable.
    def __reduce__(self):
        state = super().__reduce__()
        newstate = state[2] + (self.row_items, self.col_items, self.axis)
        return state[0], state[1], newstate

    # noinspection PyMethodOverriding,PyArgumentList
    def __setstate__(self, state):
        # defined in __new___, pylint: disable=attribute-defined-outside-init
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
        if isinstance(self.row_items, list):
            obj.row_items = list(np.array(self.row_items)[row_items])
        elif self.row_items is not None:
            obj.row_items = self.row_items[row_items]

        if self.col_items is self.row_items and col_items is row_items:
            obj.col_items = obj.row_items
        elif isinstance(self.col_items, list):
            obj.col_items = list(np.array(self.col_items)[col_items])
        elif self.col_items is not None:
            obj.col_items = self.col_items[col_items]
        return obj

    @classmethod
    def from_file(cls, filename, sheet=None):
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
        _, ext = os.path.splitext(filename)
        if ext == ".xlsx":
            matrix, row_labels, col_labels, axis \
                = _distmatrix_xlsx.read_matrix(filename, sheet)
        else:
            assert sheet is None
            matrix, row_labels, col_labels, axis = cls._from_dst(filename)
        return cls(matrix,
                   cls._labels_to_tables(row_labels),
                   cls._labels_to_tables(col_labels),
                   axis)

    @staticmethod
    def _labels_to_tables(labels):
        # prevent circular imports, pylint: disable=import-outside-toplevel
        from Orange.data import Table, StringVariable, Domain

        if labels is None or isinstance(labels, Table):
            return labels
        return Table.from_numpy(
            Domain([], metas=[StringVariable("label")]),
            np.empty((len(labels), 0)), None, np.array(labels)[:, None])

    @classmethod
    def _from_dst(cls, filename):
        # prevent circular imports, pylint: disable=import-outside-toplevel
        from Orange.data.io import detect_encoding

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
                        raise ValueError(f"invalid flag '{flag}'")
            if col_labels is not None:
                col_labels = [x.strip()
                              for x in fle.readline().strip().split("\t")]
                if len(col_labels) != n:
                    raise ValueError("mismatching number of column labels, "
                                     f"{len(col_labels)} != {n}")

            def num_or_lab(n, labels):
                return f"'{labels[n]}'" if labels else str(n + 1)

            matrix = np.zeros((n, n))
            for i, line in enumerate(fle):
                if i >= n:
                    raise ValueError("too many rows")
                line = line.strip().split("\t")
                if row_labels is not None:
                    row_labels.append(line.pop(0).strip())
                if len(line) > n:
                    raise ValueError(
                        f"too many columns in matrix row "
                        f"{num_or_lab(i, row_labels)}")
                for j, e in enumerate(line[:i + 1 if symmetric else n]):
                    try:
                        matrix[i, j] = float(e)
                    except ValueError as exc:
                        raise ValueError(
                            "invalid element at "
                            f"row {num_or_lab(i, row_labels)}, "
                            f"column {num_or_lab(j, col_labels)}") from exc
                    if symmetric:
                        matrix[j, i] = matrix[i, j]
            return matrix, row_labels, col_labels, axis

    def auto_symmetricized(self, copy=False):
        def self_or_copy():
            return self.copy() if copy else self

        def get_labels(labels):
            return np.array(labels) if isinstance(labels, list) \
                else labels.metas[:, 0] if self._trivial_labels(labels) \
                else object()

        h, w = self.shape
        m = max(w, h)
        if (abs(h - w) > 1
                or self.row_items and self.col_items
                   and np.any(get_labels(self.row_items)
                              != get_labels(self.col_items))
                or self.row_items and len(self.row_items) != m
                or self.col_items and len(self.col_items) != m):
            return self_or_copy()

        nans = np.isnan(self)
        low_indices = np.tril_indices(h, -1)
        low_empty = np.all(nans[low_indices])
        high_indices = np.triu_indices(w, 1)
        high_empty = np.all(nans[high_indices])
        if low_empty is high_empty:  # both non-empty, or both empty (only diagonal)
            return self_or_copy()

        indices = low_indices if low_empty else high_indices
        if w == h:
            matrix = np.array(self)
        else:
            if low_empty:
                row = np.vstack((self[:, -1, None], [[0]])).T
                matrix = np.vstack((self, row))
            else:
                col = np.hstack((self[-1, None], [[0]])).T
                matrix = np.hstack((self, col))
            diag_indices = np.diag_indices(len(matrix))
            matrix[diag_indices] = np.nan_to_num(matrix[diag_indices])
        matrix[indices] = self.T[indices]
        return type(self)(matrix,
                          self.row_items or self.col_items,
                          self.col_items or self.row_items)

    def _trivial_labels(self, items):
        # prevent circular imports, pylint: disable=import-outside-toplevel
        from Orange.data import Table, StringVariable

        return (isinstance(items, (list, tuple))
                and all(isinstance(item, str) for item in items)
                or
                isinstance(items, Table)
                and (self.axis == 0 or
                     sum(isinstance(meta, StringVariable)
                         for meta in items.domain.metas) == 1
                     )
                )

    def is_symmetric(self):
        # prevent circular imports, pylint: disable=import-outside-toplevel
        from Orange.data import Table

        if self.shape[0] != self.shape[1] or not np.allclose(self, self.T):
            return False
        if self.row_items is None or self.col_items is None:
            return True
        if isinstance(self.row_items, Table):
            return (isinstance(self.col_items, Table)
                and self.col_items.domain == self.row_items.domain
                and np.array_equal(self.col_items.X, self.row_items.X)
                and np.array_equal(self.col_items.Y, self.row_items.Y)
                and np.array_equal(self.col_items.metas, self.row_items.metas))
        else:
            return (not isinstance(self.col_items, Table)
                and np.array_equal(self.row_items, self.col_items))

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

    def get_labels(self, items):
        # prevent circular imports, pylint: disable=import-outside-toplevel
        from Orange.data import StringVariable

        if not self._trivial_labels(items):
            return None
        if isinstance(items, (list, tuple)) \
                and all(isinstance(x, str) for x in items):
            return items
        if self.axis == 0:
            return [attr.name for attr in items.domain.attributes]
        else:
            string_var = next(var for var in items.domain.metas
                              if isinstance(var, StringVariable))
            return items.get_column(string_var)

    def save(self, filename):
        if os.path.splitext(filename)[1] == ".xlsx":
            _distmatrix_xlsx.write_matrix(self, filename)
        else:
            self._save_dst(filename)

    def _save_dst(self, filename):
        """
        Save the distance matrix to a file in the file format described at
        :obj:`~Orange.misc.distmatrix.DistMatrix.from_file`.

        Args:
            filename: file name
        """
        n = len(self)
        data = f"{n}\taxis={self.axis}"
        row_labels = col_labels = None
        if self.has_col_labels():
            data += "\tcol_labels"
            col_labels = self.col_items
        if self.has_row_labels():
            data += "\trow_labels"
            row_labels = self.row_items
        symmetric = self.is_symmetric()
        if not symmetric:
            data += "\tasymmetric"
        with open(filename, "wt", encoding="utf-8") as fle:
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
