import os
import zlib
from collections import MutableSequence, Iterable, Sequence, Sized
from itertools import chain
from numbers import Real
import operator
from functools import reduce
from warnings import warn

import numpy as np
import bottlechest as bn
from scipy import sparse as sp
from sklearn.utils import validation

from .instance import *
from Orange.data import (domain as orange_domain,
                         io, DiscreteVariable, ContinuousVariable)
from Orange.data.storage import Storage
from . import _contingency
from . import _valuecount


def get_sample_datasets_dir():
    orange_data_table = os.path.dirname(__file__)
    dataset_dir = os.path.join(orange_data_table, '..', 'datasets')
    return os.path.realpath(dataset_dir)

dataset_dirs = ['', get_sample_datasets_dir()]


class RowInstance(Instance):
    def __init__(self, table, row_index):
        """
        Construct a data instance representing the given row of the table.
        """
        super().__init__(table.domain)
        self._x = table.X[row_index]
        if sp.issparse(self._x):
            self.sparse_x = self._x
            self._x = np.asarray(self._x.todense())[0]
        self._y = table.Y[row_index]
        if sp.issparse(self._y):
            self.sparse_y = self._y
            self._y = np.asarray(self._y.todense())[0]
        self._metas = table.metas[row_index]
        if sp.issparse(self._metas):
            self.sparse_metas = self._metas
            self._metas = np.asarray(self._metas.todense())[0]
        self._values = np.hstack((self._x, self._y))
        self.row_index = row_index
        self.table = table


    @property
    def weight(self):
        if not self.table.has_weights():
            return 1
        return self.table.W[self.row_index]


    #noinspection PyMethodOverriding
    @weight.setter
    def weight(self, weight):
        if not self.table.has_weights():
            self.table.set_weights()
        self.table.W[self.row_index] = weight


    def set_class(self, value):
        self._check_single_class()
        if not isinstance(value, Real):
            value = self.table.domain.class_var.to_val(value)
        self._values[len(self.table.domain.attributes)] = self._y[0] = value
        if self.sparse_y:
            self.table.Y[self.row_index, 0] = value


    def __setitem__(self, key, value):
        if not isinstance(key, int):
            key = self._domain.index(key)
        if isinstance(value, str):
            var = self._domain[key]
            value = var.to_val(value)
        if key >= 0:
            if not isinstance(value, Real):
                raise TypeError("Expected primitive value, got '%s'" %
                                type(value).__name__)
            if key < len(self._x):
                self._values[key] = self._x[key] = value
                if self.sparse_x:
                    self.table.X[self.row_index, key] = value
            else:
                self._values[key] = self._y[key - len(self._x)] = value
                if self.sparse_y:
                    self.table.Y[self.row_index, key - len(self._x)] = value
        else:
            self._metas[-1 - key] = value
            if self.sparse_metas:
                self.table.metas[self.row_index, -1 - key] = value


    @staticmethod
    def sp_values(matrix, row, variables):
        if sp.issparse(matrix):
            begptr, endptr = matrix.indptr[row:row + 2]
            rendptr = min(endptr, begptr + 5)
            variables = [variables[var]
                         for var in matrix.indices[begptr:rendptr]]
            s = ", ".join("{}={}".format(var.name, var.str_val(val))
                for var, val in zip(variables, matrix.data[begptr:rendptr]))
            if rendptr != endptr:
                s += ", ..."
            return s
        else:
            return Instance.str_values(matrix[row], variables)


    def __str__(self):
        table = self.table
        domain = table.domain
        row = self.row_index
        s = "[" + self.sp_values(table.X, row, domain.attributes)
        if domain.class_vars:
            s += " | " + self.sp_values(table.Y, row, domain.class_vars)
        s += "]"
        if self._domain.metas:
            s += " {" + self.sp_values(table.metas, row, domain.metas) + "}"
        return s

    __repr__ = __str__


class Columns:
    def __init__(self, domain):
        for v in chain(domain, domain.metas):
            setattr(self, v.name.replace(" ", "_"), v)


class Table(MutableSequence, Storage):
    @property
    def columns(self):
        """
        A class whose attributes contain attribute descriptors for columns.
        For a table `table`, setting `c = table.columns` will allow accessing
        the table's variables with, for instance `c.gender`, `c.age` ets.
        Spaces are replaced with underscores.
        """
        return Columns(self.domain)


    def __new__(cls, *args, **kwargs):
        if not args and not kwargs:
            return super().__new__(cls)

        if 'filename' in kwargs:
            args = [kwargs.pop('filename')]

        try:
            if isinstance(args[0], str):
                return cls.from_file(args[0], **kwargs)

            if isinstance(args[0], orange_domain.Domain):
                domain, args = args[0], args[1:]
                if not args:
                    return cls.from_domain(domain, **kwargs)
                if isinstance(args[0], Table):
                    return cls.from_table(domain, args[0])
            else:
                domain = None

            return cls.from_numpy(domain, *args, **kwargs)
        except IndexError:
            pass
        raise ValueError("Invalid arguments for Table.__new__")


    @classmethod
    def from_domain(cls, domain, n_rows=0, weights=False):
        """
        Construct a new `Table` with the given number of rows for the given
        domain. The optional vector of weights is initialized to 1's.

        :param domain: domain for the `Table`
        :type domain: Orange.data.Domain
        :param n_rows: number of rows in the new table
        :type n_rows: int
        :param weights: indicates whether to construct a vector of weights
        :type weights: bool
        :return: a new table
        :rtype: Orange.data.Table
        """
        self = cls.__new__(Table)
        self.domain = domain
        self.n_rows = n_rows
        self.X = np.zeros((n_rows, len(domain.attributes)))
        self.Y = np.zeros((n_rows, len(domain.class_vars)))
        if weights:
            self.W = np.ones(n_rows)
        else:
            self.W = np.empty((n_rows, 0))
        self.metas = np.empty((n_rows, len(self.domain.metas)), object)
        return self


    @classmethod
    def from_table(cls, domain, source, row_indices=...):
        """
        Create a new table from selected columns and/or rows of an existing
        one. The columns are chosen using a domain. The domain may also include
        variables that do not appear in the source table; they are computed
        from source variables if possible.

        The resulting data may be a view or a copy of the existing data.

        :param domain: the domain for the new table
        :type domain: Orange.data.Domain
        :param source: the source table
        :type source: Orange.data.Table
        :param row_indices: indices of the rows to include
        :type row_indices: a slice or a sequence
        :return: a new table
        :rtype: Orange.data.Table
        """


        def get_columns(row_indices, src_cols, n_rows):
            if not len(src_cols):
                return np.zeros((n_rows, 0), dtype=source.X.dtype)

            n_src_attrs = len(source.domain.attributes)
            if all(isinstance(x, int) and 0 <= x < n_src_attrs
                   for x in src_cols):
                return source.X[row_indices, src_cols]
            if all(isinstance(x, int) and x < 0 for x in src_cols):
                return source.metas[row_indices, [-1 - x for x in src_cols]]
            if all(isinstance(x, int) and x >= n_src_attrs for x in src_cols):
                return source.Y[row_indices, [x - n_src_attrs for x in
                                              src_cols]]

            types = []
            if any(isinstance(x, int) and 0 <= x < n_src_attrs
                   for x in src_cols):
                types.append(source.X.dtype)
            if any(isinstance(x, int) and x < 0 for x in src_cols):
                types.append(source.metas.dtype)
            if any(isinstance(x, int) and x >= n_src_attrs for x in src_cols):
                types.append(source.Y.dtype)
            new_type = np.find_common_type(types, [])
            a = np.empty((n_rows, len(src_cols)), dtype=new_type)
            for i, col in enumerate(src_cols):
                if col is None:
                    a[:, i] = Unknown
                elif not isinstance(col, int):
                    a[:, i] = col(source)
                elif col < 0:
                    a[:, i] = source.metas[row_indices, -1 - col]
                elif col < n_src_attrs:
                    a[:, i] = source.X[row_indices, col]
                else:
                    a[:, i] = source.Y[row_indices, col - n_src_attrs]
            return a


        if domain == source.domain:
            return Table.from_table_rows(source, row_indices)

        if isinstance(row_indices, slice):
            start, stop, stride = row_indices.indices(source.X.shape[0])
            n_rows = (stop - start) // stride
            if n_rows < 0:
                n_rows = 0
        elif row_indices is ...:
            n_rows = len(source.X)
        else:
            n_rows = len(row_indices)

        self = cls.__new__(Table)
        self.domain = domain
        conversion = domain.get_conversion(source.domain)
        self.X = get_columns(row_indices, conversion.attributes, n_rows)
        self.Y = get_columns(row_indices, conversion.class_vars, n_rows)
        self.metas = get_columns(row_indices, conversion.metas, n_rows)
        self.W = np.array(source.W[row_indices])
        self.name = getattr(source, 'name', '')
        return self


    @classmethod
    def from_table_rows(cls, source, row_indices):
        """
        Construct a new table by selecting rows from the source table.

        :param source: an existing table
        :type source: Orange.data.Table
        :param row_indices: indices of the rows to include
        :type row_indices: a slice or a sequence
        :return: a new table
        :rtype: Orange.data.Table
        """
        self = cls.__new__(Table)
        self.domain = source.domain
        self.X = source.X[row_indices]
        self.Y = source.Y[row_indices]
        self.metas = source.metas[row_indices]
        self.W = source.W[row_indices]
        self.name = getattr(source, 'name', '')
        return self


    @classmethod
    def from_numpy(cls, domain, X, Y=None, metas=None, W=None):
        """
        Construct a table from numpy arrays with the given domain. The number
        of variables in the domain must match the number of columns in the
        corresponding arrays. All arrays must have the same number of rows.
        Arrays may be of different numpy types, and may be dense or sparse.

        :param domain: the domain for the new table
        :type domain: Orange.data.Domain
        :param X: array with attribute values
        :type X: np.array
        :param Y: array with class values
        :type Y: np.array
        :param metas: array with meta attributes
        :type metas: np.array
        :param W: array with weights
        :type W: np.array
        :return:
        """
        X, Y, metas, W = _check_arrays(X, Y, metas, W)

        if Y is not None and Y.ndim == 1:
            Y = Y.reshape(Y.shape[0], 1)
        if domain is None:
            domain = orange_domain.Domain.from_numpy(X, Y, metas)

        if Y is None:
            if sp.issparse(X):
                Y = np.empty((X.shape[0], 0), object)
            else:
                Y = X[:, len(domain.attributes):]
                X = X[:, :len(domain.attributes)]
        if metas is None:
            metas = np.empty((X.shape[0], 0), object)
        if W is None:
            W = np.empty((X.shape[0], 0))
        else:
            W = W.reshape(W.size, 1)

        if X.shape[1] != len(domain.attributes):
            raise ValueError(
                "Invalid number of variable columns ({} != {})".format(
                    X.shape[1], len(domain.attributes))
            )
        if Y.shape[1] != len(domain.class_vars):
            raise ValueError(
                "Invalid number of class columns ({} != {})".format(
                    Y.shape[1], len(domain.class_vars))
            )
        if metas.shape[1] != len(domain.metas):
            raise ValueError(
                "Invalid number of meta attribute columns ({} != {})".format(
                    metas.shape[1], len(domain.metas))
            )
        if not X.shape[0] == Y.shape[0] == metas.shape[0] == W.shape[0]:
            raise ValueError(
                "Parts of data contain different numbers of rows.")

        self = Table.__new__(Table)
        self.domain = domain
        self.X = X
        self.Y = Y
        self.metas = metas
        self.W = W
        self.n_rows = self.X.shape[0]
        return self

    def save(self, filename):
        """
        Save a data table to a file. The path can be absolute or relative.

        :param filename: File name
        :type filename: str
        """
        ext = os.path.splitext(filename)[1]
        if ext == ".tab":
            io.save_tab_delimited(filename, self)
        else:
            raise IOError("Unknown file name extension.")


    @classmethod
    def from_file(cls, filename):
        """
        Read a data table from a file. The path can be absolute or relative.

        :param filename: File name
        :type filename: str
        :return: a new data table
        :rtype: Orange.data.Table
        """
        for dir in dataset_dirs:
            ext = os.path.splitext(filename)[1]
            absolute_filename = os.path.join(dir, filename)
            if not ext:
                for ext in [".tab", ".txt", ".basket"]:
                    if os.path.exists(absolute_filename + ext):
                        absolute_filename += ext
                        break
            if os.path.exists(absolute_filename):
                break
        else:
            absolute_filename = ext = ""

        if not os.path.exists(absolute_filename):
            raise IOError('File "{}" was not found.'.format(filename))
        if ext == ".tab":
            data = io.TabDelimReader().read_file(absolute_filename, cls)
        elif ext == ".txt":
            data = io.TxtReader().read_file(absolute_filename, cls)
        elif ext == ".basket":
            data = io.BasketReader().read_file(absolute_filename, cls)
        else:
            raise IOError(
                'Extension "{}" is not recognized'.format(filename))

        data.name = os.path.splitext(os.path.split(filename)[-1])[0]
        return data

    # Helper function for __setitem__ and insert:
    # Set the row of table data matrices
    def _set_row(self, example, row):
        domain = self.domain
        if isinstance(example, Instance):
            if example.domain == domain:
                if isinstance(example, RowInstance):
                    self.X[row] = example._x
                    self.Y[row] = example._y
                else:
                    self.X[row] = example._values[:len(domain.attributes)]
                    self.Y[row] = example._values[len(domain.attributes):]
                self.metas[row] = example._metas
                return
            c = self.domain.get_conversion(example.domain)
            self.X[row] = [example._values[i] if isinstance(i, int) else
                           (Unknown if not i else i(example))
                           for i in c.attributes]
            self.Y[row] = [example._values[i] if isinstance(i, int) else
                           (Unknown if not i else i(example))
                           for i in c.class_vars]
            self.metas[row] = [example._values[i] if isinstance(i, int) else
                               (Unknown if not i else i(example))
                               for i in c.metas]
        else:
            self.X[row] = [var.to_val(val)
                           for var, val in zip(domain.attributes, example)]
            self.Y[row] = [var.to_val(val)
                           for var, val in
                           zip(domain.class_vars,
                               example[len(domain.attributes):])]
            self.metas[row] = Unknown


    # Helper function for __setitem__ and insert:
    # Return a list of new attributes and column indices,
    #  or (None, self.col_indices) if no new domain needs to be constructed
    def _compute_col_indices(self, col_idx):
        if col_idx is ...:
            return None, None
        if isinstance(col_idx, np.ndarray) and col_idx.dtype == bool:
            return ([attr for attr, c in zip(self.domain, col_idx) if c],
                    np.nonzero(col_idx))
        elif isinstance(col_idx, slice):
            s = len(self.domain.variables)
            start, end, stride = col_idx.indices(s)
            if col_idx.indices(s) == (0, s, 1):
                return None, None
            else:
                return (self.domain.variables[col_idx],
                        np.arange(start, end, stride))
        elif isinstance(col_idx, Iterable) and not isinstance(col_idx, str):
            attributes = [self.domain[col] for col in col_idx]
            if attributes == self.domain.attributes:
                return None, None
            return attributes, np.fromiter(
                (self.domain.index(attr) for attr in attributes), int)
        elif isinstance(col_idx, int):
            attr = self.domain[col_idx]
        else:
            attr = self.domain[col_idx]
            col_idx = self.domain.index(attr)
        return [attr], np.array([col_idx])


    # A helper function for extend and insert
    # Resize X, Y, metas and W.
    def _resize_all(self, new_length):
        old_length = self.X.shape[0]
        if old_length == new_length:
            return
        try:
            self.X.resize(new_length, self.X.shape[1])
            self.Y.resize(new_length, self.Y.shape[1])
            self.metas.resize(new_length, self.metas.shape[1])
            if self.W.ndim == 2:
                self.W.resize((new_length, 0))
            else:
                self.W.resize(new_length)
        except Exception:
            if self.X.shape[0] == new_length:
                self.X.resize(old_length, self.X.shape[1])
            if self.Y.shape[0] == new_length:
                self.Y.resize(old_length, self.Y.shape[1])
            if self.metas.shape[0] == new_length:
                self.metas.resize(old_length, self.metas.shape[1])
            if self.W.shape[0] == new_length:
                if self.W.ndim == 2:
                    self.W.resize((old_length, 0))
                else:
                    self.W.resize(old_length)
            raise


    def __getitem__(self, key):
        if isinstance(key, int):
            return RowInstance(self, key)
        if not isinstance(key, tuple):
            return Table.from_table_rows(self, key)

        if len(key) != 2:
            raise IndexError("Table indices must be one- or two-dimensional")

        row_idx, col_idx = key
        if isinstance(row_idx, int):
            try:
                col_idx = self.domain.index(col_idx)
                var = self.domain[col_idx]
                if 0 <= col_idx < len(self.domain.attributes):
                    return Value(var, self.X[row_idx, col_idx])
                elif col_idx >= len(self.domain.attributes):
                    return Value(
                        var,
                        self.Y[row_idx,
                                col_idx - len(self.domain.attributes)])
                elif col_idx < 0:
                    return Value(var, self.metas[row_idx, -1 - col_idx])
            except TypeError:
                row_idx = [row_idx]

        # multiple rows OR single row but multiple columns:
        # construct a new table
        attributes, col_indices = self._compute_col_indices(col_idx)
        if attributes is not None:
            n_attrs = len(self.domain.attributes)
            r_attrs = [attributes[i]
                       for i, col in enumerate(col_indices)
                       if 0 <= col < n_attrs]
            r_classes = [attributes[i]
                         for i, col in enumerate(col_indices)
                         if col >= n_attrs]
            r_metas = [attributes[i]
                       for i, col in enumerate(col_indices) if col < 0]
            domain = orange_domain.Domain(r_attrs, r_classes, r_metas)
        else:
            domain = self.domain
        return Table.from_table(domain, self, row_idx)


    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            if isinstance(value, Real):
                self.X[key, :] = value
                return
            self._set_row(value, key)
            return

        if len(key) != 2:
            raise IndexError("Table indices must be one- or two-dimensional")
        row_idx, col_idx = key

        # single row
        if isinstance(row_idx, int):
            if isinstance(col_idx, slice):
                col_idx = range(*slice.indices(col_idx, self.X.shape[1]))
            if not isinstance(col_idx, str) and isinstance(col_idx, Iterable):
                col_idx = list(col_idx)
            if not isinstance(col_idx, str) and isinstance(col_idx, Sized):
                if isinstance(value, (Sequence, np.ndarray)):
                    values = value
                elif isinstance(value, Iterable):
                    values = list(value)
                else:
                    raise TypeError("Setting multiple values requires a "
                                    "sequence or numpy array")
                if len(values) != len(col_idx):
                    raise ValueError("Invalid number of values")
            else:
                col_idx, values = [col_idx], [value]
            for value, col_idx in zip(values, col_idx):
                if not isinstance(value, int):
                    value = self.domain[col_idx].to_val(value)
                if not isinstance(col_idx, int):
                    col_idx = self.domain.index(col_idx)
                if col_idx >= 0:
                    if col_idx < self.X.shape[1]:
                        self.X[row_idx, col_idx] = value
                    else:
                        self.Y[row_idx, col_idx - self.X.shape[1]] = value
                else:
                    self.metas[row_idx, -1 - col_idx] = value

        # multiple rows, multiple columns
        attributes, col_indices = self._compute_col_indices(col_idx)
        if col_indices is ...:
            col_indices = range(len(self.domain))
        n_attrs = self.X.shape[1]
        if isinstance(value, str):
            if not attributes:
                attributes = self.domain.attributes
            for var, col in zip(attributes, col_indices):
                if 0 <= col < n_attrs:
                    self.X[row_idx, col] = var.to_val(value)
                elif col >= n_attrs:
                    self.Y[row_idx, col - n_attrs] = var.to_val(value)
                else:
                    self.metas[row_idx, -1 - col] = var.to_val(value)
        else:
            attr_cols = np.fromiter(
                (col for col in col_indices if 0 <= col < n_attrs), int)
            class_cols = np.fromiter(
                (col - n_attrs for col in col_indices if col >= n_attrs), int)
            meta_cols = np.fromiter(
                (-1 - col for col in col_indices if col < col), int)
            if value is None:
                value = Unknown

            if not isinstance(value, Real) and \
                    (len(attr_cols) or len(class_cols)):
                raise TypeError(
                    "Ordinary attributes can only have primitive values")
            if len(attr_cols):
                if len(attr_cols) == 1:
                    # scipy.sparse matrices only allow primitive indices.
                    attr_cols = attr_cols[0]
                self.X[row_idx, attr_cols] = value
            if len(class_cols):
                if len(class_cols) == 1:
                    # scipy.sparse matrices only allow primitive indices.
                    class_cols = class_cols[0]
                self.Y[row_idx, class_cols] = value
            if len(meta_cols):
                self.metas[row_idx, meta_cols] = value


    def __delitem__(self, key):
        if key is ...:
            key = range(len(self))
        self.X = np.delete(self.X, key, axis=0)
        self.Y = np.delete(self.Y, key, axis=0)
        self.metas = np.delete(self.metas, key, axis=0)
        self.W = np.delete(self.W, key, axis=0)


    def __len__(self):
        return self.X.shape[0]


    def __str__(self):
        s = "[" + ",\n ".join(str(ex) for ex in self[:5])
        if len(self) > 5:
            s += ",\n ..."
        s += "\n]"
        return s


    def clear(self):
        """Remove all rows from the table."""
        del self[...]


    def append(self, instance):
        """
        Append a data instance to the table.

        :param instance: a data instance
        :type instance: Orange.data.Instance or a sequence of values
        """
        self.insert(len(self), instance)


    def insert(self, row, instance):
        """
        Insert a data instance into the table.

        :param row: row index
        :type row: int
        :param instance: a data instance
        :type instance: Orange.data.Instance or a sequence of values
        """
        if row < 0:
            row += len(self)
        if row < 0 or row > len(self):
            raise IndexError("Index out of range")
        self._resize_all(len(self) + 1)
        if row < len(self):
            self.X[row + 1:] = self.X[row:-1]
            self.Y[row + 1:] = self.Y[row:-1]
            self.metas[row + 1:] = self.metas[row:-1]
            self.W[row + 1:] = self.W[row:-1]
        try:
            self._set_row(instance, row)
            if self.W.shape[-1]:
                self.W[row] = 1
        except Exception:
            self.X[row:-1] = self.X[row + 1:]
            self.Y[row:-1] = self.Y[row + 1:]
            self.metas[row:-1] = self.metas[row + 1:]
            self.W[row:-1] = self.W[row + 1:]
            self._resize_all(len(self) - 1)
            raise

    def extend(self, instances):
        """
        Extend the table with the given instances. The instances can be given
        as a table of the same or a different domain, or a sequence. In the
        latter case, each instances can be given as
        :obj:`~Orange.data.Instance` or a sequence of values (e.g. list,
        tuple, numpy.array).

        :param instances: additional instances
        :type instances: Orange.data.Table or a sequence of instances
        """
        old_length = len(self)
        self._resize_all(old_length + len(instances))
        try:
            # shortcut
            if isinstance(instances, Table) and instances.domain == self.domain:
                self.X[old_length:] = instances.X
                self.Y[old_length:] = instances.Y
                self.metas[old_length:] = instances.metas
                if self.W.shape[-1]:
                    if instances.W.shape[-1]:
                        self.W[old_length:] = instances.W
                    else:
                        self.W[old_length:] = 1
            else:
                for i, example in enumerate(instances):
                    self[old_length + i] = example
        except Exception:
            self._resize_all(old_length)
            raise


    def is_view(self):
        """
        Return `True` if all arrays represent a view referring to another table
        """
        return ((not self.X.shape[-1] or self.X.base is not None) and
                (not self.Y.shape[-1] or self.Y.base is not None) and
                (not self.metas.shape[-1] or self.metas.base is not None) and
                (not self._weights.shape[-1] or self.W.base is not None))


    def is_copy(self):
        """
        Return `True` if the table owns its data
        """
        return ((not self.X.shape[-1] or self.X.base is None) and
                (self.Y.base is None) and
                (self.metas.base is None) and
                (self.W.base is None))


    def ensure_copy(self):
        """
        Ensure that the table owns its data; copy arrays when necessary
        """
        if self.X.base is not None:
            self.X = self.X.copy()
        if self.Y.base is not None:
            self.Y = self.Y.copy()
        if self.metas.base is not None:
            self.metas = self.metas.copy()
        if self.W.base is not None:
            self.W = self.W.copy()


    @staticmethod
    def __determine_density(data):
        if data is None:
            return Storage.Missing
        if data is not None and sp.issparse(data):
            try:
                if bn.bincount(data.data, 1)[0][0] == 0:
                    return Storage.SPARSE_BOOL
            except ValueError as e:
                pass
            return Storage.SPARSE
        else:
            return Storage.DENSE


    def X_density(self):
        if not hasattr(self, "_X_density"):
            self._X_density = Table.__determine_density(self.X)
        return self._X_density


    def Y_density(self):
        if not hasattr(self, "_Y_density"):
            self._Y_density = Table.__determine_density(self.Y)
        return self._Y_density


    def metas_density(self):
        if not hasattr(self, "_metas_density"):
            self._metas_density = Table.__determine_density(self.metas)
        return self._metas_density


    def set_weights(self, weight=1):
        """
        Set weights of data instances; create a vector of weights if necessary.
        """
        if not self.W.shape[-1]:
            self.W = np.empty(len(self))
        self.W[:] = weight


    def has_weights(self):
        """Return `True` if the data instances are weighed. """
        return self.W.shape[-1] != 0


    def total_weight(self):
        """
        Return the total weight of instances in the table, or their number if
        they are unweighted.
        """
        if self.W.shape[-1]:
            return sum(self.W)
        return len(self)


    def has_missing(self):
        """Return `True` if there are any missing attribute or class values."""
        return bn.anynan(self.X) or bn.anynan(self.Y)


    def has_missing_class(self):
        """Return `True` if there are any missing class values."""
        return bn.anynan(self.Y)


    def checksum(self, include_metas=True):
        # TODO: zlib.adler32 does not work for numpy arrays with dtype object
        # (after pickling and unpickling such arrays, checksum changes)
        # Why, and should we fix it or remove it?
        """Return a checksum over X, Y, metas and W."""
        cs = zlib.adler32(self.X)
        cs = zlib.adler32(self.Y, cs)
        if include_metas:
            cs = zlib.adler32(self.metas, cs)
        cs = zlib.adler32(self.W, cs)
        return cs


    def shuffle(self):
        """Randomly shuffle the rows of the table."""
        ind = np.arange(self.X.shape[0])
        np.random.shuffle(ind)
        self.X = self.X[ind]
        self.Y = self.Y[ind]
        self.metas = self.metas[ind]
        self.W = self.W[ind]


    def get_column_view(self, index):
        """
        Return a vector - as a view, not a copy - with a column of the table,
        and a bool flag telling whether this column is sparse. Note that
        vertical slicing of sparse matrices is inefficient.

        :param index: the index of the column
        :type index: int, str or Orange.data.Variable
        :return: (one-dimensional numpy array, sparse)
        """
        def rx(M):
            if sp.issparse(M):
                return np.asarray(M.todense())[:, 0], True
            else:
                return M, False

        if not isinstance(index, int):
            index = self.domain.index(index)
        if index >= 0:
            if index < self.X.shape[1]:
                return rx(self.X[:, index])
            else:
                return rx(self.Y[:, index - self.X.shape[1]])
        else:
            return rx(self.metas[:, -1 - index])


    def _filter_is_defined(self, columns=None, negate=False):
        if columns is None:
            if sp.issparse(self.X):
                remove = (self.X.indptr[1:] !=
                          self.X.indptr[-1:] + self.X.shape[1])
            else:
                remove = bn.anynan(self.X, axis=1)
            if sp.issparse(self.Y):
                remove = np.logical_or(remove, self.Y.indptr[1:] !=
                                       self.Y.indptr[-1:] + self.Y.shape[1])
            else:
                remove = np.logical_or(remove, bn.anynan(self.Y, axis=1))
        else:
            remove = np.zeros(len(self), dtype=bool)
            for column in columns:
                col, sparse = self.get_column_view(column)
                if sparse:
                    remove = np.logical_or(remove, col == 0)
                else:
                    remove = np.logical_or(remove, bn.anynan([col], axis=0))
        retain = remove if negate else np.logical_not(remove)
        return Table.from_table_rows(self, retain)


    def _filter_has_class(self, negate=False):
        if sp.issparse(self.Y):
            if negate:
                retain = (self.Y.indptr[1:] !=
                          self.Y.indptr[-1:] + self.Y.shape[1])
            else:
                retain = (self.Y.indptr[1:] ==
                          self.Y.indptr[-1:] + self.Y.shape[1])
        else:
            retain = bn.anynan(self.Y, axis=1)
            if not negate:
               retain = np.logical_not(retain)
        return Table.from_table_rows(self, retain)


    # filter_random is not defined - the one implemented in the
    # filter.py is just as fast


    def _filter_same_value(self, column, value, negate=False):
        if not isinstance(value, Real):
            value = self.domain[column].to_val(value)
        sel = self.get_column_view(column)[0] == value
        if negate:
            sel = np.logical_not(sel)
        return Table.from_table_rows(self, sel)


    def _filter_values(self, filter):
        from Orange.data import filter as data_filter

        if isinstance(filter, data_filter.Values):
            conditions = filter.conditions
            conjunction = filter.conjunction
        else:
            conditions = [filter]
            conjunction = True
        if conjunction:
            sel = np.ones(len(self), dtype=bool)
        else:
            sel = np.zeros(len(self), dtype=bool)

        for f in conditions:
            col = self.get_column_view(f.column)[0]
            if isinstance(f, data_filter.FilterDiscrete) and f.values is None \
                    or isinstance(f, data_filter.FilterContinuous) and \
                    f.oper == f.IsDefined:
                if conjunction:
                    sel *= np.isnan(col)
                else:
                    sel += np.isnan(col)
            elif isinstance(f, data_filter.FilterString) and \
                    f.oper == f.IsDefined:
                if conjunction:
                    sel *= (col != "")
                else:
                    sel += (col != "")
            elif isinstance(f, data_filter.FilterDiscrete):
                if conjunction:
                    s2 = np.zeros(len(self), dtype=bool)
                    for val in f.values:
                        if not isinstance(val, Real):
                            val = self.domain[f.column].to_val(val)
                        s2 += (col == val)
                    sel *= s2
                else:
                    for val in f.values:
                        if not isinstance(val, Real):
                            val = self.domain[f.column].to_val(val)
                        sel += (col == val)
            elif isinstance(f, data_filter.FilterStringList):
                if not f.case_sensitive:
                    #noinspection PyTypeChecker
                    col = np.char.lower(np.array(col, dtype=str))
                    vals = [val.lower() for val in f.values]
                else:
                    vals = f.values
                if conjunction:
                    sel *= reduce(operator.add,
                                  (col == val for val in vals))
                else:
                    sel = reduce(operator.add,
                                 (col == val for val in vals), sel)
            elif isinstance(f, (data_filter.FilterContinuous,
                                data_filter.FilterString)):
                if (isinstance(f, data_filter.FilterString) and
                        not f.case_sensitive):
                    #noinspection PyTypeChecker
                    col = np.char.lower(np.array(col, dtype=str))
                    fmin = f.min.lower()
                    if f.oper in [f.Between, f.Outside]:
                        fmax = f.max.lower()
                else:
                    fmin, fmax = f.min, f.max
                if f.oper == f.Equal:
                    col = (col == fmin)
                elif f.oper == f.NotEqual:
                    col = (col != fmin)
                elif f.oper == f.Less:
                    col = (col < fmin)
                elif f.oper == f.LessEqual:
                    col = (col <= fmin)
                elif f.oper == f.Greater:
                    col = (col > fmin)
                elif f.oper == f.GreaterEqual:
                    col = (col >= fmin)
                elif f.oper == f.Between:
                    col = (col >= fmin) * (col <= fmax)
                elif f.oper == f.Outside:
                    col = (col < fmin) + (col > fmax)
                elif not isinstance(f, data_filter.FilterString):
                    raise TypeError("Invalid operator")
                elif f.oper == f.Contains:
                    col = np.fromiter((fmin in e for e in col),
                                      dtype=bool)
                elif f.oper == f.StartsWith:
                    col = np.fromiter((e.startswith(fmin) for e in col),
                                      dtype=bool)
                elif f.oper == f.EndsWith:
                    col = np.fromiter((e.endswith(fmin) for e in col),
                                      dtype=bool)
                else:
                    raise TypeError("Invalid operator")
                if conjunction:
                    sel *= col
                else:
                    sel += col
            else:
                raise TypeError("Invalid filter")

        if filter.negate:
            sel = ~sel
        return Table.from_table_rows(self, sel)


    def _compute_basic_stats(self, columns=None,
                             include_metas=False, compute_variance=False):
        if compute_variance:
            raise NotImplementedError("computation of variance is "
                                      "not implemented yet")
        W = self.W if self.has_weights() else None
        rr = []
        if not columns:
            if self.domain.attributes:
                rr.append(bn.stats(self.X, W))
            if self.domain.class_vars:
                rr.append(bn.stats(self.Y, W))
            if include_metas and self.domain.metas:
                rr.append(bn.stats(self.metas, W))
            stats = np.vstack(tuple(rr))
        else:
            columns = [self.domain.index(c) for c in columns]
            nattrs = len(self.domain.attributes)
            Xs = any(0 <= c < nattrs for c in columns) and bn.stats(self.X, W)
            Ys = any(c >= nattrs for c in columns) and bn.stats(self.Y, W)
            ms = any(c < 0 for c in columns) and bn.stats(self.metas, W)
            stats = []
            for column in columns:
                if 0 <= column < nattrs:
                    stats.append(Xs[column, :])
                elif column >= nattrs:
                    stats.append(Ys[column - nattrs, :])
                else:
                    stats.append(ms[-1 - column])
        return stats

    def _compute_distributions(self, columns=None):
        def _get_matrix(M, cachedM, col):
            nonlocal single_column
            if not sp.issparse(M):
                return M[:, col], self.W if self.has_weights() else None, None
            if cachedM is None:
                if single_column:
                    warn(ResourceWarning,
                         "computing distributions on sparse data "
                         "for a single column is inefficient")
                cachedM = sp.csc_matrix(self.X)
            data = cachedM.data[cachedM.indptr[col]:cachedM.indptr[col+1]]
            if self.has_weights():
                weights = self.W[
                    cachedM.indices[cachedM.indptr[col]:cachedM.indptr[col+1]]]
            else:
                weights = None
            return data, weights, cachedM


        if columns is None:
            columns = range(len(self.domain.variables))
            single_column = False
        else:
            columns = [self.domain.index(var) for var in columns]
            single_column = len(columns) == 1 and len(self.domain) > 1
        distributions = []
        Xcsc = Ycsc = None
        for col in columns:
            var = self.domain[col]
            if col < self.X.shape[1]:
                m, W, Xcsc = _get_matrix(self.X, Xcsc, col)
            else:
                m, W, Ycsc = _get_matrix(self.Y, Ycsc, col - self.X.shape[1])
            if isinstance(var, DiscreteVariable):
                if W is not None:
                    W = W.ravel()
                dist, unknowns = bn.bincount(m, len(var.values)-1, W)
            elif not len(m):
                dist, unknowns = np.zeros((2, 0)), 0
            else:
                if W is not None:
                    ranks = np.argsort(m)
                    vals = np.vstack((m[ranks], W[ranks]))
                    unknowns = bn.countnans(m, W)
                else:
                    vals = np.ones((2, m.shape[0]))
                    vals[0, :] = m
                    vals[0, :].sort()
                    unknowns = bn.countnans(m)
                dist = np.array(_valuecount.valuecount(vals))
            distributions.append((dist, unknowns))

        return distributions


    def _compute_contingency(self, col_vars=None, row_var=None):
        n_atts = self.X.shape[1]

        if col_vars is None:
            col_vars = range(len(self.domain.variables))
            single_column = False
        else:
            col_vars = [self.domain.index(var) for var in col_vars]
            single_column = len(col_vars) == 1 and len(self.domain) > 1
        if row_var is None:
            row_var = self.domain.class_var
            if row_var is None:
                raise ValueError("No row variable")

        row_desc = self.domain[row_var]
        if not isinstance(row_desc, DiscreteVariable):
            raise TypeError("Row variable must be discrete")
        row_indi = self.domain.index(row_var)
        n_rows = len(row_desc.values)
        if 0 <= row_indi < n_atts:
            row_data = self.X[:, row_indi]
        elif row_indi < 0:
            row_data = self.metas[:, -1 - row_indi]
        else:
            row_data = self.Y[:, row_indi - n_atts]

        W = self.W if self.has_weights() else None

        col_desc = [self.domain[var] for var in col_vars]
        col_indi = [self.domain.index(var) for var in col_vars]

        if any(not isinstance(var, (ContinuousVariable, DiscreteVariable))
               for var in col_desc):
            raise ValueError("contingency can be computed only for discrete "
                             "and continuous values")

        if any(isinstance(var, ContinuousVariable) for var in col_desc):
            if bn.countnans(row_data):
                raise ValueError("cannot compute contigencies with missing "
                                 "row data")

        contingencies = [None] * len(col_desc)
        for arr, f_cond, f_ind in (
                (self.X, lambda i: 0 <= i < n_atts, lambda i: i),
                (self.Y, lambda i: i >= n_atts, lambda i: i - n_atts),
                (self.metas, lambda i: i < 0, lambda i: -1 - i)):

            arr_indi = [e for e, ind in enumerate(col_indi) if f_cond(ind)]

            vars = [(e, f_ind(col_indi[e]), col_desc[e]) for e in arr_indi]
            disc_vars = [v for v in vars if isinstance(v[2], DiscreteVariable)]
            if disc_vars:
                if sp.issparse(arr):
                    max_vals = max(len(v[2].values) for v in disc_vars)
                    disc_indi = {i for _, i, _ in disc_vars}
                    mask = [i in disc_indi for i in range(arr.shape[1])]
                    conts, nans = bn.contingency(arr, row_data, max_vals - 1,
                                                 n_rows - 1, W, mask)
                    for col_i, arr_i, _ in disc_vars:
                        contingencies[col_i] = (conts[arr_i], nans[arr_i])
                else:
                    for col_i, arr_i, var in disc_vars:
                        contingencies[col_i] = bn.contingency(arr[:, arr_i],
                            row_data, len(var.values) - 1, n_rows - 1, W)

            cont_vars = [v for v in vars if isinstance(v[2], ContinuousVariable)]
            if cont_vars:

                classes = row_data.astype(dtype=np.int8)
                if W is not None:
                    W = W.astype(dtype=np.float64)
                if sp.issparse(arr):
                    arr = sp.csc_matrix(arr)

                for col_i, arr_i, _ in cont_vars:
                    if sp.issparse(arr):
                        col_data = arr.data[arr.indptr[arr_i]:
                                            arr.indptr[arr_i+1]]
                        rows = arr.indices[arr.indptr[arr_i]:
                                           arr.indptr[arr_i+1]]
                        W_ = None if W is None else W[rows]
                        classes_ = classes[rows]
                    else:
                        col_data, W_, classes_ = arr[:, arr_i], W, classes

                    col_data = col_data.astype(dtype=np.float64)
                    U, C, unknown = _contingency.contingency_floatarray( \
                        col_data, classes_, n_rows, W_)
                    contingencies[col_i] = ([U, C], unknown)

        return contingencies


def _check_arrays(*arrays):
    checked = []
    if not len(arrays):
        return checked

    def ninstances(array):
        if hasattr(array, "shape"):
            return array.shape[0]
        else:
            return len(array)

    shape_1 = ninstances(arrays[0])

    for array in arrays:
        if array is None:
            checked.append(array)
            continue

        if ninstances(array) != shape_1:
            raise ValueError("Leading dimension mismatch (%d != %d)"
                             % (len(array), shape_1))

        if sp.issparse(array):
            array.data = np.asarray(array.data)
            has_inf = _check_inf(array.data)
        else:
            array = np.asarray(array)
            has_inf = _check_inf(array)

        if has_inf:
            raise ValueError("Array contains infinity.")
        checked.append(array)

    return checked


def _check_inf(array):
    return array.dtype.char in np.typecodes['AllFloat'] and \
            np.isinf(array.data).any()
