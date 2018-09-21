import operator
import os
import zlib
from collections import MutableSequence, Iterable, Sequence, Sized
from functools import reduce
from itertools import chain
from numbers import Real, Integral
from threading import Lock, RLock

import bottleneck as bn
import numpy as np
from scipy import sparse as sp

import Orange.data  # import for io.py
from Orange.data import (
    _contingency, _valuecount,
    Domain, Variable, Storage, StringVariable, Unknown, Value, Instance,
    ContinuousVariable, DiscreteVariable, MISSING_VALUES
)
from Orange.data.util import SharedComputeValue, vstack, hstack, \
    assure_array_dense, assure_array_sparse, \
    assure_column_dense, assure_column_sparse
from Orange.statistics.util import bincount, countnans, contingency, \
    stats as fast_stats, sparse_has_implicit_zeros, sparse_count_implicit_zeros, \
    sparse_implicit_zero_weights
from Orange.util import flatten

__all__ = ["dataset_dirs", "get_sample_datasets_dir", "RowInstance", "Table"]


def get_sample_datasets_dir():
    orange_data_table = os.path.dirname(__file__)
    dataset_dir = os.path.join(orange_data_table, '..', 'datasets')
    return os.path.realpath(dataset_dir)


dataset_dirs = ['', get_sample_datasets_dir()]


"""Domain conversion cache used in Table.from_table. It is global so that
chaining of domain conversions also works with caching even with descendants
of Table."""
_conversion_cache = None
_conversion_cache_lock = RLock()


class RowInstance(Instance):
    sparse_x = None
    sparse_y = None
    sparse_metas = None
    _weight = None

    def __init__(self, table, row_index):
        """
        Construct a data instance representing the given row of the table.
        """
        self.table = table
        self._domain = table.domain
        self.row_index = row_index
        self.id = table.ids[row_index]
        self._x = table.X[row_index]
        if sp.issparse(self._x):
            self.sparse_x = sp.csr_matrix(self._x)
            self._x = np.asarray(self._x.todense())[0]
        self._y = table._Y[row_index]
        if sp.issparse(self._y):
            self.sparse_y = sp.csr_matrix(self._y)
            self._y = np.asarray(self._y.todense())[0]
        self._metas = table.metas[row_index]
        if sp.issparse(self._metas):
            self.sparse_metas = sp.csr_matrix(self._metas)
            self._metas = np.asarray(self._metas.todense())[0]

    @property
    def weight(self):
        if not self.table.has_weights():
            return 1
        return self.table.W[self.row_index]

    @weight.setter
    def weight(self, weight):
        if not self.table.has_weights():
            self.table.set_weights()
        self.table.W[self.row_index] = weight

    def set_class(self, value):
        self._check_single_class()
        if not isinstance(value, Real):
            value = self.table.domain.class_var.to_val(value)
        self._y[0] = value
        if self.sparse_y:
            self.table._Y[self.row_index, 0] = value

    def __setitem__(self, key, value):
        if not isinstance(key, Integral):
            key = self._domain.index(key)
        if isinstance(value, str):
            var = self._domain[key]
            value = var.to_val(value)
        if key >= 0:
            if not isinstance(value, Real):
                raise TypeError("Expected primitive value, got '%s'" %
                                type(value).__name__)
            if key < len(self._x):
                self._x[key] = value
                if self.sparse_x is not None:
                    self.table.X[self.row_index, key] = value
            else:
                self._y[key - len(self._x)] = value
                if self.sparse_y is not None:
                    self.table._Y[self.row_index, key - len(self._x)] = value
        else:
            self._metas[-1 - key] = value
            if self.sparse_metas:
                self.table.metas[self.row_index, -1 - key] = value

    def _str(self, limit):
        def sp_values(matrix, variables):
            if not sp.issparse(matrix):
                if matrix.ndim == 1:
                    matrix = matrix[:, np.newaxis]
                return Instance.str_values(matrix[row], variables, limit)

            row_entries, idx = [], 0
            while idx < len(variables):
                # Make sure to stop printing variables if we limit the output
                if limit and len(row_entries) >= 5:
                    break

                var = variables[idx]
                if var.is_discrete or matrix[row, idx]:
                    row_entries.append("%s=%s" % (var.name, var.str_val(matrix[row, idx])))

                idx += 1

            s = ", ".join(row_entries)

            if limit and idx < len(variables):
                s += ", ..."

            return s

        table = self.table
        domain = table.domain
        row = self.row_index
        s = "[" + sp_values(table.X, domain.attributes)
        if domain.class_vars:
            s += " | " + sp_values(table.Y, domain.class_vars)
        s += "]"
        if self._domain.metas:
            s += " {" + sp_values(table.metas, domain.metas) + "}"
        return s

    def __str__(self):
        return self._str(False)

    def __repr__(self):
        return self._str(True)


class Columns:
    def __init__(self, domain):
        for v in chain(domain.variables, domain.metas):
            setattr(self, v.name.replace(" ", "_"), v)


# noinspection PyPep8Naming
class Table(MutableSequence, Storage):
    __file__ = None
    name = "untitled"

    @property
    def columns(self):
        """
        A class whose attributes contain attribute descriptors for columns.
        For a table `table`, setting `c = table.columns` will allow accessing
        the table's variables with, for instance `c.gender`, `c.age` ets.
        Spaces are replaced with underscores.
        """
        return Columns(self.domain)

    _next_instance_id = 0
    _next_instance_lock = Lock()

    @property
    def Y(self):
        if self._Y.shape[1] == 1:
            return self._Y[:, 0]
        return self._Y

    @Y.setter
    def Y(self, value):
        if len(value.shape) == 1:
            value = value[:, None]
        if sp.issparse(value) and len(self) != value.shape[0]:
            value = value.T
        self._Y = value

    def __new__(cls, *args, **kwargs):
        if not args and not kwargs:
            return super().__new__(cls)

        if 'filename' in kwargs:
            args = [kwargs.pop('filename')]

        if not args:
            raise TypeError(
                "Table takes at least 1 positional argument (0 given))")

        if isinstance(args[0], str):
            if args[0].startswith('https://') or args[0].startswith('http://'):
                return cls.from_url(args[0], **kwargs)
            else:
                return cls.from_file(args[0])
        elif isinstance(args[0], Table):
            return cls.from_table(args[0].domain, args[0])
        elif isinstance(args[0], Domain):
            domain, args = args[0], args[1:]
            if not args:
                return cls.from_domain(domain, **kwargs)
            if isinstance(args[0], Table):
                return cls.from_table(domain, *args)
            elif isinstance(args[0], list):
                return cls.from_list(domain, *args)
        else:
            domain = None

        return cls.from_numpy(domain, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        # So subclasses can expect to call super without breakage; noop
        pass

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
        self = cls()
        self.domain = domain
        self.n_rows = n_rows
        self.X = np.zeros((n_rows, len(domain.attributes)))
        self.Y = np.zeros((n_rows, len(domain.class_vars)))
        if weights:
            self.W = np.ones(n_rows)
        else:
            self.W = np.empty((n_rows, 0))
        self.metas = np.empty((n_rows, len(self.domain.metas)), object)
        cls._init_ids(self)
        self.attributes = {}
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

        global _conversion_cache

        def get_columns(row_indices, src_cols, n_rows, dtype=np.float64, is_sparse=False):
            if not len(src_cols):
                if is_sparse:
                    return sp.csr_matrix((n_rows, 0), dtype=source.X.dtype)
                else:
                    return np.zeros((n_rows, 0), dtype=source.X.dtype)

            # match density for subarrays
            match_density = assure_array_sparse if is_sparse else assure_array_dense
            n_src_attrs = len(source.domain.attributes)
            if all(isinstance(x, Integral) and 0 <= x < n_src_attrs
                   for x in src_cols):
                return match_density(_subarray(source.X, row_indices, src_cols))
            if all(isinstance(x, Integral) and x < 0 for x in src_cols):
                arr = match_density(_subarray(source.metas, row_indices,
                                            [-1 - x for x in src_cols]))
                if arr.dtype != dtype:
                    return arr.astype(dtype)
                return arr
            if all(isinstance(x, Integral) and x >= n_src_attrs
                   for x in src_cols):
                return match_density(_subarray(
                    source._Y, row_indices,
                    [x - n_src_attrs for x in src_cols]))

            # initialize final array & set `match_density` for columns
            if is_sparse:
                a = sp.dok_matrix((n_rows, len(src_cols)), dtype=dtype)
                match_density = assure_column_sparse
            else:
                a = np.empty((n_rows, len(src_cols)), dtype=dtype)
                match_density = assure_column_dense

            shared_cache = _conversion_cache
            for i, col in enumerate(src_cols):
                if col is None:
                    a[:, i] = Unknown
                elif not isinstance(col, Integral):
                    if isinstance(col, SharedComputeValue):
                        if (id(col.compute_shared), id(source)) not in shared_cache:
                            shared_cache[id(col.compute_shared), id(source)] = \
                                col.compute_shared(source)
                        shared = shared_cache[id(col.compute_shared), id(source)]
                        if row_indices is not ...:
                            a[:, i] = match_density(
                                col(source, shared_data=shared)[row_indices])
                        else:
                            a[:, i] = match_density(
                                col(source, shared_data=shared))
                    else:
                        if row_indices is not ...:
                            a[:, i] = match_density(col(source)[row_indices])
                        else:
                            a[:, i] = match_density(col(source))
                elif col < 0:
                    a[:, i] = match_density(source.metas[row_indices, -1 - col])
                elif col < n_src_attrs:
                    a[:, i] = match_density(source.X[row_indices, col])
                else:
                    a[:, i] = match_density(
                        source._Y[row_indices, col - n_src_attrs])

            if is_sparse:
                a = a.tocsr()

            return a

        with _conversion_cache_lock:
            new_cache = _conversion_cache is None
            try:
                if new_cache:
                    _conversion_cache = {}
                else:
                    cached = _conversion_cache.get((id(domain), id(source)))
                    if cached:
                        return cached
                if domain == source.domain:
                    table = cls.from_table_rows(source, row_indices)
                    # assure resulting domain is the instance passed on input
                    table.domain = domain
                    # since sparse flags are not considered when checking for
                    # domain equality, fix manually.
                    table = assure_domain_conversion_sparsity(table, source)
                    return table

                if isinstance(row_indices, slice):
                    start, stop, stride = row_indices.indices(source.X.shape[0])
                    n_rows = (stop - start) // stride
                    if n_rows < 0:
                        n_rows = 0
                elif row_indices is ...:
                    n_rows = len(source)
                else:
                    n_rows = len(row_indices)

                self = cls()
                self.domain = domain
                conversion = domain.get_conversion(source.domain)
                self.X = get_columns(row_indices, conversion.attributes, n_rows,
                                     is_sparse=conversion.sparse_X)
                if self.X.ndim == 1:
                    self.X = self.X.reshape(-1, len(self.domain.attributes))

                self.Y = get_columns(row_indices, conversion.class_vars, n_rows,
                                     is_sparse=conversion.sparse_Y)

                dtype = np.float64
                if any(isinstance(var, StringVariable) for var in domain.metas):
                    dtype = np.object
                self.metas = get_columns(row_indices, conversion.metas,
                                         n_rows, dtype,
                                         is_sparse=conversion.sparse_metas)
                if self.metas.ndim == 1:
                    self.metas = self.metas.reshape(-1, len(self.domain.metas))
                if source.has_weights():
                    self.W = source.W[row_indices]
                else:
                    self.W = np.empty((n_rows, 0))
                self.name = getattr(source, 'name', '')
                if hasattr(source, 'ids'):
                    self.ids = source.ids[row_indices]
                else:
                    cls._init_ids(self)
                self.attributes = getattr(source, 'attributes', {})
                _conversion_cache[(id(domain), id(source))] = self
                return self
            finally:
                if new_cache:
                    _conversion_cache = None

    def transform(self, domain):
        """
        Construct a table with a different domain.

        The new table keeps the row ids and other information. If the table
        is a subclass of :obj:`Table`, the resulting table will be of the same
        type.

        In a typical scenario, an existing table is augmented with a new
        column by ::

            domain = Domain(old_domain.attributes + [new_attribute],
                            old_domain.class_vars,
                            old_domain.metas)
            table = data.transform(domain)
            table[:, new_attribute] = new_column

        Args:
            domain (Domain): new domain

        Returns:
            A new table
        """
        return type(self).from_table(domain, self)

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
        self = cls()
        self.domain = source.domain
        self.X = source.X[row_indices]
        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, len(self.domain.attributes))
        self.Y = source._Y[row_indices]
        self.metas = source.metas[row_indices]
        if self.metas.ndim == 1:
            self.metas = self.metas.reshape(-1, len(self.domain.metas))
        self.W = source.W[row_indices]
        self.name = getattr(source, 'name', '')
        self.ids = np.array(source.ids[row_indices])
        self.attributes = getattr(source, 'attributes', {})
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
        X, Y, W = _check_arrays(X, Y, W, dtype='float64')
        metas, = _check_arrays(metas, dtype=object)

        if Y is not None and Y.ndim == 1:
            Y = Y.reshape(Y.shape[0], 1)
        if domain is None:
            domain = Domain.from_numpy(X, Y, metas)

        if Y is None:
            if sp.issparse(X):
                Y = np.empty((X.shape[0], 0), dtype=np.float64)
            else:
                Y = X[:, len(domain.attributes):]
                X = X[:, :len(domain.attributes)]
        if metas is None:
            metas = np.empty((X.shape[0], 0), object)
        if W is None or W.size == 0:
            W = np.empty((X.shape[0], 0))
        else:
            W = W.reshape(W.size)

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

        self = cls()
        self.domain = domain
        self.X = X
        self.Y = Y
        self.metas = metas
        self.W = W
        self.n_rows = self.X.shape[0]
        cls._init_ids(self)
        self.attributes = {}
        return self

    @classmethod
    def from_list(cls, domain, rows, weights=None):
        if weights is not None and len(rows) != len(weights):
            raise ValueError("mismatching number of instances and weights")
        self = cls.from_domain(domain, len(rows), weights is not None)
        attrs, classes = domain.attributes, domain.class_vars
        metas = domain.metas
        nattrs, ncls = len(domain.attributes), len(domain.class_vars)
        for i, row in enumerate(rows):
            if isinstance(row, Instance):
                row = row.list
            for j, (var, val) in enumerate(zip(attrs, row)):
                self.X[i, j] = var.to_val(val)
            for j, (var, val) in enumerate(zip(classes, row[nattrs:])):
                self._Y[i, j] = var.to_val(val)
            for j, (var, val) in enumerate(zip(metas, row[nattrs + ncls:])):
                self.metas[i, j] = var.to_val(val)
        if weights is not None:
            self.W = np.array(weights)
        return self

    @classmethod
    def _init_ids(cls, obj):
        with cls._next_instance_lock:
            obj.ids = np.array(range(cls._next_instance_id, cls._next_instance_id + obj.X.shape[0]))
            cls._next_instance_id += obj.X.shape[0]

    @classmethod
    def new_id(cls):
        with cls._next_instance_lock:
            id = cls._next_instance_id
            cls._next_instance_id += 1
            return id

    def save(self, filename):
        """
        Save a data table to a file. The path can be absolute or relative.

        :param filename: File name
        :type filename: str
        """
        ext = os.path.splitext(filename)[1]
        from Orange.data.io import FileFormat
        writer = FileFormat.writers.get(ext)
        if not writer:
            desc = FileFormat.names.get(ext)
            if desc:
                raise IOError(
                    "Writing of {}s is not supported".format(desc.lower()))
            else:
                raise IOError("Unknown file name extension.")
        writer.write_file(filename, self)

    @classmethod
    def from_file(cls, filename, sheet=None):
        """
        Read a data table from a file. The path can be absolute or relative.

        :param filename: File name
        :type filename: str
        :param sheet: Sheet in a file (optional)
        :type sheet: str
        :return: a new data table
        :rtype: Orange.data.Table
        """
        from Orange.data.io import FileFormat

        absolute_filename = FileFormat.locate(filename, dataset_dirs)
        reader = FileFormat.get_reader(absolute_filename)
        reader.select_sheet(sheet)
        data = reader.read()

        # Readers return plain table. Make sure to cast it to appropriate
        # (subclass) type
        if cls != data.__class__:
            data = cls(data)

        # no need to call _init_ids as fuctions from .io already
        # construct a table with .ids

        data.__file__ = absolute_filename
        return data

    @classmethod
    def from_url(cls, url):
        from Orange.data.io import UrlReader
        reader = UrlReader(url)
        data = reader.read()
        if cls != data.__class__:
            data = cls(data)
        return data

    # Helper function for __setitem__ and insert:
    # Set the row of table data matrices
    # noinspection PyProtectedMember
    def _set_row(self, example, row):
        domain = self.domain
        if isinstance(example, Instance):
            if example.domain == domain:
                if isinstance(example, RowInstance):
                    self.X[row] = example._x
                    self._Y[row] = example._y
                else:
                    self.X[row] = example._x
                    self._Y[row] = example._y
                self.metas[row] = example._metas
                return

            self.X[row], self._Y[row], self.metas[row] = \
                self.domain.convert(example)
            try:
                self.ids[row] = example.id
            except:
                with type(self)._next_instance_lock:
                    self.ids[row] = type(self)._next_instance_id
                    type(self)._next_instance_id += 1

        else:
            self.X[row] = [var.to_val(val)
                           for var, val in zip(domain.attributes, example)]
            self._Y[row] = [var.to_val(val)
                            for var, val in
                            zip(domain.class_vars,
                                example[len(domain.attributes):])]
            self.metas[row] = np.array([var.Unknown for var in domain.metas],
                                       dtype=object)

    def _check_all_dense(self):
        return all(x in (Storage.DENSE, Storage.MISSING)
                   for x in (self.X_density(), self.Y_density(),
                             self.metas_density()))

    # A helper function for extend and insert
    # Resize X, Y, metas and W.
    def _resize_all(self, new_length):
        old_length = self.X.shape[0]
        if old_length == new_length:
            return
        if not self._check_all_dense():
            raise ValueError("Tables with sparse data cannot be resized")
        try:
            self.X.resize(new_length, self.X.shape[1])
            self._Y.resize(new_length, self._Y.shape[1])
            self.metas.resize(new_length, self.metas.shape[1])
            if self.W.ndim == 2:
                self.W.resize((new_length, 0))
            else:
                self.W.resize(new_length)
            self.ids.resize(new_length)
        except Exception:
            if self.X.shape[0] == new_length:
                self.X.resize(old_length, self.X.shape[1])
            if self._Y.shape[0] == new_length:
                self._Y.resize(old_length, self._Y.shape[1])
            if self.metas.shape[0] == new_length:
                self.metas.resize(old_length, self.metas.shape[1])
            if self.W.shape[0] == new_length:
                if self.W.ndim == 2:
                    self.W.resize((old_length, 0))
                else:
                    self.W.resize(old_length)
            if self.ids.shape[0] == new_length:
                self.ids.resize(old_length)
            raise

    def __getitem__(self, key):
        if isinstance(key, Integral):
            return RowInstance(self, key)
        if not isinstance(key, tuple):
            return self.from_table_rows(self, key)

        if len(key) != 2:
            raise IndexError("Table indices must be one- or two-dimensional")

        row_idx, col_idx = key
        if isinstance(row_idx, Integral):
            if isinstance(col_idx, (str, Integral, Variable)):
                col_idx = self.domain.index(col_idx)
                var = self.domain[col_idx]
                if 0 <= col_idx < len(self.domain.attributes):
                    return Value(var, self.X[row_idx, col_idx])
                elif col_idx >= len(self.domain.attributes):
                    return Value(
                        var,
                        self._Y[row_idx,
                                col_idx - len(self.domain.attributes)])
                elif col_idx < 0:
                    return Value(var, self.metas[row_idx, -1 - col_idx])
            else:
                row_idx = [row_idx]

        # multiple rows OR single row but multiple columns:
        # construct a new table
        attributes, col_indices = self.domain._compute_col_indices(col_idx)
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
            domain = Domain(r_attrs, r_classes, r_metas)
        else:
            domain = self.domain
        return self.from_table(domain, self, row_idx)

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
        if isinstance(row_idx, Integral):
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
                if not isinstance(value, Integral):
                    value = self.domain[col_idx].to_val(value)
                if not isinstance(col_idx, Integral):
                    col_idx = self.domain.index(col_idx)
                if col_idx >= 0:
                    if col_idx < self.X.shape[1]:
                        self.X[row_idx, col_idx] = value
                    else:
                        self._Y[row_idx, col_idx - self.X.shape[1]] = value
                else:
                    self.metas[row_idx, -1 - col_idx] = value

        # multiple rows, multiple columns
        attributes, col_indices = self.domain._compute_col_indices(col_idx)
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
                    self._Y[row_idx, col - n_attrs] = var.to_val(value)
                else:
                    self.metas[row_idx, -1 - col] = var.to_val(value)
        else:
            attr_cols = np.fromiter(
                (col for col in col_indices if 0 <= col < n_attrs), int)
            class_cols = np.fromiter(
                (col - n_attrs for col in col_indices if col >= n_attrs), int)
            meta_cols = np.fromiter(
                (-1 - col for col in col_indices if col < 0), int)
            if value is None:
                value = Unknown

            if not isinstance(value, (Real, np.ndarray)) and \
                    (len(attr_cols) or len(class_cols)):
                raise TypeError(
                    "Ordinary attributes can only have primitive values")
            if len(attr_cols):
                self.X[row_idx, attr_cols] = value
            if len(class_cols):
                self._Y[row_idx, class_cols] = value
            if len(meta_cols):
                self.metas[row_idx, meta_cols] = value

    def __delitem__(self, key):
        if not self._check_all_dense():
            raise ValueError("Rows of sparse data cannot be deleted")
        if key is ...:
            key = range(len(self))
        self.X = np.delete(self.X, key, axis=0)
        self.Y = np.delete(self._Y, key, axis=0)
        self.metas = np.delete(self.metas, key, axis=0)
        self.W = np.delete(self.W, key, axis=0)
        self.ids = np.delete(self.ids, key, axis=0)

    def __len__(self):
        return self.X.shape[0]

    def __str__(self):
        return "[" + ",\n ".join(str(ex) for ex in self) + "]"

    def __repr__(self):
        head = 5
        if self.is_sparse():
            head = min(self.X.shape[0], head)
        s = "[" + ",\n ".join(repr(ex) for ex in self[:head])
        if len(self) > head:
            s += ",\n ..."
        s += "\n]"
        return s

    def clear(self):
        """Remove all rows from the table."""
        if not self._check_all_dense():
            raise ValueError("Tables with sparse data cannot be cleared")
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
        self.ensure_copy()  # ensure that numpy arrays are single-segment for resize
        self._resize_all(len(self) + 1)
        if row < len(self):
            self.X[row + 1:] = self.X[row:-1]
            self._Y[row + 1:] = self._Y[row:-1]
            self.metas[row + 1:] = self.metas[row:-1]
            self.W[row + 1:] = self.W[row:-1]
            self.ids[row + 1:] = self.ids[row:-1]
        try:
            self._set_row(instance, row)
            if self.W.shape[-1]:
                self.W[row] = 1
        except Exception:
            self.X[row:-1] = self.X[row + 1:]
            self._Y[row:-1] = self._Y[row + 1:]
            self.metas[row:-1] = self.metas[row + 1:]
            self.W[row:-1] = self.W[row + 1:]
            self.ids[row:-1] = self.ids[row + 1:]
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
        if isinstance(instances, Table) and instances.domain == self.domain:
            self.X = vstack((self.X, instances.X))
            self._Y = vstack((self._Y, instances._Y))
            self.metas = vstack((self.metas, instances.metas))
            self.W = vstack((self.W, instances.W))
            self.ids = hstack((self.ids, instances.ids))
        else:
            try:
                old_length = len(self)
                self._resize_all(old_length + len(instances))
                for i, example in enumerate(instances):
                    self[old_length + i] = example
                    try:
                        self.ids[old_length + i] = example.id
                    except AttributeError:
                        self.ids[old_length + i] = self.new_id()
            except Exception:
                self._resize_all(old_length)
                raise

    @staticmethod
    def concatenate(tables, axis=1):
        """Return concatenation of `tables` by `axis`."""
        if not tables:
            raise ValueError('need at least one table to concatenate')
        if len(tables) == 1:
            return tables[0].copy()
        CONCAT_ROWS, CONCAT_COLS = 0, 1
        if axis == CONCAT_ROWS:
            table = tables[0].copy()
            for t in tables[1:]:
                table.extend(t)
            return table
        elif axis == CONCAT_COLS:
            if reduce(operator.iand,
                      (set(map(operator.attrgetter('name'),
                               chain(t.domain.variables, t.domain.metas)))
                       for t in tables)):
                raise ValueError('Concatenating two domains with variables '
                                 'with same name is undefined')
            domain = Domain(flatten(t.domain.attributes for t in tables),
                            flatten(t.domain.class_vars for t in tables),
                            flatten(t.domain.metas for t in tables))

            def ndmin(A):
                return A if A.ndim > 1 else A.reshape(A.shape[0], 1)

            table = Table.from_numpy(domain,
                                     np.hstack(tuple(ndmin(t.X) for t in tables)),
                                     np.hstack(tuple(ndmin(t.Y) for t in tables)),
                                     np.hstack(tuple(ndmin(t.metas) for t in tables)),
                                     np.hstack(tuple(ndmin(t.W) for t in tables)))
            return table
        raise ValueError('axis {} out of bounds [0, 2)'.format(axis))

    def is_view(self):
        """
        Return `True` if all arrays represent a view referring to another table
        """
        return ((not self.X.shape[-1] or self.X.base is not None) and
                (not self._Y.shape[-1] or self._Y.base is not None) and
                (not self.metas.shape[-1] or self.metas.base is not None) and
                (not self._weights.shape[-1] or self.W.base is not None))

    def is_copy(self):
        """
        Return `True` if the table owns its data
        """
        return ((not self.X.shape[-1] or self.X.base is None) and
                (self._Y.base is None) and
                (self.metas.base is None) and
                (self.W.base is None))

    def is_sparse(self):
        """
        Return `True` if the table stores data in sparse format
        """
        return any(sp.issparse(i) for i in [self.X, self.Y, self.metas])

    def ensure_copy(self):
        """
        Ensure that the table owns its data; copy arrays when necessary.
        """
        def is_view(x):
            # Sparse matrices don't have views like numpy arrays. Since indexing on
            # them creates copies in constructor we can skip this check here.
            return not sp.issparse(x) and x.base is not None

        if is_view(self.X):
            self.X = self.X.copy()
        if is_view(self._Y):
            self._Y = self._Y.copy()
        if is_view(self.metas):
            self.metas = self.metas.copy()
        if is_view(self.W):
            self.W = self.W.copy()

    def copy(self):
        """
        Return a copy of the table
        """
        t = self.__class__(self)
        t.ensure_copy()
        return t

    @staticmethod
    def __determine_density(data):
        if data is None:
            return Storage.Missing
        if data is not None and sp.issparse(data):
            return Storage.SPARSE_BOOL if (data.data == 1).all() else Storage.SPARSE
        else:
            return Storage.DENSE

    def X_density(self):
        if not hasattr(self, "_X_density"):
            self._X_density = self.__determine_density(self.X)
        return self._X_density

    def Y_density(self):
        if not hasattr(self, "_Y_density"):
            self._Y_density = self.__determine_density(self._Y)
        return self._Y_density

    def metas_density(self):
        if not hasattr(self, "_metas_density"):
            self._metas_density = self.__determine_density(self.metas)
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
        missing_x = not sp.issparse(self.X) and bn.anynan(self.X)   # do not check for sparse X
        return missing_x or bn.anynan(self._Y)

    def has_missing_class(self):
        """Return `True` if there are any missing class values."""
        return bn.anynan(self._Y)

    def checksum(self, include_metas=True):
        # TODO: zlib.adler32 does not work for numpy arrays with dtype object
        # (after pickling and unpickling such arrays, checksum changes)
        # Why, and should we fix it or remove it?
        """Return a checksum over X, Y, metas and W."""
        cs = zlib.adler32(np.ascontiguousarray(self.X))
        cs = zlib.adler32(np.ascontiguousarray(self._Y), cs)
        if include_metas:
            cs = zlib.adler32(np.ascontiguousarray(self.metas), cs)
        cs = zlib.adler32(np.ascontiguousarray(self.W), cs)
        return cs

    def shuffle(self):
        """Randomly shuffle the rows of the table."""
        if not self._check_all_dense():
            raise ValueError("Rows of sparse data cannot be shuffled")
        ind = np.arange(self.X.shape[0])
        np.random.shuffle(ind)
        self.X = self.X[ind]
        self._Y = self._Y[ind]
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

        if not isinstance(index, Integral):
            index = self.domain.index(index)
        if index >= 0:
            if index < self.X.shape[1]:
                return rx(self.X[:, index])
            else:
                return rx(self._Y[:, index - self.X.shape[1]])
        else:
            return rx(self.metas[:, -1 - index])

    def _filter_is_defined(self, columns=None, negate=False):
        if columns is None:
            if sp.issparse(self.X):
                remove = (self.X.indptr[1:] !=
                          self.X.indptr[-1:] + self.X.shape[1])
            else:
                remove = bn.anynan(self.X, axis=1)
            if sp.issparse(self._Y):
                remove = np.logical_or(remove, self._Y.indptr[1:] !=
                                       self._Y.indptr[-1:] + self._Y.shape[1])
            else:
                remove = np.logical_or(remove, bn.anynan(self._Y, axis=1))
        else:
            remove = np.zeros(len(self), dtype=bool)
            for column in columns:
                col, sparse = self.get_column_view(column)
                if sparse:
                    remove = np.logical_or(remove, col == 0)
                else:
                    remove = np.logical_or(remove, bn.anynan([col], axis=0))
        retain = remove if negate else np.logical_not(remove)
        return self.from_table_rows(self, retain)

    def _filter_has_class(self, negate=False):
        if sp.issparse(self._Y):
            if negate:
                retain = (self._Y.indptr[1:] !=
                          self._Y.indptr[-1:] + self._Y.shape[1])
            else:
                retain = (self._Y.indptr[1:] ==
                          self._Y.indptr[-1:] + self._Y.shape[1])
        else:
            retain = bn.anynan(self._Y, axis=1)
            if not negate:
                retain = np.logical_not(retain)
        return self.from_table_rows(self, retain)

    def _filter_same_value(self, column, value, negate=False):
        if not isinstance(value, Real):
            value = self.domain[column].to_val(value)
        sel = self.get_column_view(column)[0] == value
        if negate:
            sel = np.logical_not(sel)
        return self.from_table_rows(self, sel)

    def _filter_values(self, filter):
        selection = self._values_filter_to_indicator(filter)
        return self.from_table(self.domain, self, selection)

    def _values_filter_to_indicator(self, filter):
        """Return selection of rows matching the filter conditions

        Handles conjunction/disjunction and negate modifiers

        Parameters
        ----------
        filter: Values object containing the conditions

        Returns
        -------
        A 1d bool array. len(result) == len(self)
        """
        from Orange.data.filter import Values

        if isinstance(filter, Values):
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
            selection = self._filter_to_indicator(f)

            if conjunction:
                sel *= selection
            else:
                sel += selection

        if filter.negate:
            sel = ~sel
        return sel

    def _filter_to_indicator(self, filter):
        """Return selection of rows that match the condition.

        Parameters
        ----------
        filter: ValueFilter describing the condition

        Returns
        -------
        A 1d bool array. len(result) == len(self)
        """
        from Orange.data.filter import (
            FilterContinuous, FilterDiscrete, FilterRegex, FilterString,
            FilterStringList, Values
        )
        if isinstance(filter, Values):
            return self._values_filter_to_indicator(filter)

        col = self.get_column_view(filter.column)[0]

        if isinstance(filter, FilterDiscrete):
            return self._discrete_filter_to_indicator(filter, col)

        if isinstance(filter, FilterContinuous):
            return self._continuous_filter_to_indicator(filter, col)

        if isinstance(filter, FilterString):
            return self._string_filter_to_indicator(filter, col)

        if isinstance(filter, FilterStringList):
            if not filter.case_sensitive:
                col = np.char.lower(np.array(col, dtype=str))
                vals = [val.lower() for val in filter.values]
            else:
                vals = filter.values
            return reduce(operator.add, (col == val for val in vals))

        if isinstance(filter, FilterRegex):
            return np.vectorize(filter)(col)

        raise TypeError("Invalid filter")

    def _discrete_filter_to_indicator(self, filter, col):
        """Return selection of rows matched by the given discrete filter.

        Parameters
        ----------
        filter: FilterDiscrete
        col: np.ndarray

        Returns
        -------
        A 1d bool array. len(result) == len(self)
        """
        if filter.values is None:  # <- is defined filter
            col = col.astype(float)
            return ~np.isnan(col)

        sel = np.zeros(len(self), dtype=bool)
        for val in filter.values:
            if not isinstance(val, Real):
                val = self.domain[filter.column].to_val(val)
            sel += (col == val)
        return sel

    def _continuous_filter_to_indicator(self, filter, col):
        """Return selection of rows matched by the given continuous filter.

        Parameters
        ----------
        filter: FilterContinuous
        col: np.ndarray

        Returns
        -------
        A 1d bool array. len(result) == len(self)
        """
        if filter.oper == filter.IsDefined:
            col = col.astype(float)
            return ~np.isnan(col)

        return self._range_filter_to_indicator(filter, col, filter.min, filter.max)

    def _string_filter_to_indicator(self, filter, col):
        """Return selection of rows matched by the given string filter.

        Parameters
        ----------
        filter: FilterString
        col: np.ndarray

        Returns
        -------
        A 1d bool array. len(result) == len(self)
        """
        if filter.oper == filter.IsDefined:
            return col.astype(bool)

        col = col.astype(str)
        fmin = filter.min or ""
        fmax = filter.max or ""

        if not filter.case_sensitive:
            # convert all to lower case
            col = np.char.lower(col)
            fmin = fmin.lower()
            fmax = fmax.lower()

        if filter.oper == filter.Contains:
            return np.fromiter((fmin in e for e in col),
                               dtype=bool)
        if filter.oper == filter.StartsWith:
            return np.fromiter((e.startswith(fmin) for e in col),
                               dtype=bool)
        if filter.oper == filter.EndsWith:
            return np.fromiter((e.endswith(fmin) for e in col),
                               dtype=bool)

        return self._range_filter_to_indicator(filter, col, fmin, fmax)

    @staticmethod
    def _range_filter_to_indicator(filter, col, fmin, fmax):
        if filter.oper == filter.Equal:
            return col == fmin
        if filter.oper == filter.NotEqual:
            return col != fmin
        if filter.oper == filter.Less:
            return col < fmin
        if filter.oper == filter.LessEqual:
            return col <= fmin
        if filter.oper == filter.Greater:
            return col > fmin
        if filter.oper == filter.GreaterEqual:
            return col >= fmin
        if filter.oper == filter.Between:
            return (col >= fmin) * (col <= fmax)
        if filter.oper == filter.Outside:
            return (col < fmin) + (col > fmax)

        raise TypeError("Invalid operator")

    def _compute_basic_stats(self, columns=None,
                             include_metas=False, compute_variance=False):
        if compute_variance:
            raise NotImplementedError("computation of variance is "
                                      "not implemented yet")
        W = self.W if self.has_weights() else None
        rr = []
        stats = []
        if not columns:
            if self.domain.attributes:
                rr.append(fast_stats(self.X, W))
            if self.domain.class_vars:
                rr.append(fast_stats(self._Y, W))
            if include_metas and self.domain.metas:
                rr.append(fast_stats(self.metas, W))
            if len(rr):
                stats = np.vstack(tuple(rr))
        else:
            nattrs = len(self.domain.attributes)
            for column in columns:
                c = self.domain.index(column)
                if 0 <= c < nattrs:
                    S = fast_stats(self.X[:, [c]], W and W[:, [c]])
                elif c >= nattrs:
                    S = fast_stats(self._Y[:, [c-nattrs]], W and W[:, [c-nattrs]])
                else:
                    S = fast_stats(self.metas[:, [-1-c]], W and W[:, [-1-c]])
                stats.append(S[0])
        return stats

    def _compute_distributions(self, columns=None):
        if columns is None:
            columns = range(len(self.domain.variables))
        else:
            columns = [self.domain.index(var) for var in columns]

        distributions = []
        if sp.issparse(self.X):
            self.X = self.X.tocsc()

        W = self.W.ravel() if self.has_weights() else None

        for col in columns:
            variable = self.domain[col]

            # Select the correct data column from X, Y or metas
            if 0 <= col < self.X.shape[1]:
                x = self.X[:, col]
            elif col < 0:
                x = self.metas[:, col * (-1) - 1]
                if np.issubdtype(x.dtype, np.dtype(object)):
                    x = x.astype(float)
            else:
                x = self._Y[:, col - self.X.shape[1]]

            if variable.is_discrete:
                dist, unknowns = bincount(x, weights=W, max_val=len(variable.values) - 1)
            elif not x.shape[0]:
                dist, unknowns = np.zeros((2, 0)), 0
            else:
                if W is not None:
                    if sp.issparse(x):
                        arg_sort = np.argsort(x.data)
                        ranks = x.indices[arg_sort]
                        vals = np.vstack((x.data[arg_sort], W[ranks]))
                    else:
                        ranks = np.argsort(x)
                        vals = np.vstack((x[ranks], W[ranks]))
                else:
                    x_values = x.data if sp.issparse(x) else x
                    vals = np.ones((2, x_values.shape[0]))
                    vals[0, :] = x_values
                    vals[0, :].sort()

                dist = np.array(_valuecount.valuecount(vals))
                # If sparse, then 0s will not be counted with `valuecount`, so
                # we have to add them to the result manually.
                if sp.issparse(x) and sparse_has_implicit_zeros(x):
                    if W is not None:
                        zero_weights = sparse_implicit_zero_weights(x, W).sum()
                    else:
                        zero_weights = sparse_count_implicit_zeros(x)
                    zero_vec = [0, zero_weights]
                    dist = np.insert(dist, np.searchsorted(dist[0], 0), zero_vec, axis=1)
                # Since `countnans` assumes vector shape to be (1, n) and `x`
                # shape is (n, 1), we pass the transpose
                unknowns = countnans(x.T, W)
            distributions.append((dist, unknowns))

        return distributions

    def _compute_contingency(self, col_vars=None, row_var=None):
        n_atts = self.X.shape[1]

        if col_vars is None:
            col_vars = range(len(self.domain.variables))
        else:
            col_vars = [self.domain.index(var) for var in col_vars]
        if row_var is None:
            row_var = self.domain.class_var
            if row_var is None:
                raise ValueError("No row variable")

        row_desc = self.domain[row_var]
        if not row_desc.is_discrete:
            raise TypeError("Row variable must be discrete")
        row_indi = self.domain.index(row_var)
        n_rows = len(row_desc.values)
        if 0 <= row_indi < n_atts:
            row_data = self.X[:, row_indi]
        elif row_indi < 0:
            row_data = self.metas[:, -1 - row_indi]
        else:
            row_data = self._Y[:, row_indi - n_atts]

        W = self.W if self.has_weights() else None
        nan_inds = None

        col_desc = [self.domain[var] for var in col_vars]
        col_indi = [self.domain.index(var) for var in col_vars]

        if any(not (var.is_discrete or var.is_continuous)
               for var in col_desc):
            raise ValueError("contingency can be computed only for discrete "
                             "and continuous values")

        if row_data.dtype.kind != "f": #meta attributes can be stored as type object
            row_data = row_data.astype(float)

        unknown_rows = countnans(row_data)
        if unknown_rows:
            nan_inds = np.isnan(row_data)
            row_data = row_data[~nan_inds]
            if W:
                W = W[~nan_inds]
                unknown_rows = np.sum(W[nan_inds])

        contingencies = [None] * len(col_desc)
        for arr, f_cond, f_ind in (
                (self.X, lambda i: 0 <= i < n_atts, lambda i: i),
                (self._Y, lambda i: i >= n_atts, lambda i: i - n_atts),
                (self.metas, lambda i: i < 0, lambda i: -1 - i)):

            if nan_inds is not None:
                arr = arr[~nan_inds]

            arr_indi = [e for e, ind in enumerate(col_indi) if f_cond(ind)]

            vars = [(e, f_ind(col_indi[e]), col_desc[e]) for e in arr_indi]
            disc_vars = [v for v in vars if v[2].is_discrete]
            if disc_vars:
                if sp.issparse(arr):
                    max_vals = max(len(v[2].values) for v in disc_vars)
                    disc_indi = {i for _, i, _ in disc_vars}
                    mask = [i in disc_indi for i in range(arr.shape[1])]
                    conts, nans = contingency(arr, row_data, max_vals - 1,
                                              n_rows - 1, W, mask)
                    for col_i, arr_i, var in disc_vars:
                        n_vals = len(var.values)
                        contingencies[col_i] = (conts[arr_i][:, :n_vals],
                                                nans[arr_i])
                else:
                    for col_i, arr_i, var in disc_vars:
                        contingencies[col_i] = contingency(
                            arr[:, arr_i].astype(float),
                            row_data, len(var.values) - 1, n_rows - 1, W)

            cont_vars = [v for v in vars if v[2].is_continuous]
            if cont_vars:

                classes = row_data.astype(dtype=np.intp)
                if W is not None:
                    W = W.astype(dtype=np.float64)
                if sp.issparse(arr):
                    arr = sp.csc_matrix(arr)

                for col_i, arr_i, _ in cont_vars:
                    if sp.issparse(arr):
                        col_data = arr.data[arr.indptr[arr_i]:arr.indptr[arr_i + 1]]
                        rows = arr.indices[arr.indptr[arr_i]:arr.indptr[arr_i + 1]]
                        W_ = None if W is None else W[rows]
                        classes_ = classes[rows]
                    else:
                        col_data, W_, classes_ = arr[:, arr_i], W, classes

                    col_data = col_data.astype(dtype=np.float64)
                    U, C, unknown = _contingency.contingency_floatarray(
                        col_data, classes_, n_rows, W_)
                    contingencies[col_i] = ([U, C], unknown)

        return contingencies, unknown_rows

    @classmethod
    def transpose(cls, table, feature_names_column="", meta_attr_name="Feature name",
                  feature_name="Feature"):
        """
        Transpose the table.

        :param table: Table - table to transpose
        :param feature_names_column: str - name of (String) meta attribute to
            use for feature names
        :param meta_attr_name: str - name of new meta attribute into which
            feature names are mapped
        :return: Table - transposed table
        """

        self = cls()
        n_cols, self.n_rows = table.X.shape
        old_domain = table.attributes.get("old_domain")

        # attributes
        # - classes and metas to attributes of attributes
        # - arbitrary meta column to feature names
        self.X = table.X.T
        attributes = [ContinuousVariable(str(row[feature_names_column]))
                      for row in table] if feature_names_column else \
            [ContinuousVariable(feature_name + " " + str(i + 1).zfill(
                int(np.ceil(np.log10(n_cols))))) for i in range(n_cols)]
        if old_domain is not None and feature_names_column:
            for i, _ in enumerate(attributes):
                if attributes[i].name in old_domain:
                    var = old_domain[attributes[i].name]
                    attr = ContinuousVariable(var.name) if var.is_continuous \
                        else DiscreteVariable(var.name, var.values)
                    attr.attributes = var.attributes.copy()
                    attributes[i] = attr

        def set_attributes_of_attributes(_vars, _table):
            for i, variable in enumerate(_vars):
                if variable.name == feature_names_column:
                    continue
                for j, row in enumerate(_table):
                    value = variable.repr_val(row) if np.isscalar(row) \
                        else row[i] if isinstance(row[i], str) \
                        else variable.repr_val(row[i])

                    if value not in MISSING_VALUES:
                        attributes[j].attributes[variable.name] = value

        set_attributes_of_attributes(table.domain.class_vars, table.Y)
        set_attributes_of_attributes(table.domain.metas, table.metas)

        # weights
        self.W = np.empty((self.n_rows, 0))

        def get_table_from_attributes_of_attributes(_vars, _dtype=float):
            T = np.empty((self.n_rows, len(_vars)), dtype=_dtype)
            for i, _attr in enumerate(table.domain.attributes):
                for j, _var in enumerate(_vars):
                    val = str(_attr.attributes.get(_var.name, ""))
                    if not _var.is_string:
                        val = np.nan if val in MISSING_VALUES else \
                            _var.values.index(val) if \
                                _var.is_discrete else float(val)
                    T[i, j] = val
            return T

        # class_vars - attributes of attributes to class - from old domain
        class_vars = []
        if old_domain is not None:
            class_vars = old_domain.class_vars
        self.Y = get_table_from_attributes_of_attributes(class_vars)

        # metas
        # - feature names and attributes of attributes to metas
        self.metas, metas = np.empty((self.n_rows, 0), dtype=object), []
        if meta_attr_name not in [m.name for m in table.domain.metas] and \
                table.domain.attributes:
            self.metas = np.array([[a.name] for a in table.domain.attributes],
                                  dtype=object)
            metas.append(StringVariable(meta_attr_name))

        names = chain.from_iterable(list(attr.attributes)
                                    for attr in table.domain.attributes)
        names = sorted(set(names) - {var.name for var in class_vars})

        def guessed_var(i, var_name):
            orig_vals = M[:, i]
            val_map, vals, var_type = Orange.data.io.guess_data_type(orig_vals)
            values, variable = Orange.data.io.sanitize_variable(
                val_map, vals, orig_vals, var_type, {}, name=var_name)
            M[:, i] = values
            return variable

        _metas = [StringVariable(n) for n in names]
        if old_domain is not None:
            _metas = [m for m in old_domain.metas if m.name != meta_attr_name]
        M = get_table_from_attributes_of_attributes(_metas, _dtype=object)
        if old_domain is None:
            _metas = [guessed_var(i, m.name) for i, m in enumerate(_metas)]
        if _metas:
            self.metas = np.hstack((self.metas, M))
            metas.extend(_metas)

        self.domain = Domain(attributes, class_vars, metas)
        cls._init_ids(self)
        self.attributes = table.attributes.copy()
        self.attributes["old_domain"] = table.domain
        return self

    def to_sparse(self, sparse_attributes=True, sparse_class=False,
                  sparse_metas=False):
        def sparsify(features):
            for f in features:
                f.sparse = True

        new_domain = self.domain.copy()

        if sparse_attributes:
            sparsify(new_domain.attributes)
        if sparse_class:
            sparsify(new_domain.class_vars)
        if sparse_metas:
            sparsify(new_domain.metas)
        return self.transform(new_domain)

    def to_dense(self, dense_attributes=True, dense_class=True,
                 dense_metas=True):
        def densify(features):
            for f in features:
                f.sparse = False

        new_domain = self.domain.copy()

        if dense_attributes:
            densify(new_domain.attributes)
        if dense_class:
            densify(new_domain.class_vars)
        if dense_metas:
            densify(new_domain.metas)
        t = self.transform(new_domain)
        t.ids = self.ids    # preserve indices
        return t


def _check_arrays(*arrays, dtype=None):
    checked = []
    if not len(arrays):
        return checked

    def ninstances(array):
        if hasattr(array, "shape"):
            return array.shape[0]
        else:
            return len(array) if array is not None else 0

    shape_1 = ninstances(arrays[0])

    for array in arrays:
        if array is None:
            checked.append(array)
            continue

        if ninstances(array) != shape_1:
            raise ValueError("Leading dimension mismatch (%d != %d)"
                             % (ninstances(array), shape_1))

        if sp.issparse(array):
            array.data = np.asarray(array.data)
            has_inf = _check_inf(array.data)
        else:
            if dtype is not None:
                array = np.asarray(array, dtype=dtype)
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


def _subarray(arr, rows, cols):
    rows = _optimize_indices(rows, arr.shape[0])
    cols = _optimize_indices(cols, arr.shape[1])
    return arr[_rxc_ix(rows, cols)]


def _optimize_indices(indices, maxlen):
    """
    Convert integer indices to slice if possible. It only converts increasing
    integer ranges with positive steps and valid starts and ends.
    Only convert valid ends so that invalid ranges will still raise
    an exception.

    Allows numpy to reuse the data array, because it defaults to copying
    if given indices.

    Parameters
    ----------
    indices : 1D sequence, slice or Ellipsis
    """
    if isinstance(indices, slice):
        return indices

    if indices is ...:
        return slice(None, None, 1)

    if len(indices) >= 1:
        indices = np.asarray(indices)
        if indices.dtype != np.bool:
            begin = indices[0]
            end = indices[-1]
            steps = np.diff(indices) if len(indices) > 1 else np.array([1])
            step = steps[0]

            # continuous ranges with constant step and valid start and stop index can be slices
            if np.all(steps == step) and step > 0 and begin >= 0 and end < maxlen:
                return slice(begin, end + step, step)

    return indices


def _rxc_ix(rows, cols):
    """
    Construct an index object to index the `rows` x `cols` cross product.

    Rows and columns can be a 1d bool or int sequence, or a slice.
    The later is a convenience and is interpreted the same
    as `slice(None, None, -1)`

    Parameters
    ----------
    rows : 1D sequence, slice
        Row indices.
    cols : 1D sequence, slice
        Column indices.

    See Also
    --------
    numpy.ix_

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(10).reshape(2, 5)
    >>> a[_rxc_ix([0, 1], [3, 4])]
    array([[3, 4],
           [8, 9]])
    >>> a[_rxc_ix([False, True], slice(None, None, 1))]
    array([[5, 6, 7, 8, 9]])

    """
    isslice = (isinstance(rows, slice), isinstance(cols, slice))
    if isslice == (True, True):
        return rows, cols
    elif isslice == (True, False):
        return rows, np.asarray(np.ix_(cols), int).ravel()
    elif isslice == (False, True):
        return np.asarray(np.ix_(rows), int).ravel(), cols
    else:
        r, c = np.ix_(rows, cols)
        return np.asarray(r, int), np.asarray(c, int)


def assure_domain_conversion_sparsity(target, source):
    """
    Assure that the table obeys the domain conversion's suggestions about sparsity.

    Args:
        target (Table): the target table.
        source (Table): the source table.

    Returns:
        Table: with fixed sparsity. The sparsity is set as it is recommended by domain conversion
            for transformation from source to the target domain.
    """
    conversion = target.domain.get_conversion(source.domain)
    match_density = [assure_array_dense, assure_array_sparse]
    target.X = match_density[conversion.sparse_X](target.X)
    target.Y = match_density[conversion.sparse_Y](target.Y)
    target.metas = match_density[conversion.sparse_metas](target.metas)
    return target
