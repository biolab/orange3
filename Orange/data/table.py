import operator
import os
import threading
import warnings
import weakref
import zlib
from collections import Iterable, Sequence, Sized
from functools import reduce
from itertools import chain
from numbers import Real, Integral
from threading import Lock

import bottleneck as bn
import numpy as np
from Orange.misc.collections import frozendict
from Orange.util import OrangeDeprecationWarning
from scipy import sparse as sp
from scipy.sparse import issparse, csc_matrix

import Orange.data  # import for io.py
from Orange.data import (
    _contingency, _valuecount,
    Domain, Variable, Storage, StringVariable, Unknown, Value, Instance,
    ContinuousVariable, DiscreteVariable, MISSING_VALUES,
    DomainConversion)
from Orange.data.util import SharedComputeValue, \
    assure_array_dense, assure_array_sparse, \
    assure_column_dense, assure_column_sparse, get_unique_names_duplicates
from Orange.statistics.util import bincount, countnans, contingency, \
    stats as fast_stats, sparse_has_implicit_zeros, sparse_count_implicit_zeros, \
    sparse_implicit_zero_weights

__all__ = ["dataset_dirs", "get_sample_datasets_dir", "RowInstance", "Table"]


def get_sample_datasets_dir():
    orange_data_table = os.path.dirname(__file__)
    dataset_dir = os.path.join(orange_data_table, '..', 'datasets')
    return os.path.realpath(dataset_dir)


dataset_dirs = ['', get_sample_datasets_dir()]


class _ThreadLocal(threading.local):
    def __init__(self):
        super().__init__()
        # Domain conversion cache used in Table.from_table. It is defined
        # here instead of as a class variable of a Table so that caching also works
        # with descendants of Table.
        self.conversion_cache = None


_thread_local = _ThreadLocal()


class DomainTransformationError(Exception):
    pass


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
class Table(Sequence, Storage):
    __file__ = None
    name = "untitled"

    domain = Domain([])
    X = _Y = metas = W = np.zeros((0, 0))
    X.setflags(write=False)
    ids = np.zeros(0)
    ids.setflags(write=False)
    attributes = frozendict()

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
        if sp.issparse(value):
            value = value.toarray()
        self._Y = value

    def __new__(cls, *args, **kwargs):
        def warn_deprecated(method):
            warnings.warn("Direct calls to Table's constructor are deprecated "
                          "and will be removed. Replace this call with "
                          f"Table.{method}", OrangeDeprecationWarning,
                          stacklevel=3)

        if not args:
            if not kwargs:
                return super().__new__(cls)
            else:
                raise TypeError("Table() must not be called directly")

        if isinstance(args[0], str):
            if len(args) > 1:
                raise TypeError("Table(name: str) expects just one argument")
            if args[0].startswith('https://') or args[0].startswith('http://'):
                return cls.from_url(args[0], **kwargs)
            else:
                return cls.from_file(args[0], **kwargs)

        elif isinstance(args[0], Table):
            if len(args) > 1:
                raise TypeError("Table(table: Table) expects just one argument")
            return cls.from_table(args[0].domain, args[0], **kwargs)
        elif isinstance(args[0], Domain):
            domain, args = args[0], args[1:]
            if not args:
                warn_deprecated("from_domain")
                return cls.from_domain(domain, **kwargs)
            if isinstance(args[0], Table):
                warn_deprecated("from_table")
                return cls.from_table(domain, *args, **kwargs)
            elif isinstance(args[0], list):
                warn_deprecated("from_list")
                return cls.from_list(domain, *args, **kwargs)
        else:
            warnings.warn("Omitting domain in a call to Table(X, Y, metas), is "
                          "deprecated and will be removed. "
                          "Call Table.from_numpy(None, X, Y, metas) instead.",
                          OrangeDeprecationWarning, stacklevel=2)
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

        def valid_refs(weakrefs):
            for r in weakrefs:
                if r() is None:
                    return False
            return True

        def get_columns(row_indices, src_cols, n_rows, dtype=np.float64,
                        is_sparse=False, variables=[]):
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

            # initialize arrays & set `match_density` for columns
            # F-order enables faster writing to the array while accessing and
            # matrix operations work with same speed (e.g. dot)
            a = None if is_sparse else np.zeros(
                (n_rows, len(src_cols)), order="F", dtype=dtype)
            data = []
            sp_col = []
            sp_row = []
            match_density = (
                assure_column_sparse if is_sparse else assure_column_dense
            )

            # converting to csc before instead of each column is faster
            # do not convert if not required
            if any([isinstance(x, int) for x in src_cols]):
                X = csc_matrix(source.X) if is_sparse else source.X
                Y = csc_matrix(source._Y) if is_sparse else source._Y

            shared_cache = _thread_local.conversion_cache
            for i, col in enumerate(src_cols):
                if col is None:
                    col_array = match_density(
                        np.full((n_rows, 1), variables[i].Unknown)
                    )
                elif not isinstance(col, Integral):
                    if isinstance(col, SharedComputeValue):
                        shared, weakrefs = shared_cache.get(
                            (id(col.compute_shared), id(source)),
                            (None, None)
                        )
                        if shared is None or not valid_refs(weakrefs):
                            shared, _ = shared_cache[(id(col.compute_shared), id(source))] = \
                                col.compute_shared(source), \
                                (weakref.ref(col.compute_shared), weakref.ref(source))

                        if row_indices is not ...:
                            col_array = match_density(
                                col(source, shared_data=shared)[row_indices])
                        else:
                            col_array = match_density(
                                col(source, shared_data=shared))
                    else:
                        if row_indices is not ...:
                            col_array = match_density(col(source)[row_indices])
                        else:
                            col_array = match_density(col(source))
                elif col < 0:
                    col_array = match_density(
                        source.metas[row_indices, -1 - col]
                    )
                elif col < n_src_attrs:
                    col_array = match_density(X[row_indices, col])
                else:
                    col_array = match_density(
                        Y[row_indices, col - n_src_attrs]
                    )

                if is_sparse:
                    # col_array should be coo matrix
                    data.append(col_array.data)
                    sp_col.append(np.full(len(col_array.data), i))
                    sp_row.append(col_array.indices)  # row indices should be same
                else:
                    a[:, i] = col_array

            if is_sparse:
                # creating csr directly would need plenty of manual work which
                # would probably slow down the process - conversion coo to csr
                # is fast
                a = sp.coo_matrix(
                    (np.hstack(data), (np.hstack(sp_row), np.hstack(sp_col))),
                    shape=(n_rows, len(src_cols)),
                    dtype=dtype
                )
                a = a.tocsr()

            return a

        new_cache = _thread_local.conversion_cache is None
        try:
            if new_cache:
                _thread_local.conversion_cache = {}
            else:
                cached, weakrefs = \
                    _thread_local.conversion_cache.get((id(domain), id(source)), (None, None))
                if cached and valid_refs(weakrefs):
                    return cached
            if domain is source.domain:
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
            conversion = DomainConversion(source.domain, domain)
            self.X = get_columns(row_indices, conversion.attributes, n_rows,
                                 is_sparse=conversion.sparse_X,
                                 variables=domain.attributes)
            if self.X.ndim == 1:
                self.X = self.X.reshape(-1, len(self.domain.attributes))

            self.Y = get_columns(row_indices, conversion.class_vars, n_rows,
                                 is_sparse=conversion.sparse_Y,
                                 variables=domain.class_vars)

            dtype = np.float64
            if any(isinstance(var, StringVariable) for var in domain.metas):
                dtype = np.object
            self.metas = get_columns(row_indices, conversion.metas,
                                     n_rows, dtype,
                                     is_sparse=conversion.sparse_metas,
                                     variables=domain.metas)
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
            _thread_local.conversion_cache[(id(domain), id(source))] = \
                self, (weakref.ref(domain), weakref.ref(source))
            return self
        finally:
            if new_cache:
                _thread_local.conversion_cache = None

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
    def from_numpy(cls, domain, X, Y=None, metas=None, W=None,
                   attributes=None, ids=None):
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
        metas, = _check_arrays(metas, dtype=object, shape_1=X.shape[0])
        ids, = _check_arrays(ids, dtype=int, shape_1=X.shape[0])

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
        if ids is None:
            cls._init_ids(self)
        else:
            self.ids = ids
        self.attributes = {} if attributes is None else attributes
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
        self.attributes = {}
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

    # Helper function for __setitem__:
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
                    val = self.X[row_idx, col_idx]
                elif col_idx >= len(self.domain.attributes):
                    val = self._Y[row_idx,
                                  col_idx - len(self.domain.attributes)]
                else:
                    val = self.metas[row_idx, -1 - col_idx]
                if isinstance(col_idx, DiscreteVariable) and var is not col_idx:
                    val = col_idx.get_mapper_from(var)(val)
                return Value(var, val)
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
            if isinstance(col_idx, DiscreteVariable) \
                    and self.domain[col_idx] != col_idx:
                values = self.domain[col_idx].get_mapper_from(col_idx)(values)
            for val, col_idx in zip(values, col_idx):
                if not isinstance(val, Integral):
                    val = self.domain[col_idx].to_val(val)
                if not isinstance(col_idx, Integral):
                    col_idx = self.domain.index(col_idx)
                if col_idx >= 0:
                    if col_idx < self.X.shape[1]:
                        self.X[row_idx, col_idx] = val
                    else:
                        self._Y[row_idx, col_idx - self.X.shape[1]] = val
                else:
                    self.metas[row_idx, -1 - col_idx] = val

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

    @classmethod
    def concatenate(cls, tables, axis=0):
        """Concatenate tables into a new table"""
        def vstack(arrs):
            return [np, sp][any(sp.issparse(arr) for arr in arrs)].vstack(arrs)

        def merge1d(arrs):
            arrs = list(arrs)
            ydims = {arr.ndim for arr in arrs}
            if ydims == {1}:
                return np.hstack(arrs)
            else:
                return vstack([
                    arr if arr.ndim == 2 else np.atleast_2d(arr).T
                    for arr in arrs
                ])

        def collect(attr):
            return [getattr(arr, attr) for arr in tables]

        if axis == 1:
            raise ValueError("concatenate no longer supports axis 1")
        if not tables:
            raise ValueError('need at least one table to concatenate')
        if len(tables) == 1:
            return tables[0].copy()
        domain = tables[0].domain
        if any(table.domain != domain for table in tables):
            raise ValueError('concatenated tables must have the same domain')

        conc = cls.from_numpy(
            domain,
            vstack(collect("X")),
            merge1d(collect("Y")),
            vstack(collect("metas")),
            merge1d(collect("W"))
        )
        conc.ids = np.hstack([t.ids for t in tables])
        names = [table.name for table in tables if table.name != "untitled"]
        if names:
            conc.name = names[0]
        # TODO: Add attributes = {} to __init__
        conc.attributes = getattr(conc, "attributes", {})
        for table in reversed(tables):
            conc.attributes.update(table.attributes)
        return conc

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

    def has_missing_attribute(self):
        """Return `True` if there are any missing attribute values."""
        return not sp.issparse(self.X) and bn.anynan(self.X)    # do not check for sparse X

    def has_missing_class(self):
        """Return `True` if there are any missing class values."""
        return bn.anynan(self._Y)

    def get_nan_frequency_attribute(self):
        if self.X.size == 0:
            return 0
        return np.isnan(self.X).sum() / self.X.size

    def get_nan_frequency_class(self):
        if self.Y.size == 0:
            return 0
        return np.isnan(self._Y).sum() / self._Y.size

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

        if isinstance(index, Integral):
            col_index = index
        else:
            col_index = self.domain.index(index)
        if col_index >= 0:
            if col_index < self.X.shape[1]:
                col = rx(self.X[:, col_index])
            else:
                col = rx(self._Y[:, col_index - self.X.shape[1]])
        else:
            col = rx(self.metas[:, -1 - col_index])

        if isinstance(index, DiscreteVariable) \
                and index.values != self.domain[col_index].values:
            col = index.get_mapper_from(self.domain[col_index])(col[0]), col[1]
            col[0].flags.writeable = False
        return col

    def _filter_is_defined(self, columns=None, negate=False):
        # structure of function is obvious; pylint: disable=too-many-branches
        def _sp_anynan(a):
            return a.indptr[1:] != a[-1:] + a.shape[1]

        if columns is None:
            if sp.issparse(self.X):
                remove = _sp_anynan(self.X)
            else:
                remove = bn.anynan(self.X, axis=1)
            if sp.issparse(self._Y):
                remove += _sp_anynan(self._Y)
            else:
                remove += bn.anynan(self._Y, axis=1)
            if sp.issparse(self.metas):
                remove += _sp_anynan(self._metas)
            else:
                for i, var in enumerate(self.domain.metas):
                    col = self.metas[:, i].flatten()
                    if var.is_primitive():
                        remove += np.isnan(col.astype(float))
                    else:
                        remove += ~col.astype(bool)
        else:
            remove = np.zeros(len(self), dtype=bool)
            for column in columns:
                col, sparse = self.get_column_view(column)
                if sparse:
                    remove += col == 0
                elif self.domain[column].is_primitive():
                    remove += bn.anynan([col.astype(float)], axis=0)
                else:
                    remove += col.astype(bool)
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
            FilterStringList, IsDefined, Values
        )
        if isinstance(filter, Values):
            return self._values_filter_to_indicator(filter)

        def get_col_indices():
            cols = chain(self.domain.variables, self.domain.metas)
            if isinstance(filter, IsDefined):
                return list(cols)

            if filter.column is not None:
                return [filter.column]

            if isinstance(filter, FilterDiscrete):
                raise ValueError("Discrete filter can't be applied across rows")
            if isinstance(filter, FilterContinuous):
                return [col for col in cols if col.is_continuous]
            if isinstance(filter,
                          (FilterString, FilterStringList, FilterRegex)):
                return [col for col in cols if col.is_string]
            raise TypeError("Invalid filter")

        def col_filter(col_idx):
            col = self.get_column_view(col_idx)[0]
            if isinstance(filter, IsDefined):
                if self.domain[col_idx].is_primitive():
                    return ~np.isnan(col.astype(float))
                else:
                    return col.astype(np.bool)
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

        col_indices = get_col_indices()
        if len(col_indices) == 1:
            return col_filter(col_indices[0])

        sel = np.ones(len(self), dtype=bool)
        for col_idx in col_indices:
            sel *= col_filter(col_idx)
        return sel

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
        with np.errstate(invalid="ignore"):   # nan's are properly discarded
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

        col_desc = [self.domain[var] for var in col_vars]
        col_indi = [self.domain.index(var) for var in col_vars]

        if any(not (var.is_discrete or var.is_continuous)
               for var in col_desc):
            raise ValueError("contingency can be computed only for discrete "
                             "and continuous values")

        # when we select a column in sparse matrix it is still two dimensional
        # and sparse - since it is just a column we can afford to transform
        # it to dense and make it 1D
        if issparse(row_data):
            row_data = row_data.toarray().ravel()
        if row_data.dtype.kind != "f": #meta attributes can be stored as type object
            row_data = row_data.astype(float)

        contingencies = [None] * len(col_desc)
        for arr, f_cond, f_ind in (
                (self.X, lambda i: 0 <= i < n_atts, lambda i: i),
                (self._Y, lambda i: i >= n_atts, lambda i: i - n_atts),
                (self.metas, lambda i: i < 0, lambda i: -1 - i)):

            arr_indi = [e for e, ind in enumerate(col_indi) if f_cond(ind)]

            vars = [(e, f_ind(col_indi[e]), col_desc[e]) for e in arr_indi]
            disc_vars = [v for v in vars if v[2].is_discrete]
            if disc_vars:
                if sp.issparse(arr):
                    max_vals = max(len(v[2].values) for v in disc_vars)
                    disc_indi = {i for _, i, _ in disc_vars}
                    mask = [i in disc_indi for i in range(arr.shape[1])]
                    conts, nans_cols, nans_rows, nans = contingency(
                        arr, row_data, max_vals - 1, n_rows - 1, W, mask)
                    for col_i, arr_i, var in disc_vars:
                        n_vals = len(var.values)
                        contingencies[col_i] = (
                            conts[arr_i][:, :n_vals], nans_cols[arr_i],
                            nans_rows[arr_i], nans[arr_i])
                else:
                    for col_i, arr_i, var in disc_vars:
                        contingencies[col_i] = contingency(
                            arr[:, arr_i].astype(float),
                            row_data, len(var.values) - 1, n_rows - 1, W)

            cont_vars = [v for v in vars if v[2].is_continuous]
            if cont_vars:
                W_ = None
                if W is not None:
                    W_ = W.astype(dtype=np.float64)
                if sp.issparse(arr):
                    arr = sp.csc_matrix(arr)

                for col_i, arr_i, _ in cont_vars:
                    if sp.issparse(arr):
                        col_data = arr.data[arr.indptr[arr_i]:arr.indptr[arr_i + 1]]
                        rows = arr.indices[arr.indptr[arr_i]:arr.indptr[arr_i + 1]]
                        W_ = None if W_ is None else W_[rows]
                        classes_ = row_data[rows]
                    else:
                        col_data, W_, classes_ = arr[:, arr_i], W_, row_data

                    col_data = col_data.astype(dtype=np.float64)
                    contingencies[col_i] = _contingency.contingency_floatarray(
                        col_data, classes_, n_rows, W_)

        return contingencies

    @classmethod
    def transpose(cls, table, feature_names_column="",
                  meta_attr_name="Feature name", feature_name="Feature"):
        """
        Transpose the table.

        :param table: Table - table to transpose
        :param feature_names_column: str - name of (String) meta attribute to
            use for feature names
        :param meta_attr_name: str - name of new meta attribute into which
            feature names are mapped
        :param feature_name: str - default feature name prefix
        :return: Table - transposed table
        """

        self = cls()
        n_cols, self.n_rows = table.X.shape
        old_domain = table.attributes.get("old_domain")

        # attributes
        # - classes and metas to attributes of attributes
        # - arbitrary meta column to feature names
        self.X = table.X.T
        if feature_names_column:
            names = [str(row[feature_names_column]) for row in table]
            names = get_unique_names_duplicates(names)
            attributes = [ContinuousVariable(name) for name in names]
        else:
            places = int(np.ceil(np.log10(n_cols)))
            attributes = [ContinuousVariable(f"{feature_name} {i:0{places}}")
                          for i in range(1, n_cols + 1)]
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


def _check_arrays(*arrays, dtype=None, shape_1=None):
    checked = []
    if not len(arrays):
        return checked

    def ninstances(array):
        if hasattr(array, "shape"):
            return array.shape[0]
        else:
            return len(array) if array is not None else 0

    if shape_1 is None:
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
            array[np.isinf(array)] = np.nan
            warnings.warn("Array contains infinity.", RuntimeWarning)
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
    conversion = DomainConversion(source.domain, target.domain)
    match_density = [assure_array_dense, assure_array_sparse]
    target.X = match_density[conversion.sparse_X](target.X)
    target.Y = match_density[conversion.sparse_Y](target.Y)
    target.metas = match_density[conversion.sparse_metas](target.metas)
    return target
