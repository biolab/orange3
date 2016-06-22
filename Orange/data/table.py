import operator
import os
import zlib
from collections import Iterable, Sequence, Sized
from functools import reduce
from threading import Lock
from warnings import warn
from numbers import Number

import bottleneck as bn
from scipy import sparse as sp
from pandas import DataFrame, SparseDataFrame, Series, SparseSeries, Panel, SparsePanel, concat

from Orange.statistics.util import bincount, countnans, contingency, stats as fast_stats
from Orange.util import flatten
from Orange.data import Domain, Variable, StringVariable, ContinuousVariable, DiscreteVariable
from Orange.data.storage import Storage
from Orange.util import flatten, deprecated
from . import _contingency
from . import _valuecount
from .instance import *


def get_sample_datasets_dir():
    orange_data_table = os.path.dirname(__file__)
    dataset_dir = os.path.join(orange_data_table, '..', 'datasets')
    return os.path.realpath(dataset_dir)


dataset_dirs = ['', get_sample_datasets_dir()]


class Role:
    """
    An enum of variable roles, provided for static convenience and a parser.
    """
    # enum values
    x, y, meta = "x", "y", "meta"

    # parsing map, also aliases
    _mappings = {"data": x, "target": y, x: x, y: y, meta: meta}

    @staticmethod
    def from_string(role_string):
        role_string = role_string.strip("s ")  # simple plurality, whitespace
        return Role._mappings.get(role_string)


# noinspection PyPep8Naming
class Table(Storage, DataFrame):
    _WEIGHTS_COLUMN = ContinuousVariable("__weights__")
    _WEIGHTS_COLUMN.is_weight = True

    # a counter for indexing rows, important for deterministically selecting rows
    # and keeping pandas indices sane
    _next_instance_id = 0
    _next_instance_lock = Lock()

    __file__ = None

    # custom properties, preserved through pandas manipulations
    _metadata = ['name',
                 '__file__',
                 '_columns_X',
                 '_columns_Y',
                 '_columns_meta']

    @property
    def _constructor(self):
        """Proper pandas extension as per http://pandas.pydata.org/pandas-docs/stable/internals.html"""
        return Table

    @property
    def _constructor_sliced(self):
        """Proper pandas extension as per http://pandas.pydata.org/pandas-docs/stable/internals.html"""
        return TableSeries

    @property
    def _constructor_expanddim(self):
        return TablePanel

    @property
    def domain(self):
        # TODO: probably return a Domain object constructed from columns
        return self.columns

    def filter_roles(self, roles):
        """
        Return a new Table which includes columns with specified roles.
        Columns are in the same order as in the current table.
        """
        if isinstance(roles, str):
            roles = [roles]
        roles = [Role.from_string(r) for r in roles]
        cols = []
        cols += [c for c in self.columns if c in self._columns_X] if Role.x in roles else []
        cols += [c for c in self.columns if c in self._columns_Y] if Role.y in roles else []
        cols += [c for c in self.columns if c in self._columns_meta] if Role.meta in roles else []
        return self[cols]

    def _to_numpy(self, X=False, Y=False, meta=False, writable=False):
        """
        Exports a numpy matrix. The order is always X, Y, meta.
        The columns are in the same order as in Table.columns.
        If writable == False (default), the numpy writable flag is set to false.
            This means write operations on this array will loudly fail. 
        """
        # TODO: only return numeric values here, need to transform
        roles = []
        roles += [Role.x] if X else []
        roles += [Role.y] if Y else []
        roles += [Role.meta] if meta else []
        res = self.filter_roles(roles).values
        res.setflags(write=writable)
        return res

    @property
    def X(self):
        """
        Return a read-only numpy matrix of X.
        The columns are in the same order as the X columns in Table.columns.
        """
        return self._to_numpy(X=True)

    @property
    def Y(self):
        """
        Return a read-only numpy matrix of Y.
        The columns are in the same order as the Y columns in Table.columns.
        """
        return self._to_numpy(Y=True)

    @property
    def metas(self):
        """
        Return a read-only numpy matrix of metas.
        The columns are in the same order as the meta columns in Table.columns.
        """
        return self._to_numpy(meta=True)

    @property
    def weights(self):
        # TODO: do we switch to .weights and deprecate .W or keep .W?
        """Get the weights as a numpy array."""
        return self[Table._WEIGHTS_COLUMN].values

    @property
    def W(self):
        return self.weights

    def set_role(self, column_names, column_roles):
        """
        Sets the role a column (or multiple columns) takes in this table.
        """
        if isinstance(column_names, str):
            column_names = [column_names]
        if isinstance(column_roles, str):
            column_roles = [column_roles]
        for n, rstr in zip(column_names, column_roles):
            r = Role.from_string(rstr)
            self._columns_X.discard(n)
            self._columns_Y.discard(n)
            self._columns_meta.discard(n)
            if r == Role.x:
                self._columns_X.add(n)
            elif r == Role.y:
                self._columns_Y.add(n)
            elif r == Role.meta:
                self._columns_meta.add(n)
            else:
                raise ValueError("{} is not a valid role name".format(rstr))

    def set_weights(self, weight):
        """
        Set the weights for the instances in this table.
        If a number, weights to set to that value.
        If a string, weights are set to whatever the column with that name's values are,
            but only if those values are all numbers and are not NA/NaN.
        If a sequence of (non-NA/NaN) numbers, set those values as the sequence.
        """
        if isinstance(weight, Number):
            self[Table._WEIGHTS_COLUMN] = weight
        elif isinstance(weight, str):
            if weight not in self.columns:
                raise ValueError("{} is not a column.".format(weight))
            if self[weight].isnull().any() and np.issubdtype(self[weight].dtype, Number):
                raise ValueError("All values in the target column must be valid numbers.")
            self[Table._WEIGHTS_COLUMN] = self[weight]
        elif isinstance(weight, Sequence):
            if len(weight) != len(self):
                raise ValueError("The sequence has length {}, expected length {}.".format(len(weight), len(self)))
            self[Table._WEIGHTS_COLUMN] = weight
        else:
            raise TypeError("Expected one of [Number, str, Sequence].")

    def __new__(cls, *args, **kwargs):
        # TODO: use modified from_X functions here

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
                return cls.from_file(args[0], **kwargs)
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
        # TODO: use this instead of __new__
        # all weights initialized to 1 (see the weight functions for details)
        self[Table._WEIGHTS_COLUMN] = 1

        self.name = kwargs.get("name") or self.name

        # used for differentiating columns into x/y/meta, as a pandas property
        self._columns_X = set()
        self._columns_Y = set()
        self._columns_meta = set()

    @classmethod
    def from_domain(cls, domain, n_rows=0, weights=False):
        # TODO: change, ignore filling with zeroes (noone uses that)
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

    conversion_cache = None

    @classmethod
    def from_table(cls, domain, source, row_indices=...):
        # TODO: change
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

        def get_columns(row_indices, src_cols, n_rows, dtype=np.float64):
            def sparse_to_flat(x):
                if sp.issparse(x):
                    x = np.ravel(x.toarray())
                return x

            if not len(src_cols):
                return np.zeros((n_rows, 0), dtype=source.X.dtype)

            n_src_attrs = len(source.domain.attributes)
            if all(isinstance(x, Integral) and 0 <= x < n_src_attrs
                   for x in src_cols):
                return _subarray(source.X, row_indices, src_cols)
            if all(isinstance(x, Integral) and x < 0 for x in src_cols):
                arr = _subarray(source.metas, row_indices,
                                 [-1 - x for x in src_cols])
                if arr.dtype != dtype:
                    return arr.astype(dtype)
                return arr
            if all(isinstance(x, Integral) and x >= n_src_attrs
                   for x in src_cols):
                return _subarray(source._Y, row_indices,
                                 [x - n_src_attrs for x in src_cols])

            a = np.empty((n_rows, len(src_cols)), dtype=dtype)
            for i, col in enumerate(src_cols):
                if col is None:
                    a[:, i] = Unknown
                elif not isinstance(col, Integral):
                    if row_indices is not ...:
                        a[:, i] = col(source)[row_indices]
                    else:
                        a[:, i] = col(source)
                elif col < 0:
                    a[:, i] = source.metas[row_indices, -1 - col]
                elif col < n_src_attrs:
                    a[:, i] = sparse_to_flat(source.X[row_indices, col])
                else:
                    a[:, i] = source._Y[row_indices, col - n_src_attrs]
            return a

        new_cache = cls.conversion_cache is None
        try:
            if new_cache:
                cls.conversion_cache = {}
            else:
                cached = cls.conversion_cache.get((id(domain), id(source)))
                if cached:
                    return cached
            if domain == source.domain:
                return cls.from_table_rows(source, row_indices)

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
            self.X = get_columns(row_indices, conversion.attributes, n_rows)
            if self.X.ndim == 1:
                self.X = self.X.reshape(-1, len(self.domain.attributes))
            self.Y = get_columns(row_indices, conversion.class_vars, n_rows)

            dtype = np.float64
            if any(isinstance(var, StringVariable) for var in domain.metas):
                dtype = np.object
            self.metas = get_columns(row_indices, conversion.metas,
                                     n_rows, dtype)
            if self.metas.ndim == 1:
                self.metas = self.metas.reshape(-1, len(self.domain.metas))
            if source.has_weights():
                self.W = np.array(source.W[row_indices])
            else:
                self.W = np.empty((n_rows, 0))
            self.name = getattr(source, 'name', '')
            if hasattr(source, 'ids'):
                self.ids = np.array(source.ids[row_indices])
            else:
                cls._init_ids(self)
            self.attributes = getattr(source, 'attributes', {})
            cls.conversion_cache[(id(domain), id(source))] = self
            return self
        finally:
            if new_cache:
                cls.conversion_cache = None

    @classmethod
    def from_table_rows(cls, source, row_indices):
        # TODO: change
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
        # TODO: change
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
                Y = np.empty((X.shape[0], 0), object)
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
        # TODO: change
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
    def _new_id(cls, num=1):
        """
        Generate new globally unique numbers.
        Generate a single number or a list of them, if specified.
        """
        with cls._next_instance_lock:
            out = np.arange(cls._next_instance_id, cls._next_instance_id + num)
            return out[0] if num == 1 else out

    def save(self, filename):
        # TODO: change, will likely need to modify FileFormat.writers
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
                raise IOError("Writing of {}s is not supported".
                    format(desc.lower()))
            else:
                raise IOError("Unknown file name extension.")
        writer.write_file(filename, self)

    @classmethod
    def from_file(cls, filename):
        # TODO: change, will likely need to modify FileFormat.readers
        """
        Read a data table from a file. The path can be absolute or relative.

        :param filename: File name
        :type filename: str
        :return: a new data table
        :rtype: Orange.data.Table
        """
        from Orange.data.io import FileFormat

        absolute_filename = FileFormat.locate(filename, dataset_dirs)
        reader = FileFormat.get_reader(absolute_filename)
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

    def __setitem__(self, key, value):
        # we only override this for certain types:
        #  - Variables
        #  - plain strings (their Variable is constructed on the fly)
        # if the column is already in the table, we aren't adding new columns
        # otherwise we pass this along to the parent
        if not (isinstance(key, Variable) or isinstance(key, str)) or key in self.columns:
            super(Table, self).__setitem__(key, value)

        # assertion: the column is new from here on
        # the if ordering is important here, since Variable extends str
        force_meta = False
        if isinstance(key, Variable):
            # the variable already exists, don't argue
            var = key
        else:  # plain string
            # we need to construct a new variable (in order of precedence)
            #  - all numerics and all elements \in {0, 1}: discrete
            #  - all numerics: continuous
            #  - all strings and all distinct: string, meta
            #  - any string: discrete
            #  - otherwise: error, can't determine type

            proc_val = value  # for easier type checking even when broadcasting
            if not isinstance(proc_val, Sequence) or isinstance(proc_val, str):
                proc_val = [value]
            proc_val = np.array([value])  # for type checking later

            if np.issubdtype(proc_val.dtype, np.number) and set(proc_val) == {0, 1}:
                var = DiscreteVariable(key, values=[0, 1])
            elif np.issubdtype(proc_val.dtype, np.number):
                var = ContinuousVariable(key)
            elif np.issubdtype(proc_val.dtype, 'U') and len(set(proc_val)) == len(proc_val):
                var = StringVariable(key)
                force_meta = True
            elif np.issubdtype(proc_val.dtype, 'U'):
                var = DiscreteVariable(key, values=sorted(set(proc_val)))
            else:
                raise ValueError("Cannot automatically determine variable type. ")

        self[var] = value
        # default behaviour is to include this in X, except when meta is set
        # we can't reasonably separate X and Y here
        if force_meta:
            self._columns_meta.add(var)
        else:
            self._columns_X.add(var)

    # TODO: update str and repr
    def __str__(self):
        return "[" + ",\n ".join(str(ex) for ex in self)

    def __repr__(self):
        s = "[" + ",\n ".join(repr(ex) for ex in self[:5])
        if len(self) > 5:
            s += ",\n ..."
        s += "\n]"
        return s

    def clear(self):
        """Remove all rows from the table in-place."""
        self.drop(self.index, inplace=True)

    def append(self, row):
        """
        Append a new row to the table.
        row can either be a single value (broadcast),
        a list-like of values or a TableSeries (a single row slice).
        """
        new_ix = Table._new_id()
        self.loc[new_ix] = row

    @deprecated('Use t.append() for adding new rows. This inserts a new column. ')
    def insert(self, loc, column, value, allow_duplicates=False):
        if not allow_duplicates and column in self.columns:
            raise ValueError("Column already exists. ")
        self[column] = value

    # TODO: deprecate this?
    def extend(self, rows, weight=1):
        """
        Extend the table with the given rows.
        rows can be either a list of rows or a descendant of DataFrame.
        """
        if not isinstance(rows, DataFrame):
            rows = Table(rows)
        if Table._WEIGHTS_COLUMN not in rows:
            rows[Table._WEIGHTS_COLUMN] = weight
        return Table.concatenate([self, rows], axis=1, rowstack=True)

    @staticmethod
    def concatenate(tables, axis=1, reindex=True, colstack=True, rowstack=False):
        """
        Concatenate tables by rows (axis = 0) or columns (axis = 1).
        If concatenating by columns, all tables must be the same length and
            no two columns may have the same name.
        If concatenating by rows, perform an outer join if rowstack == False, otherwise stack.
        By default, this performs reindexing: all resulting rows will be given a new index.
        If reindex == False
            - when concatenating rows: some rows may have the same index.
            - when concatenating columns: the index of the first table is preserved.
        If colstack == False, perform an outer join instead of column stacking.
        The resulting table will always retain the properties (name etc.) of the first table.
        """
        if not tables:
            raise ValueError('Need at least one table to concatenate.')
        if len(tables) == 1:
            return tables[0].copy()
        CONCAT_ROWS, CONCAT_COLS = 0, 1
        if axis == CONCAT_ROWS:
            if rowstack:
                # check for the same number of columns
                if len(set(len(t.columns) for t in tables)) != 1:
                    raise ValueError("Cannot rowstack with differing numbers of columns.")
                # rename non-first columns to be the same as first (only way to stack)
                # this ia a bit convoluted because we can't chain renames
                newtables = [tables[0]]
                for t in tables[1:]:
                    new = t.copy()
                    new.columns = tables[0].columns
                    newtables.append(new)
                res = concat(newtables, axis=0, ignore_index=True)
            else:
                res = concat(tables, axis=0, ignore_index=True)
            new_index = Table._new_id(len(res))
            res.index = new_index
        elif axis == CONCAT_COLS:
            # check for same name
            columns = flatten([v.name for v in [t.columns for t in tables] if v.name != Table._WEIGHTS_COLUMN])
            if len(set(columns)) != len(columns):
                raise ValueError("Cannot concatenate domains with same names.")
            if colstack:
                # check for same length
                if len(set(len(t) for t in tables)) != 1:
                    raise ValueError("Cannot colstack tables with differing numbers of rows. ")
                # reset index temporarily because this joins by index by default
                res = concat([t.reset_index(drop=True) for t in tables], axis=1, join_axes=[tables[0].index])
            else:
                res = concat(tables, axis=1)

            # fix multiple weight columns
            weight_columns = res[Table._WEIGHTS_COLUMN]
            for i in range(1, len(weight_columns.columns)):
                weight_columns.fillna([weight_columns[[i]]], axis=1, inplace=True)
            res = res.drop(Table._WEIGHTS_COLUMN, axis=0)
            res[Table._WEIGHTS_COLUMN] = weight_columns[[0]]

            if reindex:
                new_index = Table._new_id(len(res))
                res.index = new_index
        else:
            raise ValueError('axis {} out of bounds [0, 2)'.format(axis))
        res._transfer_properties(tables[0])  # pd.concat does not do this by itself
        return res

    def _transfer_properties(self, from_table):
        """
        Transfer properties (such as the name) to this table.
        This should normally not be used, but it is used when these properties
        are not automatically transferred on manipulation, in particular when using pd.concat.
        """
        for name in self._metadata:
            if hasattr(from_table, name):
                setattr(self, name, getattr(from_table, name))

    def density(self):
        """
        Compute the table density:
         - for sparse tables, return the reported density.
         - for dense tables, return the ratio of null values (pandas interpretation of null).
        :return:
        """
        if isinstance(self, SparseDataFrame):
            return super(Table, self).density
        else:
            return 1 - self.isnull().sum().sum() / self.size

    def has_missing(self):
        """Return `True` if there are any missing attribute or class values."""
        # manual access to columns because dumping to a numpy array (with self.X) is slower
        return self.filter_roles(Role.x).isnull().any().any() or self.has_missing_class()

    def has_missing_class(self):
        """Return `True` if there are any missing class values."""
        return self.filter_roles(Role.y).isnull().any().any()

    @deprecated
    def checksum(self, include_metas=True):
        # TODO: zlib.adler32 does not work for numpy arrays with dtype object
        # (after pickling and unpickling such arrays, checksum changes)
        # Why, and should we fix it or remove it?
        """Return a checksum over X, Y, metas and W."""
        cs = zlib.adler32(np.ascontiguousarray(self.X))
        cs = zlib.adler32(np.ascontiguousarray(self.Y), cs)
        if include_metas:
            cs = zlib.adler32(np.ascontiguousarray(self.metas), cs)
        cs = zlib.adler32(np.ascontiguousarray(self.weights), cs)
        return cs

    def shuffle(self):
        """
        Shuffle the rows of the table.
        Return a new table (with the same index).
        """
        return self.sample(frac=1)

    @deprecated('pandas-style column access: t[["colname1", "colname2"]]')
    def get_column_view(self, index):
        """
        Return a vector - as a view, not a copy - with a column of the table,
        and a bool flag telling whether this column is sparse. Note that
        vertical slicing of sparse matrices is inefficient.

        :param index: the index of the column
        :type index: int, str or Orange.data.Variable
        :return: (one-dimensional numpy array, sparse)
        """
        if isinstance(index, str):
            col = self[index]
        else:
            col = self[self.columns[index]]
        return col.values, isinstance(col, SparseSeries)

    # TODO: remove filters in general, transform the few usages to pandas
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

    def _filter_values_indicators(self, filter):
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
            if isinstance(f, data_filter.Values):
                if conjunction:
                    sel *= self._filter_values_indicators(f)
                else:
                    sel += self._filter_values_indicators(f)
                continue
            col = self.get_column_view(f.column)[0]
            if isinstance(f, data_filter.FilterDiscrete) and f.values is None \
                    or isinstance(f, data_filter.FilterContinuous) and \
                                    f.oper == f.IsDefined:
                if conjunction:
                    sel *= ~np.isnan(col)
                else:
                    sel += ~np.isnan(col)
            elif isinstance(f, data_filter.FilterString) and \
                            f.oper == f.IsDefined:
                if conjunction:
                    sel *= col.astype(bool)
                else:
                    sel += col.astype(bool)
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
                    # noinspection PyTypeChecker
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
            elif isinstance(f, data_filter.FilterRegex):
                sel = np.vectorize(f)(col)
            elif isinstance(f, (data_filter.FilterContinuous,
                                data_filter.FilterString)):
                if (isinstance(f, data_filter.FilterString) and
                        not f.case_sensitive):
                    # noinspection PyTypeChecker
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
        return sel

    def _filter_values(self, filter):
        sel = self._filter_values_indicators(filter)
        return self.from_table(self.domain, self, sel)

    # TODO: move this to statistics.py and use pandas instead if this code
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
            columns = [self.domain.index(c) for c in columns]
            nattrs = len(self.domain.attributes)
            Xs = any(0 <= c < nattrs for c in columns) and fast_stats(self.X, W)
            Ys = any(c >= nattrs for c in columns) and fast_stats(self._Y, W)
            ms = any(c < 0 for c in columns) and fast_stats(self.metas, W)
            for column in columns:
                if 0 <= column < nattrs:
                    stats.append(Xs[column, :])
                elif column >= nattrs:
                    stats.append(Ys[column - nattrs, :])
                else:
                    stats.append(ms[-1 - column])
        return stats

    # TODO: move this to distributions.py and use pandas instead if this code
    def _compute_distributions(self, columns=None):
        def _get_matrix(M, cachedM, col):
            nonlocal single_column
            if not sp.issparse(M):
                return M[:, col], self.W if self.has_weights() else None, None
            if cachedM is None:
                if single_column:
                    warn("computing distributions on sparse data "
                         "for a single column is inefficient")
                cachedM = sp.csc_matrix(self.X)
            data = cachedM.data[cachedM.indptr[col]:cachedM.indptr[col + 1]]
            if self.has_weights():
                weights = self.W[
                    cachedM.indices[cachedM.indptr[col]:cachedM.indptr[col + 1]]]
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
            if 0 <= col < self.X.shape[1]:
                m, W, Xcsc = _get_matrix(self.X, Xcsc, col)
            elif col < 0:
                m, W, Xcsc = _get_matrix(self.metas, Xcsc, col * (-1) - 1)
            else:
                m, W, Ycsc = _get_matrix(self._Y, Ycsc, col - self.X.shape[1])
            if var.is_discrete:
                if W is not None:
                    W = W.ravel()
                dist, unknowns = bincount(m, len(var.values) - 1, W)
            elif not len(m):
                dist, unknowns = np.zeros((2, 0)), 0
            else:
                if W is not None:
                    ranks = np.argsort(m)
                    vals = np.vstack((m[ranks], W[ranks].flatten()))
                    unknowns = countnans(m, W)
                else:
                    vals = np.ones((2, m.shape[0]))
                    vals[0, :] = m
                    vals[0, :].sort()
                    unknowns = countnans(m.astype(float))
                dist = np.array(_valuecount.valuecount(vals))
            distributions.append((dist, unknowns))

        return distributions

    # TODO: move this to contingency.py and use pandas instead if this code
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
                    for col_i, arr_i, _ in disc_vars:
                        contingencies[col_i] = (conts[arr_i], nans[arr_i])
                else:
                    for col_i, arr_i, var in disc_vars:
                        contingencies[col_i] = contingency(
                            arr[:, arr_i].astype(float),
                            row_data, len(var.values) - 1, n_rows - 1, W)

            cont_vars = [v for v in vars if v[2].is_continuous]
            if cont_vars:

                classes = row_data.astype(dtype=np.int8)
                if W is not None:
                    W = W.astype(dtype=np.float64)
                if sp.issparse(arr):
                    arr = sp.csc_matrix(arr)

                for col_i, arr_i, _ in cont_vars:
                    if sp.issparse(arr):
                        col_data = arr.data[arr.indptr[arr_i]:
                        arr.indptr[arr_i + 1]]
                        rows = arr.indices[arr.indptr[arr_i]:
                        arr.indptr[arr_i + 1]]
                        W_ = None if W is None else W[rows]
                        classes_ = classes[rows]
                    else:
                        col_data, W_, classes_ = arr[:, arr_i], W, classes

                    col_data = col_data.astype(dtype=np.float64)
                    U, C, unknown = _contingency.contingency_floatarray(
                        col_data, classes_, n_rows, W_)
                    contingencies[col_i] = ([U, C], unknown)

        return contingencies, unknown_rows


class TableSeries(Series):
    """
    A subclass of pandas' Series to properly override constructors to avoid problems.
    """
    @property
    def _constructor(self):
        """Proper pandas extension as per http://pandas.pydata.org/pandas-docs/stable/internals.html"""
        return TableSeries

    @property
    def _constructor_expanddim(self):
        """Proper pandas extension as per http://pandas.pydata.org/pandas-docs/stable/internals.html"""
        return Table


class TablePanel(Panel):
    """
    A subclass of pandas' Panel to properly override constructors to avoid problems.
    """
    @property
    def _constructor(self):
        return TablePanel

    @property
    def _constructor_sliced(self):
        return Table


# TODO: check usages for this (and below) and remove them once their users are gone
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
                             % (len(array), shape_1))

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
    return arr[_rxc_ix(rows, cols)]


def _rxc_ix(rows, cols):
    """
    Construct an index object to index the `rows` x `cols` cross product.

    Rows and columns can be a 1d bool or int sequence, a slice or an
    Ellipsis (`...`). The later is a convenience and is interpreted the same
    as `slice(None, None, -1)`

    Parameters
    ----------
    rows : 1D sequence, slice or Ellipsis
        Row indices.
    cols : 1D sequence, slice or Ellipsis
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
    >>> a[_rxc_ix([False, True], ...)]
    array([[5, 6, 7, 8, 9]])

    """
    rows = slice(None, None, 1) if rows is ... else rows
    cols = slice(None, None, 1) if cols is ... else cols

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
