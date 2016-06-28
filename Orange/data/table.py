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
import pandas as pd

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


# noinspection PyPep8Naming
class Table(pd.DataFrame):
    _WEIGHTS_COLUMN = "__weights__"

    # a counter for indexing rows, important for deterministically selecting rows
    # and keeping pandas indices sane
    _next_instance_id = 0
    _next_instance_lock = Lock()

    conversion_cache = None

    # custom properties, preserved through pandas manipulations
    _metadata = ['name',
                 'domain',
                 'attributes',
                 '__file__']

    @staticmethod
    def pandas_constructor_proxy(new_data, *args, **kwargs):
        """
        A proxy constructor, needed because we override __new__.
        Example: when selecting a subset of a Table, pandas calls _constructor (or similar)
                 to get the class which has to be constructed. In our case this is Table, but
                 because __new__ is complicated--calls different factories depending on
                 the arguments passed. Because we can't handle this behaviour (it's pandas internal),
                 we proxy a constructor with this callable to allow pandas internals
                 to still work.
        """
        # TODO: just for testing, remove afterwards
        # we expect only one argument
        if args or kwargs:
            for _ in range(10):
                print("UNEXPECTED PANDAS BEHAVIOUR")
        return Table(data=new_data)

    @property
    def _constructor(self):
        """Proper pandas extension as per http://pandas.pydata.org/pandas-docs/stable/internals.html"""
        return Table.pandas_constructor_proxy

    @property
    def _constructor_sliced(self):
        return TableSeries

    @property
    def _constructor_expanddim(self):
        return TablePanel

    def _to_numpy(self, X=False, Y=False, meta=False, writable=False):
        """
        Exports a numpy matrix. The order is always X, Y, meta. Always 2D.
        The columns are in the same order as in Table.domain._.
        If writable == False (default), the numpy writable flag is set to false.
            This means write operations on this array will loudly fail. 
        """
        cols = []
        cols += self.domain.attributes if X else []
        cols += self.domain.class_vars if Y else []
        cols += self.domain.metas if meta else []

        # preallocate result, we fill it in-place
        # we need a more general dtype for metas (strings),
        # otherwise assignment fails later
        res = np.zeros((len(self), len(cols)), dtype=object if meta else None)
        # effectively a double for loop, see if this is a bottleneck later
        for i, col in enumerate(cols):
            res[:, i] = self[col].apply(col.to_val).values
        res.setflags(write=writable)
        return res

    @property
    def X(self):
        """
        Return a read-only numpy matrix of X.
        The columns are in the same order as the columns in Table.domain.attributes.
        """
        return self._to_numpy(X=True)

    @property
    def Y(self):
        """
        Return a read-only numpy matrix of Y.
        If there is only one column, a one-dimensional array is returned. Otherwise 2D.
        The columns are in the same order as the columns in Table.domain.class_vars.
        """
        res = self._to_numpy(Y=True)
        return res[:, 0] if res.shape[1] == 1 else res

    @property
    def metas(self):
        """
        Return a read-only numpy matrix of metas.
        The columns are in the same order as the columns in Table.domain.metas.
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

    def set_weights(self, weight):
        """
        Set the weights for the instances in this table.
        If a number, weights to set to that value.
        If a string, weights are set to whatever the column with that name's values are,
            but only if those values are all numbers and are not NA/NaN.
        If a sequence of (non-NA/NaN) numbers, set those values as the sequence.
        """
        # TODO: handle NAs
        if isinstance(weight, Number):
            self[Table._WEIGHTS_COLUMN] = weight
        elif isinstance(weight, str):
            if weight not in self.columns:
                raise ValueError("{} is not a column.".format(weight))
            if self[weight].isnull().any() and np.issubdtype(self[weight].dtype, Number):
                raise ValueError("All values in the target column must be valid numbers.")
            self[Table._WEIGHTS_COLUMN] = self[weight]
        elif isinstance(weight, (Sequence, np.ndarray)):  # np.ndarray is not a Sequence
            if len(weight) != len(self):
                raise ValueError("The sequence has length {}, expected length {}.".format(len(weight), len(self)))
            self[Table._WEIGHTS_COLUMN] = weight
        else:
            raise TypeError("Expected one of [Number, str, Sequence].")

    def __new__(cls, *args, **kwargs):
        """
        Create a new Table. Needed because we have two construction paths: Table() or Table.from_X.
        If called without arguments, create and initialize a blank Table, otherwise
        intelligently call one of the Table.from_X functions, depending on the arguments.
        Also passes through pandas.DataFrame constructor keyword arguments.
        Do not pass positional arguments through to pandas.
        """
        # if we called the constructor without arguments or
        # if we only called this with pandas DataFrame kwargs (not args),
        # create an empty Table, the kwargs will be passed through to init (for pandas)
        if not args and (not kwargs
                         or not set(kwargs.keys()).difference(["data", "index", "columns", "dtype", "copy"])):
            return super().__new__(cls)

        if 'filename' in kwargs:
            args = [kwargs.pop('filename')]

        if not args:
            raise TypeError("Table takes at least 1 positional argument (0 given))")

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
        # see the comment in __new__ for the rationale here
        # also, another tidbit is that pandas has some internals that need to be set up
        # and expects its arguments to be set appropriately
        # because we override the constructor arguments in a completely incompatible way,
        # we need to pass ourselves as the data object if we have already set things up
        # previously by e.g. creating an empty Table via the __new__ hack in from_X
        # functions, then filling up with columns.
        # to check for this, we check for domain existence because tables without domains
        # can't really be used in Orange in any meaningful way
        if hasattr(self, 'domain'):
            kwargs['data'] = self
        super(Table, self).__init__(**kwargs)

        # all weights initialized to 1 (see the weight functions for details)
        self.name = kwargs.get("name", "untitled")
        self.attributes = kwargs.get("attributes", {})
        self.__file__ = kwargs.get("__file__")

        # we need to filter the domain to only include the columns present in the table
        # but we still need to allow constructing an empty table (with no domain)
        # also, we only set the domain if it has changed (==number of variables),
        # so in those cases, id(domain_before) == id(domain_after)
        if hasattr(self, 'domain'):
            new_domain = Domain(
                [c for c in self.domain.attributes if c in self.columns],
                [c for c in self.domain.class_vars if c in self.columns],
                [c for c in self.domain.metas if c in self.columns]
            )
            if len(new_domain.variables) + len(new_domain.metas) != \
               len(self.domain.variables) + len(self.domain.metas):
                self.domain = new_domain
        else:
            self.domain = None

        # only set the weights if they aren't set already
        if Table._WEIGHTS_COLUMN not in self.columns:
            self.set_weights(1)

    @classmethod
    def from_domain(cls, domain):
        """
        Construct a new `Table` for the given domain.

        :param domain: domain for the `Table`
        :type domain: Orange.data.Domain
        :return: a new table
        :rtype: Orange.data.Table
        """
        res = cls(columns=domain.attributes + domain.class_vars + domain.metas)
        res.domain = domain
        return res

    @classmethod
    def from_table(cls, target_domain, source_table, row_indices=...):
        """
        Create a new table from selected columns and/or rows of an existing
        one. The columns are chosen using a domain. The domain may also include
        variables that do not appear in the source table; they are computed
        from source variables if possible.

        The resulting data may be a view or a copy of the existing data.

        :param target_domain: the domain for the new table
        :type target_domain: Orange.data.Domain
        :param source_table: the source table
        :type source_table: Orange.data.Table
        :param row_indices: indices of the rows to include
        :type row_indices: a slice or a sequence
        :return: a new table
        :rtype: Orange.data.Table
        """
        new_cache = cls.conversion_cache is None
        try:
            if new_cache:
                cls.conversion_cache = {}
            else:
                cached = cls.conversion_cache.get((id(target_domain), id(source_table)))
                if cached:
                    return cached
            if target_domain == source_table.domain:
                return cls.from_table_rows(source_table, row_indices)

            res = cls()
            conversion = target_domain.get_conversion(source_table.domain)

            for conversion, target_column in zip(chain(conversion.variables, conversion.metas),
                                                 chain(target_domain.variables, target_domain.metas)):
                if isinstance(conversion, Number):
                    res[target_column.name] = source_table[source_table.domain[conversion]]
                else:
                    res[target_column.name] = conversion(source_table)
            res.domain = target_domain

            res.set_weights(source_table.weights)
            res.index = source_table.index  # keep previous index
            res = res.iloc[row_indices]

            cls.conversion_cache[(id(target_domain), id(source_table))] = res
            return res
        finally:
            if new_cache:
                cls.conversion_cache = None

    @classmethod
    @deprecated("t.iloc[row_indices].copy()")
    def from_table_rows(cls, source, row_indices):
        """
        Construct a new table (copy) by selecting rows from the source table by their
        position on the table.

        :param source: an existing table
        :type source: Orange.data.Table
        :param row_indices: indices (positional) of the rows to include
        :type row_indices: a slice or a sequence
        :return: a new table
        :rtype: Orange.data.Table
        """
        return source.iloc[row_indices].copy()

    @classmethod
    def _from_data_inferred(cls, X_or_data, Y=None, meta=None, infer_roles=True):
        """
        Create a Table and infer its domain.

        X_or_data, Y and meta can be instances of Table, DataFrame,
        np.ndarray or a list of rows.

        If X_or_data is the sole argument and infer_roles == True,
        we will try to infer the column role (x/y/meta) from the data.
        If infer_roles = False or Y or meta are given, column roles will be
        set to what its container argument represents.

        This only does shallow inference on data types. Example:
        if given a numpy matrix of dtype object (e.g. mixed numbers and strings),
        pandas will interpret all columns as objects, and so will we.

        Return a new Table with the inferred domain.
        Where possible, column names are preserved form the input, otherwise they are named
        "Feature <n>", "Class <n>", "Target <n>" or "Meta <n>".
        The domain is marked as anonymous.
        """
        role_vars = {'x': [], 'y': [], 'meta': []}
        res = cls()

        # used for data inference, shape checks, consolidated access
        X_df = pd.DataFrame(data=X_or_data)
        Y_df = pd.DataFrame(data=Y)
        meta_df = pd.DataFrame(data=meta)

        def _compute_name(colname, r):
            if isinstance(colname, Integral):
                # choose a new name, we don't want numbers
                if r == 'x':
                    return "Feature {}".format(len(role_vars['x']) + 1)
                elif r == 'y' and isinstance(var, ContinuousVariable):
                    return "Target {}".format(len(role_vars['y']) + 1)
                elif r == 'y' and isinstance(var, DiscreteVariable):
                    return "Class {}".format(len(role_vars['y']) + 1)
                else:
                    return "Meta {}".format(len(role_vars['meta']) + 1)
            else:
                return colname

        # override, because the user wishes to specify roles manually
        if Y is not None or meta is not None:
            infer_roles = False

        # process every input segment with its intended role
        for df, initial_role in zip((X_df, Y_df, meta_df), ('x', 'y', 'meta')):
            # choose whether to force a role or allow inference
            role = initial_role if not infer_roles else None
            for column_name, column, uniq in ((c, df[c], df[c].unique()) for c in df.columns):
                # if there are at most 3 different values of any kind, they are discrete
                if len(uniq) <= 3:
                    role = role or 'x'
                    var = DiscreteVariable(_compute_name(column_name, role), values=sorted(uniq))
                # all other all-number columns are continuous features
                elif np.issubdtype(column.dtype, np.number):
                    role = role or 'x'
                    var = ContinuousVariable(_compute_name(column_name, role))
                # all others (including those we can't determine) are string metas
                else:
                    role = role or 'meta'
                    var = StringVariable(_compute_name(column_name, role))
                res[var.name] = column
                role_vars[role].append(var)
        res.domain = Domain(role_vars['x'], role_vars['y'], role_vars['meta'])
        res.domain.anonymous = True
        return res

    @classmethod
    def from_numpy(cls, domain, X, Y=None, metas=None, weights=None):
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
        :param weights: array with weights
        :type weights: np.array
        :return:
        """
        def correct_shape(what):
            if what is None or len(what.shape) == 2:
                return what
            else:
                return np.atleast_2d(what).T

        if domain is None:
            res = cls._from_data_inferred(X, Y, metas)
            if weights is not None:
                res.set_weights(weights)
            return res

        # ensure correct shapes (but not sizes) so we can iterate
        X = correct_shape(X)
        Y = correct_shape(Y)
        metas = correct_shape(metas)

        res = cls()
        for role_array, variables in zip((X, Y, metas),
                                         (domain.attributes, domain.class_vars, domain.metas)):
            if role_array is None:
                if variables:
                    raise ValueError("Variable and column count mismatch. ")
                continue
            if role_array.shape[1] != len(variables):
                raise ValueError("Variable and column count mismatch. ")
            for column, variable in zip(role_array.T, variables):
                res[variable.name] = column
        res.domain = domain
        return res

    @classmethod
    def from_list(cls, domain, rows, weights=None):
        """
        Construct a table from a list of rows and optionally some weights.
        """
        if weights is not None and len(rows) != len(weights):
            raise ValueError("Mismatching number of instances and weights.")
        # check dimensions, pandas raises a very nondescript error
        row_width = len(rows[0])
        for r in rows:
            if len(r) != row_width:
                raise ValueError("Inconsistent number of columns.")

        res = cls(data=rows,
                  columns=[a.name for a in chain(domain.attributes, domain.class_vars, domain.metas)])
        res.domain = domain
        if weights is not None:
            res.set_weights(weights)
        return res

    @classmethod
    def from_dataframe(cls, df):
        """
        Convert a pandas.DataFrame object to a Table.
        This infers column variable types and roles.
        """
        return cls._from_data_inferred(df)

    @classmethod
    def _new_id(cls, num=1, force_list=False):
        """
        Generate new globally unique numbers.
        Generate a single number or a list of them, if specified.
        """
        with cls._next_instance_lock:
            out = np.arange(cls._next_instance_id, cls._next_instance_id + num)
            cls._next_instance_id += num
            return out[0] if num == 1 and not force_list else out

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

    def __getitem__(self, item):
        # if selecting a column subset, we need to transfer weights so they don't just disappear
        # only do this for multiple column selection, which returns a DataFrame by contract
        if isinstance(item, (Sequence, pd.Index)) and not isinstance(item, str) \
                and all(isinstance(i, str) for i in item) \
                and Table._WEIGHTS_COLUMN not in item:
            item = list(item) + [Table._WEIGHTS_COLUMN]
        return super(Table, self).__getitem__(item)

    def __setitem__(self, key, value):
        # if the table has an empty index and we're inserting a new row,
        # the index would be created by pandas automatically.
        # we want to maintain unique indices, so we override the index manually.
        # we also need to set default weights, lest they be NA
        new_index_and_weights = len(self.index) == 0

        super(Table, self).__setitem__(key, value)
        if new_index_and_weights:
            new_id = Table._new_id(len(self))
            self.index = new_id
            # super call because we'd otherwise recurse back into this
            super(Table, self).__setitem__(Table._WEIGHTS_COLUMN, 1)

    # TODO: update str and repr
    def __str__(self):
        # return "[" + ",\n ".join(str(ex) for ex in self)
        return super(Table, self).__str__()

    def __repr__(self):
        # s = "[" + ",\n ".join(repr(ex) for ex in self[:5])
        # if len(self) > 5:
        #     s += ",\n ..."
        # s += "\n]"
        # return s
        return super(Table, self).__str__()

    def clear(self):
        """Remove all rows from the table in-place."""
        self.drop(self.index, inplace=True)

    def append(self, other, ignore_index=False, verify_integrity=False):
        """
        Append a new row to the table, returning a new Table.
        row can be a list-like of a single row, TableSeries (a single row slice) or a Table.
        """
        # handle all indexing (this needs to be different in Table) in concatenate
        # the pandas contract is not in-place anyway.
        if not isinstance(other, pd.DataFrame):
            other = pd.DataFrame(data={col: val for col, val in zip(self.columns, other)})
        other.index = Table._new_id(len(other), force_list=True)
        return Table.concatenate([self, other], axis=0, reindex=False, rowstack=True)

    @deprecated('Use Table.append() for adding new rows. This inserts a new column. ')
    def insert(self, *args, **kwargs):
        super(Table, self).insert(*args, **kwargs)

    @deprecated("Table.append(...)")
    def extend(self, rows, weight=1):
        return self.append(rows)

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
                res = pd.concat(newtables, axis=0, ignore_index=True)
            else:
                res = pd.concat(tables, axis=0, ignore_index=True)
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
                res = pd.concat([t.reset_index(drop=True) for t in tables], axis=1, join_axes=[tables[0].index])
            else:
                res = pd.concat(tables, axis=1)

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
        """
        if isinstance(self, pd.SparseDataFrame):
            return super(Table, self).density
        else:
            return 1 - self.isnull().sum().sum() / self.size

    def is_sparse(self):
        return isinstance(self, pd.SparseDataFrame)

    def is_dense(self):
        return not self.is_sparse()

    def has_missing(self):
        """Return `True` if there are any missing attribute or class values."""
        # manual access to columns because dumping to a numpy array (with self.X) is slower
        return self[self.domain.attributes].isnull().any().any() or self.has_missing_class()

    def has_missing_class(self):
        """Return `True` if there are any missing class values."""
        return self[self.domain.class_vars].isnull().any().any()

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
        return col.values, isinstance(col, pd.SparseSeries)

    # TODO: move this to statistics.py and use pandas instead if this code
    def _compute_basic_stats(self, columns=None,
                             include_metas=False, compute_variance=False):
        """
        Compute basic stats for each of the columns.

        :param columns: columns to calculate stats for. None = all of them
        :return: tuple(min, max, mean, 0, #nans, #non-nans)
        """
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
        """
        Compute distribution of values for the given columns.

        :param columns: columns to calculate distributions for
        :return: a list of distributions. Type of distribution depends on the
                 type of the column:
                   - for discrete, distribution is a 1d np.array containing the
                     occurrence counts for each of the values.
                   - for continuous, distribution is a 2d np.array with
                     distinct (ordered) values of the variable in the first row
                     and their counts in second.
        """
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
        """
        Compute contingency matrices for one or more discrete or
        continuous variables against the specified discrete variable.

        The resulting list  contains a pair for each column variable.
        The first element contains the contingencies and the second
        elements gives the distribution of the row variables for instances
        in which the value of the column variable is missing.

        The format of contingencies returned depends on the variable type:

        - for discrete variables, it is a numpy array, where
          element (i, j) contains count of rows with i-th value of the
          row variable and j-th value of the column variable.

        - for continuous variables, contingency is a list of two arrays,
          where the first array contains ordered distinct values of the
          column_variable and the element (i,j) of the second array
          contains count of rows with i-th value of the row variable
          and j-th value of the ordered column variable.

        :param col_vars: variables whose values will correspond to columns of
            contingency matrices
        :type col_vars: list of ints, variable names or descriptors of type
            :obj:`Orange.data.Variable`
        :param row_var: a discrete variable whose values will correspond to the
            rows of contingency matrices
        :type row_var: int, variable name or :obj:`Orange.data.DiscreteVariable`
        """
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


class TableSeries(pd.Series):
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
        return Table.pandas_constructor_proxy


class TablePanel(pd.Panel):
    """
    A subclass of pandas' Panel to properly override constructors to avoid problems.
    """
    @property
    def _constructor(self):
        return TablePanel

    @property
    def _constructor_sliced(self):
        return Table.pandas_constructor_proxy
