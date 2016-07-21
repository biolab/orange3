import os
import zlib
from collections import Sequence
from threading import Lock
from numbers import Number

from itertools import chain
import numpy as np
import pandas as pd
import pandas.core.internals

from Orange.data import Domain, StringVariable, ContinuousVariable, DiscreteVariable, Variable, TimeVariable
from Orange.util import flatten, deprecated


# noinspection PyPep8Naming
class TableBase:
    KNOWN_PANDAS_KWARGS = {}

    # these were previously in Storage
    MISSING, DENSE, SPARSE, SPARSE_BOOL = range(4)

    # the default name for weights columns
    # subclasses may override this to rename the column
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

    def __new__(cls, *args, **kwargs):
        """
        Create a new Table. Needed because we have two construction paths: Table() or Table.from_X.
        If called without arguments, create and initialize a blank Table, otherwise
        intelligently call one of the Table.from_X functions, depending on the arguments.
        Also passes through pandas.DataFrame constructor keyword arguments.
        Do not pass positional arguments through to pandas.
        """
        # is pandas is calling this as part of its transformations, pass it through
        all_kwargs_are_pandas = len(set(kwargs.keys()).difference(cls.KNOWN_PANDAS_KWARGS)) == 0

        ##### START PANDAS SUBCLASS COMPATIBILITY SECTION #####
        # this compatibility hack must exist because we override __new__ in a way incompatible
        # with pandas' subclassing scheme---hacks are needed for compatibility, read on

        # if we called the constructor without arguments, create empty
        # all possible pandas kwargs will be passed to __init__, where
        # pandas will be able to process them
        if not args and (not kwargs or all_kwargs_are_pandas):
            return super().__new__(cls)

        # pandas can call this constructor (through Table._constructor) internally
        # when doing transformations: its signatures are either a BlockManager as the sole arg,
        # or a ndarray as the sole arg, accompanied by at least one kwarg
        # this serves the purpose of transforming the first arg into its matching kwarg,
        # which will then be processed by the if clause a couple of lines above this
        if len(args) == 1 and (isinstance(args[0], pd.core.internals.BlockManager)
                               or (isinstance(args[0], np.ndarray) and len(kwargs) != 0 and all_kwargs_are_pandas)):
            return cls(data=args[0], **kwargs)
        ##### END PANDAS SUBCLASS COMPATIBILITY SECTION #####

        if 'filename' in kwargs:
            args = [kwargs.pop('filename')]

        if not args:
            raise TypeError("Table takes at least 1 positional argument (0 given))")

        if isinstance(args[0], str):
            if args[0].startswith('https://') or args[0].startswith('http://'):
                return cls.from_url(args[0], **kwargs)
            else:
                return cls.from_file(args[0], **kwargs)
        elif isinstance(args[0], TableBase):
            return cls.from_table(args[0].domain, args[0])
        elif isinstance(args[0], Domain):
            domain, args = args[0], args[1:]
            if not args:
                return cls.from_domain(domain, **kwargs)
            if isinstance(args[0], TableBase):
                return cls.from_table(domain, *args)
            elif isinstance(args[0], list):
                return cls.from_list(domain, *args)
            elif isinstance(args[0], pd.DataFrame):
                return cls.from_dataframe(args[0], domain=domain, **kwargs)
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

        # only pass through things known to pandas, e.g.
        # Table(..., weights=1) passes weights to from_numpy, but would error
        # when passing to pandas upstream
        super().__init__(**{k: v for k, v in kwargs.items() if k in self.KNOWN_PANDAS_KWARGS})

        # all weights initialized to 1 (see the weight functions for details)
        self.name = getattr(self, 'name', kwargs.get("name", "untitled"))
        self.attributes = getattr(self, 'attributes', kwargs.get("attributes", {}))
        self.__file__ = getattr(self, '__file__', kwargs.get("__file__"))

        # we need to filter the domain to only include the columns present in the table
        # but we still need to allow constructing an empty table (with no domain)
        # also, we only set the domain if it has changed (==number of variables),
        # so in those cases, id(domain_before) == id(domain_after)
        if hasattr(self, 'domain') and self.domain is not None:
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
        if self._WEIGHTS_COLUMN not in self.columns:
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
        result = cls(columns=domain.attributes + domain.class_vars + domain.metas)
        result.domain = domain
        return result

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
        # don't just plain copy here: in case of subclasses of Table, a plain table is passed
        # through the constructor and from_table to here, and expects to be converted
        # into a proper subclass type
        result = cls(data=source.iloc[row_indices]).copy()
        result._transfer_properties(source, transfer_domain=True)  # because we manually copy data, not the whole table
        return result

    @classmethod
    def from_table(cls, target_domain, source_table, row_indices=slice(None)):
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
        # if a series is passed, convert it to a frame
        if isinstance(source_table, SeriesBase):
            d = source_table.domain
            # to pandas df first to avoid adding a column/row of weights automatically
            source_table = cls(source_table.domain, pd.DataFrame(source_table).transpose())

        new_cache = cls.conversion_cache is None
        try:
            if new_cache:
                cls.conversion_cache = {}
            else:
                cached = cls.conversion_cache.get((id(target_domain), id(source_table)))
                if cached is not None:
                    return cached
            if target_domain == source_table.domain:
                return cls.from_table_rows(source_table, row_indices)

            result = cls()
            conversion = target_domain.get_conversion(source_table.domain)

            for conversion, target_column in zip(chain(conversion.variables, conversion.metas),
                                                 chain(target_domain.variables, target_domain.metas)):
                if isinstance(conversion, Number):
                    # mandatory copy
                    result[target_column.name] = source_table[source_table.domain[conversion]].copy()
                else:
                    # when converting a single instance (row), this results in a single value
                    # (and not e.g. a Series), so assigning to a table fails:
                    # we ensure it's at least a 1D ndarray, and not a 0D ndarray
                    result[target_column.name] = np.atleast_1d(conversion(source_table))
            result.domain = target_domain

            # if the new domain has 0 columns, the length of the table is also 0
            # and we can't set the previous weights or index (which are len(source_table))
            if len(result) != 0:
                result.set_weights(source_table.weights)
                result.index = source_table.index  # keep previous index
            result = result.iloc[row_indices]

            cls.conversion_cache[(id(target_domain), id(source_table))] = result

            # transform any values we believe are null into actual null values
            result.replace(to_replace=list(Variable.MISSING_VALUES), value=np.nan, inplace=True)
            result._transform_discrete_into_categorical()
            result._transform_timevariable_into_datetime()

            return result
        finally:
            if new_cache:
                cls.conversion_cache = None

    @classmethod
    def from_dataframe(cls, df, domain=None, reindex=False, weights=None):
        """
        Convert a pandas.DataFrame object to a Table.
        This can infer infers column variable types and roles, reindex and set weights.
        """
        if domain is None:
            result = cls._from_data_inferred(df)
        else:
            result = cls(data=df)
            result.domain = domain

        # transform any values we believe are null into actual null values
        result.replace(to_replace=list(Variable.MISSING_VALUES), value=np.nan, inplace=True)
        result._transform_discrete_into_categorical()
        result._transform_timevariable_into_datetime()

        if reindex:
            result.index = cls._new_id(len(result), force_list=True)
        if weights is not None:
            result.set_weights(weights)
        return result

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
        result = cls()

        # used for data inference, shape checks, consolidated access
        X_df = pd.DataFrame(data=X_or_data)
        Y_df = pd.DataFrame(data=Y)
        meta_df = pd.DataFrame(data=meta)

        # transform any values we believe are null into actual null values
        X_df.replace(to_replace=list(Variable.MISSING_VALUES), value=np.nan, inplace=True)
        Y_df.replace(to_replace=list(Variable.MISSING_VALUES), value=np.nan, inplace=True)
        meta_df.replace(to_replace=list(Variable.MISSING_VALUES), value=np.nan, inplace=True)

        # override, because the user wishes to specify roles manually
        if Y is not None or meta is not None:
            infer_roles = False

        # process every input segment with its intended role
        for df, initial_role in zip((X_df, Y_df, meta_df), ('x', 'y', 'meta')):
            for column_name, column in ((c, df[c]) for c in df.columns):
                t, r = Domain.infer_type_role(column, initial_role if not infer_roles else None)
                name = Domain.infer_name(t, r, column_name,
                                         role_vars['x'], role_vars['y'], role_vars['meta'])
                if t is DiscreteVariable:
                    var = t(name, values=DiscreteVariable.generate_unique_values(column))
                else:
                    var = t(name)
                result[var.name] = column
                role_vars[r].append(var)
        result.domain = Domain(role_vars['x'], role_vars['y'], role_vars['meta'])
        result.domain.anonymous = True
        result._transform_discrete_into_categorical()
        result._transform_timevariable_into_datetime()
        return result

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
            result = cls._from_data_inferred(X, Y, metas)
            if weights is not None:
                result.set_weights(weights)
            return result

        # ensure correct shapes (but not sizes) so we can iterate
        X = correct_shape(X)
        Y = correct_shape(Y)
        metas = correct_shape(metas)

        result = cls()
        for role_array, variables in zip((X, Y, metas),
                                         (domain.attributes, domain.class_vars, domain.metas)):
            if role_array is None:
                if variables:
                    raise ValueError("Variable and column count mismatch. ")
                continue
            if role_array.shape[1] != len(variables):
                raise ValueError("Variable and column count mismatch. ")
            for column, variable in zip(role_array.T, variables):
                result[variable.name] = column
        result.domain = domain
        if weights is not None:
            result.set_weights(weights)

        # transform any values we believe are null into actual null values
        result.replace(to_replace=list(Variable.MISSING_VALUES), value=np.nan, inplace=True)
        result._transform_discrete_values_to_descriptors()
        result._transform_discrete_into_categorical()
        result._transform_timevariable_into_datetime()

        return result

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

        # check row lengths
        # allow not specifying the class variable (but only that): set it to nan in all rows
        domain_columns = len(domain.variables) + len(domain.metas)
        for r in rows:
            if len(r) != domain_columns:
                if len(r) == len(domain.attributes):
                    r.extend([np.nan] * len(domain.class_vars))
                else:
                    raise ValueError("Data and domain column count mismatch. ")

        result = cls(data=rows,
                     columns=[a.name for a in chain(domain.attributes, domain.class_vars, domain.metas)])
        result.domain = domain

        # transform any values we believe are null into actual null values
        result.replace(to_replace=list(Variable.MISSING_VALUES), value=np.nan, inplace=True)
        result._transform_discrete_into_categorical()
        result._transform_timevariable_into_datetime()

        if weights is not None:
            result.set_weights(weights)
        return result

    @classmethod
    def from_file(cls, filename):
        """
        Read a data table from a file. The path can be absolute or relative.

        :param filename: File name
        :type filename: str
        :return: a new data table
        :rtype: Orange.data.Table
        """
        from Orange.data.io import FileFormat
        from Orange.data import dataset_dirs

        absolute_filename = FileFormat.locate(filename, dataset_dirs)
        reader = FileFormat.get_reader(absolute_filename)
        data = reader.read()

        # Readers return plain table. Make sure to cast it to appropriate
        # (subclass) type
        if cls != data.__class__:
            data = cls(data)

        data.name = os.path.splitext(os.path.split(filename)[-1])[0]
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

        # support using this in TableSeries, whose len gives the number of columns
        n_rows = 1 if isinstance(self, SeriesBase) else len(self)

        # preallocate result, we fill it in-place
        # we need a more general dtype for metas (commonly strings),
        # otherwise assignment fails later
        dtype = object if meta else \
            int if all(c.is_discrete for c in cols) else \
                None
        result = np.zeros((n_rows, len(cols)), dtype=object if meta else None)
        # effectively a double for loop, see if this is a bottleneck later
        for i, col in enumerate(cols):
            if isinstance(self, SeriesBase):
                # if this is used in TableSeries, we don't have a series but an element
                result[:, i] = col.to_val(self[col])
            else:
                result[:, i] = self[col].apply(col.to_val).values
        result.setflags(write=writable)
        return result

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
        result = self._to_numpy(Y=True)
        return result[:, 0] if result.shape[1] == 1 else result

    @property
    def metas(self):
        """
        Return a read-only numpy matrix of metas.
        The columns are in the same order as the columns in Table.domain.metas.
        """
        return self._to_numpy(meta=True)

    @property
    def weights(self):
        """Get the weights as a numpy array."""
        val = self[self._WEIGHTS_COLUMN]
        if hasattr(val, 'values'):
            return val.values
        else:
            # at least 1D when using this in TableSeries, which instead returns a 0D ndarray directly
            return np.atleast_1d(val)

    def set_weights(self, weight):
        """
        Set the weights for the instances in this table.
        If a number, weights to set to that value.
        If a string, weights are set to whatever the column with that name's values are,
            but only if those values are all numbers and are not NA/NaN.
        If a sequence of (non-NA/NaN) numbers, set those values as the sequence.
        """
        if isinstance(weight, Number):
            if np.isnan(weight):
                raise ValueError("Weights cannot be nan. ")
            self[self._WEIGHTS_COLUMN] = weight
        elif isinstance(weight, str):
            if weight not in self.columns:
                raise ValueError("{} is not a column.".format(weight))
            if self[weight].isnull().any() and np.issubdtype(self[weight].dtype, Number):
                raise ValueError("All values in the target column must be valid numbers.")
            self[self._WEIGHTS_COLUMN] = self[weight].fillna(value=self[weight].median())
        elif isinstance(weight, (Sequence, np.ndarray)):  # np.ndarray is not a Sequence
            if len(weight) != len(self):
                raise ValueError("The sequence has length {}, expected length {}.".format(len(weight), len(self)))
            self[self._WEIGHTS_COLUMN] = weight
        elif isinstance(weight, pd.Series):  # not only SeriesBase
            # drop everything but the values to uncomplicate things
            if weight.isnull().any():
                raise ValueError("Weights cannot be nan. ")
            self[self._WEIGHTS_COLUMN] = weight.values
        elif isinstance(weight, pd.Categorical):
            self[self._WEIGHTS_COLUMN] = list(weight)
        else:
            raise TypeError("Expected one of [Number, str, Sequence, SeriesBase].")

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

    def _transform_discrete_values_to_descriptors(self):
        """
        Transform discrete variables given in descriptor index form
        into actual descriptors.
        """
        for var in chain(self.domain.variables, self.domain.metas):
            if isinstance(var, DiscreteVariable):
                # only transform the values if all of them appear to be integers and
                # could act as a variable value index
                # otherwise we're dealing with numeric discretes
                is_values = self[var.name].apply(lambda v: isinstance(v, Number) and
                                                           (isinstance(v, int) or v.is_integer()) and
                                                           v < len(var.values)).all()
                if is_values:
                    self[var.name] = self[var.name].apply(lambda v: var.values[int(v)])

    def _transform_discrete_into_categorical(self):
        """
        Transform discrete variables into pandas' categorical.
        This must be done after replacing null values because those aren't values,
        and also after transforming discretes to descriptors.
        """
        for var in chain(self.domain.variables, self.domain.metas):
            if isinstance(var, DiscreteVariable):
                self[var.name] = pd.Categorical(self[var.name], categories=var.values, ordered=var.ordered)

    def _transform_timevariable_into_datetime(self):
        for var in chain(self.domain.variables, self.domain.metas):
            if isinstance(var, TimeVariable):
                self[var.name] = var.column_to_datetime(self[var.name])

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
                raise IOError("Writing of {}s is not supported".format(desc.lower()))
            else:
                raise IOError("Unknown file name extension.")
        writer.write_file(filename, self)

    def __getitem__(self, item):
        # if selecting a column subset, we need to transfer weights so they don't just disappear
        # only do this for multiple column selection, which returns a DataFrame by contract
        if isinstance(item, (Sequence, pd.Index)) and not isinstance(item, str) \
                and all(isinstance(i, str) for i in item) \
                and self._WEIGHTS_COLUMN not in item:
            item = list(item) + [self._WEIGHTS_COLUMN]
        return super().__getitem__(item)

    def __setitem__(self, key, value):
        # if the table has an empty index and we're inserting a new row,
        # the index would be created by pandas automatically.
        # we want to maintain unique indices, so we override the index manually.
        # we also need to set default weights, lest they be NA
        new_index_and_weights = len(self.index) == 0

        # PANDAS CONTRACT BREAKAGE:
        # the pandas default behaviour when adding a new column as a Series is that
        # the indexes are inner-joined: only the elements at the indices that exist
        # in the current table are actually set, other elements are NA
        # for easier handling of Orange behaviour, we effectively ignore the index
        # on the series to just merge the column into the table
        if isinstance(value, pd.Series) and not new_index_and_weights:
            value.index = self.index
        super().__setitem__(key, value)

        if new_index_and_weights:
            new_id = self._new_id(len(self), force_list=True)
            self.index = new_id
            # super call because we'd otherwise recurse back into this
            super().__setitem__(self._WEIGHTS_COLUMN, 1)

    def clear(self):
        """Remove all rows from the table in-place."""
        self.drop(self.index, inplace=True)

    def append(self, other, ignore_index=False, verify_integrity=False):
        """
        Append a new row to the table, returning a new Table.
        row can be a list-like of a single row, TableSeries (a single row slice) or a Table.
        """
        if not isinstance(other, pd.DataFrame):
            other = pd.DataFrame(data=[other],
                                 columns=[c for c in self.columns if c != self._WEIGHTS_COLUMN],
                                 index=[0])
        other.index = self._new_id(len(other), force_list=True)
        if self._WEIGHTS_COLUMN not in other.columns:
            other[self._WEIGHTS_COLUMN] = 1

        # coerce incompatibilities: this happens when appending a list
        #  - category dtypes must match, coerce them
        for i, column in enumerate(self.columns):
            if self.dtypes[i].name == 'category':
                new_cats = [v for v in set(other[column]) if v not in self[column].cat.categories and v not in Variable.MISSING_VALUES]
                self[column] = self[column].cat.add_categories(new_cats)
                self.domain[column].values += new_cats
                other[column] = other[column].astype('category',
                                                     categories=self[column].cat.categories,
                                                     ordered=self[column].cat.ordered)
        result = super().append(other)
        # transform any values we believe are null into actual null values
        result.replace(to_replace=list(Variable.MISSING_VALUES), value=np.nan, inplace=True)
        # append doesn't transfer properties for some reason
        result._transfer_properties(self)
        return self.from_dataframe(result, self.domain)

    @deprecated('Use Table.append() for adding new rows. This inserts a new column. ')
    def insert(self, *args, **kwargs):
        super().insert(*args, **kwargs)

    @deprecated("Table.append(...)")
    def extend(self, rows, weight=1):
        return self.append(rows)

    @classmethod
    def concatenate(cls, tables, axis=1, reindex=True, colstack=True, rowstack=False):
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
                # this is a bit convoluted because we can't chain renames
                newtables = [tables[0]]
                for t in tables[1:]:
                    new = t.copy()
                    new.columns = tables[0].columns
                    newtables.append(new)
                result = pd.concat(newtables, axis=0)
            else:
                result = pd.concat(tables, axis=0)
            new_index = cls._new_id(len(result))
            new_domain = tables[0].domain
            result.index = new_index
        elif axis == CONCAT_COLS:

            # check for same name
            columns = [v for v in flatten([list(t.columns) for t in tables]) if v != cls._WEIGHTS_COLUMN]
            if len(set(columns)) != len(columns):
                raise ValueError("Cannot concatenate domains with same names.")
            if colstack:
                # check for same length
                if len(set(len(t) for t in tables)) != 1:
                    raise ValueError("Cannot colstack tables with differing numbers of rows. ")
                # reset index temporarily because this joins by index by default
                result = pd.concat([t.reset_index(drop=True) for t in tables], axis=1, join_axes=[tables[0].index])
            else:
                result = pd.concat(tables, axis=1)

            # merge domains
            new_domain = Domain(
                list(flatten(t.domain.attributes for t in tables)),
                list(flatten(t.domain.class_vars for t in tables)),
                list(flatten(t.domain.metas for t in tables))
            )

            # fix multiple weight columns: keep the first one
            weight_columns = result[cls._WEIGHTS_COLUMN]
            result = result.drop(cls._WEIGHTS_COLUMN, axis=1)
            result[cls._WEIGHTS_COLUMN] = weight_columns.iloc[:, 0]

            if reindex:
                new_index = cls._new_id(len(result))
                result.index = new_index
        else:
            raise ValueError('axis {} out of bounds [0, 2)'.format(axis))
        result._transfer_properties(tables[0])  # pd.concat does not do this by itself
        return cls.from_dataframe(result, new_domain)

    def _transfer_properties(self, from_table, transfer_domain=False):
        """
        Transfer properties (such as the name) to this table.
        This should normally not be used, but it is used when these properties
        are not automatically transferred on manipulation, in particular when using pd.concat.
        """
        for name in self._metadata:
            if hasattr(from_table, name) and (transfer_domain or name != "domain"):
                setattr(self, name, getattr(from_table, name))

    def approx_len(self):
        return len(self)

    def exact_len(self):
        return len(self)


    def has_missing(self):
        """Return `True` if there are any missing attribute or class values."""
        # manual access to columns because dumping to a numpy array (with self.X) is slower
        return self[self.domain.attributes].isnull().any().any() or self.has_missing_class()

    def has_missing_class(self):
        """Return `True` if there are any missing class values."""
        return self[self.domain.class_vars].isnull().any().any()

    def iterrows(self):
        """
        An override to return TableSeries instead of Series (pandas doesn't do that by default).
        """
        # super here is the next item in the MRO, e.g. pd.DataFrame or pd.SparseDataFrame
        gen = super().iterrows()
        for item in gen:
            yield (item[0], self._constructor_sliced(item[1]))

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

    def _compute_basic_stats(self, columns=None, include_metas=False):
        """
        Compute basic stats for each of the columns.

        This is a legacy method and should be avoided and/or replaced where possible.

        :param columns: columns to calculate stats for. None = all of them
        :return: A np.ndarray of (min, max, mean, var, #nans, #non-nans) rows for each column.
        """
        selected_columns = columns
        if columns is None:
            selected_columns = []
            selected_columns += self.domain.attributes
            selected_columns += self.domain.class_vars
            if include_metas:
                selected_columns += self.domain.metas

        # process continuous and discrete (== pandas categorical) separately for some,
        # because including a categorical in the computation of e.g. mean,
        # which is undefined for a categorical, returns an empty table
        selected_continuous = [c for c in selected_columns if c.is_continuous]
        selected_discrete = [c for c in selected_columns if not c.is_continuous]
        assert len(selected_columns) == len(selected_continuous) + len(selected_discrete)

        # discrete
        d_means = pd.Series({k: np.nan for k in selected_discrete})
        d_varc = pd.Series({k: np.nan for k in selected_discrete})

        subset_continuous = self[selected_continuous]
        c_means = subset_continuous.mean(axis=0)
        c_varc = subset_continuous.var(axis=0)

        # these are the same for both
        subset_all = self[selected_columns]
        mins = subset_all.min(axis=0)
        maxes = subset_all.max(axis=0)
        nans = subset_all.isnull().sum(axis=0)
        nonnans = subset_all.count(axis=0) - nans

        # ensure correct ordering and remove the weights column
        res = pd.DataFrame([mins, maxes, pd.concat((d_means, c_means)), pd.concat((d_varc, c_varc)), nans, nonnans])
        res = res[selected_columns]
        return res.values.T

    def _compute_distributions(self, columns=None):
        """
        Compute distribution of values for the given columns.

        :param columns: columns to calculate distributions for
        :return: a list of distribution tuples. Type of distribution depends on the
                 type of the column:
                   - for discrete, distribution is a 1d np.array containing the
                     occurrence counts for each of the values.
                   - for continuous, distribution is a 2d np.array with
                     distinct (ordered) values of the variable in the first row
                     and their counts in second.
                 The second element of each tuple is the number of NA values of the column.
        """
        if columns is None:
            columns = self.domain.attributes + self.domain.class_vars + self.domain.metas
        distributions = []
        for col in columns:
            var = self.domain[col]
            # use groupby instead of value_counts so we can use weighed data
            # also fill all unknown values with 0 because that's what NA means in this context
            weighed_counts = self.groupby(col)[self._WEIGHTS_COLUMN].sum().fillna(0)
            unknowns = self[col].isnull().sum()
            if var.is_discrete:
                if var.ordered:
                    distributions.append((np.array([weighed_counts.loc[val] for val in var.values]), unknowns))
                else:
                    distributions.append((weighed_counts.values, unknowns))
            else:
                distributions.append((np.array(sorted(weighed_counts.iteritems())).T, unknowns))
        return distributions

    def _compute_contingency(self, col_vars=None, row_var=None):
        """
        Compute contingency matrices for one or more discrete or
        continuous variables against the specified discrete variable.

        The result is a tuple of (list of contingencies, num missing row_var values).

        The list of contingencies contains a pair for each column variable.
        The first element contains the contingencies (this changes depending on the variable type; see below),
        and the second element is a 1D numpy array, where each element is the count of missing
        column variable elements for the respective row variable value.

        The format of contingencies returned depends on the variable type:
        - for discrete variables, it is a numpy array, where
          element (i, j) contains the count of rows with the i-th value of the
          row variable and the j-th value of the column variable.
        - for continuous variables, the contingency is a list of two arrays,
          where the first array contains ordered distinct values of the
          column_variable and the element (i, j) of the second array
          contains count of rows with i-th value of the row variable
          and j-th value of the ordered column variable.

        The final output structure looks like:
        result (tuple)
         |__contingencies (list of len(col_vars))
         |   |__contingency, one of
         |   |   |__2D np.array
         |   |   |__tuple(sorted list of continuous values, 2D np.array)
         |   |__1D np.array (missing column elements for each row value)
         |__num missing row_var values (int)

        :param col_vars: variables whose values will correspond to columns of
            contingency matrices
        :type col_vars: list of ints, variable names or descriptors of type
            :obj:`Orange.data.Variable`
        :param row_var: a discrete variable whose values will correspond to the
            rows of contingency matrices
        :type row_var: int, variable name or :obj:`Orange.data.DiscreteVariable`
        """
        if row_var is None:
            row_var = self.domain.class_var
            if row_var is None:
                raise ValueError("No row variable. ")
        if col_vars is None:
            col_vars = self.domain.attributes

        row_var = self.domain[row_var]
        col_vars = [self.domain[v] for v in col_vars]
        if not row_var.is_discrete:
            raise TypeError("Row variable must be discrete. ")
        if any(not (var.is_discrete or var.is_continuous) for var in col_vars):
            raise ValueError("Contingency can be computed only for discrete and continuous values. ")

        # assertions at this point:
        #  - row_var exists and is discrete
        #  - col_vars is a list of continuous or discrete variables
        contingencies = []
        unknown_grouper = self.groupby(row_var)
        for var in col_vars:
            if var is row_var:
                # we can't do a pivot table of a variable with itself easily,
                # so instead just compute the distributions, which are equivalent
                # row_var is always discrete
                dist_unks = self._compute_distributions(columns=[row_var])
                contingencies.append(dist_unks[0])
            else:
                # we limit ourselves to counting the weights (we only need the count, NAs don't matter)
                # so we get a slimmer result and hopefully faster processing
                pivot = pd.pivot_table(self, values=self._WEIGHTS_COLUMN, index=[row_var],
                                       columns=[var], aggfunc=np.sum).fillna(0)
                unknowns = unknown_grouper[var].agg(lambda x: x.isnull().sum())
                if var.is_discrete:
                    contingencies.append((pivot.values, unknowns.values))
                else:
                    contingencies.append(((pivot.columns.values, pivot.values), unknowns.values))
        return contingencies, self[row_var].isnull().sum()

    def density(self):
        raise NotImplementedError

    def X_density(self):
        """Get an enum value for the self.X density status. """
        raise NotImplementedError

    def Y_density(self):
        """Get an enum value for the self.Y density status. """
        raise NotImplementedError

    def metas_density(self):
        """Get an enum value for the self.metas density status. """
        raise NotImplementedError

    def is_sparse(self):
        raise NotImplementedError

    def is_dense(self):
        return not self.is_sparse()


class SeriesBase:
    """
    A common superclass for Series (as in pd.Series or pd.SparseSeries) objects.
    Transfers Table x/y/metas/weights functionality to the Series.
    """
    _metadata = ['domain']

    # use the same functions as in Table for this
    # WARNING: depends on TableSeries having a domain, which is ensured
    # in Table._constructor_sliced
    _to_numpy = TableBase._to_numpy
    X = TableBase.X
    Y = TableBase.Y
    metas = TableBase.metas
    weights = w = TableBase.weights


class PanelBase:
    """
    A common superclass for Panel (as in pd.Panel or pd.SparsePanel) objects.
    """
    pass
