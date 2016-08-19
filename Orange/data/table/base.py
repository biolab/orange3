import os
from collections import Sequence
from threading import Lock
from numbers import Number

from itertools import chain
import numpy as np
import scipy.sparse as sp
import pandas as pd

from Orange.data import Domain, DiscreteVariable, Variable, TimeVariable, filter_visible
from Orange.util import flatten, deprecated


class _transferer:
    """A 'minor' hack for transferring attributes to TableSeries in _constructor_sliced."""
    def __init__(self, cls, attrs):
        self.cls = cls
        self.attrs = attrs

    # this is a class and not a function because sometimes, pandas
    # wants _constructor_sliced.from_array
    def from_array(self, *args, **kwargs):
        return self._attr_setter(self.cls.from_array(*args, **kwargs))

    def __call__(self, *args, **kwargs):
        return self._attr_setter(self.cls(*args, **kwargs))

    def _attr_setter(self, target):
        for k, v in self.attrs.items():
            setattr(target, k, v)
        return target


# noinspection PyPep8Naming
class TableBase:
    """An abstract base class for data storage structures in Orange."""

    # a list of all kwargs supported by this class,
    # used for proper compatibility with pandas' constructors
    _ORANGE_KWARG_NAMES = {"weights", "attributes"}

    # the default name for weights columns
    # subclasses may override this to rename the column
    _WEIGHTS_COLUMN = "__weights__"

    # a collection of all known internal column names
    # use case: subclasses add additional columns (not only weights) and we need to know
    # of all existing names to be able to process things wrt these internals,
    # e.g. adding missing columns in the constructor, transparently passing them through
    # when selecting a subset
    _INTERNAL_COLUMN_NAMES = [_WEIGHTS_COLUMN]

    # a counter for indexing rows, important for deterministically selecting rows
    # and keeping pandas indices sane
    _next_instance_id = 0
    _next_instance_lock = Lock()

    # the conversion cache for converting tables to different domains
    _conversion_cache = None

    # custom properties, preserved through pandas manipulations
    _metadata = ['name',  # the table name
                 'domain',  # the domain assigned to the table
                 'attributes',  # any custom attributes the table has
                 '__file__',  # if read from a file, the filename of that file
                 '_already_inited']  # skip multiple constructor calls, see __init__

    @classmethod
    def _is_orange_construction_path(cls, *args, **kwargs):
        """Determine whether the constructor was called as part of the Orange construction path.

        This exists for pandas compatibility: we need to know whether we are being
        called by pandas internals (through _constructor) or by Orange directly.

        Uses cls._ORANGE_KWARG_NAMES to filter kwargs.

        Parameters
        ----------
        args : tuple
            The args passed to the constructor.
        kwargs : dict
            The kwargs passed to the constructor.

        Returns
        -------
        bool
            Whether we are being called by Orange directly, not pandas internals.
        """
        # all pandas kwargs to determine what construction path to take
        non_orange_kwargs = {k: v for k, v in kwargs.items() if k not in cls._ORANGE_KWARG_NAMES}

        single_arg_types = (str, TableBase, pd.DataFrame, np.ndarray, sp.spmatrix, list)
        after_domain_arg_types = (TableBase, pd.DataFrame, np.ndarray, sp.spmatrix, list)

        # to preserve compatibility with pandas' constructor, we except our own behaviour here
        # this is done solely on the basis of args, because we've stripped the known kwargs
        single_argument = len(args) > 0 and isinstance(args[0], single_arg_types)
        multi_argument = ((len(args) == 1 and isinstance(args[0], Domain)) or
                          (len(args) > 1 and (args[0] is None or isinstance(args[1], after_domain_arg_types))))
        return len(non_orange_kwargs) == 0 and (single_argument or multi_argument)

    def __new__(cls, *args, **kwargs):
        """Create a new Table, the exact result type depends on the data passed.

        If called without arguments, create and initialize a blank Table, otherwise
        intelligently call one of the TableBase.from_X functions, depending on the arguments.
        Also passes through pandas.DataFrame constructor keyword arguments.
        Does not pass positional arguments through to pandas, unless called from pandas internals.

        Parameters
        ----------
        args
            One of:
             - (str): create a Table from a file or URL
             - (list): create a table based on a list of rows, infer the domain.
             - (Domain): create an empty table with the domain
             - (TableBase): initialize from another TableBase,
             - (pd.DataFrame): create a table based in an existing pandas DataFrame,
               inferring the domain.
             - (np.ndarray or scipy.sparse): a dense or sparse matrix, resulting
               Table data type depends on the type passed, infer a domain
                - Up to four may be passed to fit into X/Y/metas/weights respectively.
             - (Domain, np.ndarray or scipy.sparse): Same as above, without domain inference
             - (Domain, TableBase): convert another TableBase object to the passed domain
             - (Domain, list): create a Table based on a list of rows.
             - (Domain, pd.DataFrame): create a table based on an existing pandas DataFrame and Domain.
        kwargs
            Used exclusively for compatibility with pandas.

        Returns
        -------
        Table or SparseTable
        A new Table or SparseTable, depending on the arguments passed and the calling class.
        """
        if cls._is_orange_construction_path(*args, **kwargs):
            if isinstance(args[0], str):
                if args[0].startswith(('ftp://', 'http://', 'https://')):
                    return cls.from_url(args[0], **kwargs)
                else:
                    return cls.from_file(args[0], **kwargs)
            elif isinstance(args[0], TableBase):
                return cls.from_table(args[0].domain, args[0], **kwargs)
            elif isinstance(args[0], pd.DataFrame):
                return cls.from_dataframe(None, args[0], **kwargs)
            elif isinstance(args[0], Domain):
                domain, args = args[0], args[1:]
                if not args:
                    return cls.from_domain(domain, **kwargs)
                if isinstance(args[0], TableBase):
                    return cls.from_table(domain, *args, **kwargs)
                elif isinstance(args[0], list):
                    return cls.from_list(domain, *args, **kwargs)
                elif isinstance(args[0], pd.DataFrame):
                    return cls.from_dataframe(domain, args[0], **kwargs)
            else:
                domain = None
            return cls.from_numpy(domain, *args, **kwargs)
        else:
            return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        """Initialize the object.

        Sets the name and other attributes and calls pandas' initialization.
        Filters the domain: retains only the columns actually in the table.
        Sets the weights if they are not set already.
        """
        if getattr(self, '_already_inited', False):
            # we may have already been initialized before
            # e.g. __new__ constructs a complete, initialized object, which
            # then passes through this __init__ again, unnecessarily
            # avoid this by quickly exiting
            return

        super().__init__(*args, **kwargs)

        # these won't override things that are already set, as we have
        # the _already_inited short-circuit in place
        self.name = 'untitled'
        self.domain = None
        self.attributes = kwargs.get('attributes', {})
        self.__file__ = None

        # only set the weights if they aren't set already
        if self._WEIGHTS_COLUMN not in self.columns:
            self.set_weights(1)

        self._already_inited = True

    def __finalize__(self, from_table, **kwargs):
        """Transfer properties from _metadata and filter the domain.

        This extends the pandas __finalize__ which transfers attributes from _metadata,
        and adds filtering the domain. See the examples for more.

        Parameters
        ----------
        from_table : TableBase or dict
            The table to source the attributes from.

        Examples
        --------
            >>> from Orange.data import Table
            >>> i = Table('iris')
            >>> i.attributes["attr"] = "val"
            >>> i.domain
            [sepal length, sepal width, petal length, petal width | iris]
            >>> sub = i[['sepal length', 'iris']]
            >>> sub.domain
            [sepal length | iris]
            >>> sub.attributes
            {'attr': 'val'}

        Notes
        -----
        This is automatically called by pandas internally *almost* every time
        a new object is constructed, e.g. when slicing rows, but not when things are
        ambiguous, like concatenation.

        This does not work with Series (rows or columns).
        """
        result = super(TableBase, self).__finalize__(from_table, **kwargs)
        if self.domain is not None:
            var_filter = lambda vars: [var for var in vars if var in result.columns]
            filtered_domain = Domain(
                var_filter(result.domain.attributes),
                var_filter(result.domain.class_vars),
                var_filter(result.domain.metas)
            )
            # we only want to set the domain if it has changed (==number of variables),
            # so in those cases, id(domain_before) == id(domain_after)
            if len(filtered_domain.variables) + len(filtered_domain.metas) != \
                            len(result.domain.variables) + len(result.domain.metas):
                result.domain = filtered_domain
        return result

    @classmethod
    def from_domain(cls, domain):
        """Construct a new Table for the given Domain.

        Parameters
        ----------
        domain : Domain
            The domain passed to the new Table object.

        Returns
        -------
        TableBase
            The new Table with a Domain assigned.
        """
        result = cls(columns=domain.attributes + domain.class_vars + domain.metas)
        result.domain = domain
        return result

    @classmethod
    def from_table(cls, target_domain, source_table, row_indices=slice(None)):
        """Create a new Table as a conversion from the source Table's domain to the new one.

        Row indices may be given to select a subset at the same time.
        The new domain defines the columns of the new table - new columns are computed
        on the fly from source columns whenever appropriate.

        Parameters
        ----------
        target_domain : Domain
            The new domain. Defines the resulting columns and the transformation
            from the original Table to the new one.
        source_table : TableBase
            The source table.
        row_indices : slice or list[int] or list[bool], optional
            The row indices to select a subset of the resulting table.

        Returns
        -------
        TableBase
            A table converted from the source domain to the target domain.
        """
        # if a series is passed, convert it to a frame
        if isinstance(source_table, SeriesBase):
            d = source_table.domain
            # to pandas df first to avoid adding a column/row of weights automatically
            source_table = cls(source_table.domain, pd.DataFrame(source_table).transpose())

        new_cache = cls._conversion_cache is None
        try:
            if new_cache:
                cls._conversion_cache = {}
            else:
                cached = cls._conversion_cache.get((id(target_domain), id(source_table)))
                if cached is not None:
                    return cached
            if target_domain == source_table.domain:
                # intentional casting to subclass
                result = cls(data=source_table.iloc[row_indices])
                # because we manually copy data, not the whole table
                result.__finalize__(source_table)
                return result

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

            # if the new domain has 0 columns, the length of the resulting table is also 0
            # and we can't set the previous weights or index (which are len(source_table))
            if len(result) != 0:
                result.set_weights(source_table.weights)
                result.index = source_table.index  # keep previous index
                result = result.iloc[row_indices]

            cls._conversion_cache[(id(target_domain), id(source_table))] = result

            result._transform_discrete_into_categorical()
            result._transform_timevariable_into_datetime()

            return result
        finally:
            if new_cache:
                cls._conversion_cache = None

    @classmethod
    def from_dataframe(cls, domain, df, reindex=False, weights=None):
        """Create a new Table from a pandas DataFrame.

        Parameters
        ----------
        domain : Domain or None
            The domain to use for the result. If None, the domain is inferred.
        df : pd.DataFrame
            The source pandas DataFrame.
        reindex : bool, default False
            Whether to override the original DataFrame's index with a new one.
        weights : anything TableBase.set_weights accepts, optional, default None
            The weights to use on the resulting DataFrame.

        Returns
        -------
        TableBase
            A new Table generated from the source DataFrame.
        """
        if domain is None:
            result = cls._from_data_inferred(df)
        else:
            result = cls(data=df)
            result.domain = domain

        result._transform_discrete_into_categorical()
        result._transform_timevariable_into_datetime()

        if reindex:
            result.index = cls._new_id(len(result), force_list=True)
        if weights is not None:
            result.set_weights(weights)
        return result

    @classmethod
    def _from_data_inferred(cls, X_or_data, Y=None, meta=None, infer_roles=True):
        """Create a Table and infer its domain.

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

        Where possible, column names are preserved form the input, otherwise they are named
        "Feature <n>", "Class <n>", "Target <n>" or "Meta <n>".
        The domain is marked as anonymous.

        Parameters
        ----------
        X_or_data : np.ndarray or pd.DataFrame
            Either the X component of the data as a numpy ndarray or a complete pandas DataFrame.
            If this is the only data argument, the column roles are also inferred.
        Y : np.ndarray, optional, default None
            The Y component of the data. Always assigned to Y in the result.
        meta : np.ndarray, optional, default None
            The meta attributes of the data. Always assigned to metas in the result.
        infer_roles : bool, optional, default True
            Whether to enable column role inference.

        Returns
        -------
        TableBase
            Return a new Table with the inferred domain.
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
        for df, initial_role in ((X_df, 'x'), (Y_df, 'y'), (meta_df, 'meta')):
            for column_name in df.columns:
                column = df[column_name]
                var_type, var_role = Domain.infer_type_role(column, initial_role if not infer_roles else None)
                name = Domain.infer_name(var_type, var_role, column_name,
                                         role_vars['x'], role_vars['y'], role_vars['meta'])
                if var_type is DiscreteVariable:
                    var = var_type(name, values=DiscreteVariable.generate_unique_values(column))
                else:
                    var = var_type(name)
                result[var.name] = column
                role_vars[var_role].append(var)
        result.domain = Domain(role_vars['x'], role_vars['y'], role_vars['meta'])
        result.domain.anonymous = True
        result._transform_discrete_into_categorical()
        result._transform_timevariable_into_datetime()
        return result

    @classmethod
    def from_numpy(cls, domain, X, Y=None, metas=None, weights=None):
        """Construct a table from numpy arrays with the given domain.

        The number of variables in the domain must match the number of columns in the
        corresponding arrays. All arrays must have the same number of rows.

        Parameters
        ----------
        domain : Domain or None
            If None, the domain is inferred from the data. Otherwise, specifies
            the column assignment to the new Table.
        X : np.ndarray
            The X component of the data (or undetermined, depending on the domain).
        Y : np.ndarray, optional, default None
            The Y component of the data.
        metas : np.ndarray, optional, default None
            The meta attributes of the data.
        weights : anything TableBase.set_weights accepts, optional, default None
            The weights to use for the resulting Table.
        Returns
        -------
        TableBase
            A new Table constructed from the given data.
        """
        # if anything is sparse, use the sparse version
        from .impl import SparseTable
        if any(sp.issparse(arr) for arr in (X, Y, metas, weights)) or issubclass(cls, SparseTable):
            # explicitly construct a sparse table as this can be called from Table(...),
            # but allow subclasses to call correctly
            if issubclass(cls, SparseTable):
                return cls._from_sparse_numpy(domain, X, Y, metas, weights)
            else:
                return SparseTable._from_sparse_numpy(domain, X, Y, metas, weights)

        if domain is None:
            result = cls._from_data_inferred(X, Y, metas)
            if weights is not None:
                result.set_weights(weights)
            return result

        # ensure correct shapes (but not sizes) so we can iterate
        def correct_shape(what):
            if what is None or what.ndim == 2:
                return what
            else:
                return np.atleast_2d(what).T

        X = correct_shape(X)
        Y = correct_shape(Y)
        metas = correct_shape(metas)

        result = cls()
        for role_array, variables in ((X, domain.attributes), (Y, domain.class_vars), (metas, domain.metas)):
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

        result._transform_discrete_values_to_descriptors()
        result._transform_discrete_into_categorical()
        result._transform_timevariable_into_datetime()

        return result

    @classmethod
    def from_list(cls, domain, rows, weights=None):
        """Construct a table from a list of rows.

        Parameters
        ----------
        domain : Domain
            The domain to use for the data.
        rows : list[list[obj]]
            A list of rows. Must be rectangular with the number of columns
            matching the columns in the domain.
        weights : anything TableBase.set_weights accepts, optional, default None
            The weights to use for the resulting Table.

        Returns
        -------
        TableBase
            A Table constructed from the given data.
        """
        if weights is not None and len(rows) != len(weights):
            raise ValueError("Mismatching number of instances and weights.")
        # check dimensions, pandas raises a very nondescript error
        row_width = len(rows[0])
        domain_columns = len(domain.variables) + len(domain.metas)
        if row_width != domain_columns or any(len(r) != domain_columns for r in rows):
            raise ValueError("Inconsistent number of columns.")

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
        """Read a Table from a file.

        Parameters
        ----------
        filename : str
            The relative or absolute filename for the file from which to read the data.

        Returns
        -------
        TableBase
            A table constructed from the given file.
        """
        from Orange.data.io import FileFormat
        from Orange.data import dataset_dirs

        absolute_filename = FileFormat.locate(filename, dataset_dirs)
        reader = FileFormat.get_reader(absolute_filename)
        data = reader.read()

        # Readers return plain table. Make sure to cast it to appropriate (subclass) type
        if cls != data.__class__:
            data = cls(data)

        data.name = os.path.splitext(os.path.split(filename)[-1])[0]
        data.__file__ = absolute_filename
        return data

    @classmethod
    def from_url(cls, url):
        """Read a table from a URL.

        Parameters
        ----------
        url : str
            The URL to read the table from.

        Returns
        -------
        TableBase
            A table constructed from the given URL.
        """
        from Orange.data.io import UrlReader
        reader = UrlReader(url)
        data = reader.read()
        if cls != data.__class__:
            data = cls(data)
        return data

    def _to_numpy(self, X=False, Y=False, meta=False, writable=False):
        """Export a transformed numpy matrix.

        The order is always X, Y, meta. Always 2D.
        The columns are in the same order as in Table.domain._.
        If writable == False (default), the numpy writable flag is set to false.
        This means write operations on this array will loudly fail.

        This is not the same as .values, but instead transforms the attributes to
        numeric values (where possible); e.g. uses var.values indices instead of
        actual descriptors.

        Parameters
        ----------
        X : bool, default False
            Whether to include the domain attributes in the result.
        Y : bool, default False
            Whether to include the domain class variables in the result.
        meta : bool, default False
            Whether to include the domain metas in the result.
        writable : bool, default False
            Whether to mark the resulting domain as writable.
        Returns
        -------
        np.ndarray
            The numpy array of the selected and transformed table data.
        """
        cols = []
        if X:
            cols += self.domain.attributes
        if Y:
            cols += self.domain.class_vars
        if meta:
            cols += self.domain.metas

        # support using this method in TableSeries, whose len gives the number of columns
        # (because it's a row slice)
        n_rows = 1 if isinstance(self, SeriesBase) else len(self)

        # preallocate result, we fill it in-place
        # we need a more general dtype for metas (commonly strings),
        # otherwise assignment fails later
        result = np.zeros((n_rows, len(cols)), dtype=object if meta else None)
        # effectively a double for loop, see if this is a bottleneck later
        for i, col in enumerate(cols):
            if isinstance(self, SeriesBase):
                # if this is used in TableSeries, we don't have a series but an element,
                # because we are iterating over a row
                result[:, i] = col.to_val(self[col])
            else:
                result[:, i] = col.to_val(self[col]).values
        result.setflags(write=writable)
        return result

    @property
    def X(self):
        """Return a read-only 2D numpy array of X.

        The columns are in the same order as the columns in Table.domain.attributes.

        Returns
        -------
        np.ndarray
        """
        return self._to_numpy(X=True)

    @property
    def Y(self):
        """Return a read-only numpy array of Y.

        If there is only one column, a one-dimensional array is returned. Otherwise 2D.
        The columns are in the same order as the columns in Table.domain.class_vars.

        Returns
        -------
        np.ndarray
        """
        result = self._to_numpy(Y=True)
        return result[:, 0] if result.shape[1] == 1 else result

    @property
    def metas(self):
        """Return a read-only 2D numpy array of metas.

        The columns are in the same order as the columns in Table.domain.metas.

        Returns
        -------
        np.ndarray
        """
        return self._to_numpy(meta=True)

    @property
    def weights(self):
        """Get the weights as a 1D numpy array."""
        if isinstance(self, SeriesBase):
            # at least 1D when using this in TableSeries, which instead returns a 0D ndarray directly
            result = np.atleast_1d(self[self._WEIGHTS_COLUMN])
        else:
            result = self[self._WEIGHTS_COLUMN].values
        # even if weights are set with an integer, we need to return a float
        return result.astype(float)

    def set_weights(self, weight):
        """Set the weights for the instances in this table.

        Parameters
        ----------
        weight : Number or str or Sequence or np.ndarray or pd.Series
            If a number, all weights are set to that value.
            If a string, weights are set to whatever the column with that name's values are,
            but only if those values are all numbers and are not NA/NaN.
            If a sequence of (non-NA/NaN) numbers, set those values as the sequence.

        Raises
        ------
        ValueError
            If weights are nan, the column does not exist or the sequence length is mismatched.
        TypeError
            If an unrecognized type is passed.
        """
        if isinstance(weight, Number):
            if np.isnan(weight):
                raise ValueError("Weights cannot be nan. ")
            new_weights = weight
        elif isinstance(weight, str):
            if weight not in self.columns:
                raise ValueError("{} is not a column.".format(repr(weight)))
            if self[weight].isnull().any() and np.issubdtype(self[weight].dtype, Number):
                raise ValueError("All values in the target column must be valid numbers.")
            new_weights = self[weight].fillna(value=self[weight].median())
        elif isinstance(weight, Sequence):
            if len(weight) != len(self):
                raise ValueError("The sequence has length {}, expected length {}.".format(len(weight), len(self)))
            new_weights = weight
        elif isinstance(weight, np.ndarray):  # np.ndarray is not a Sequence
            # allow row or column vectors
            if weight.ndim > 1 and not (weight.ndim == 2 and weight.shape[1] == 1):
                raise ValueError("Dimension mismatch.")
            new_weights = np.ravel(weight)
            if len(weight) != len(self):
                raise ValueError("There are {} weights, expected {}.".format(len(weight), len(self)))
        elif isinstance(weight, pd.Series):  # not only SeriesBase
            # drop everything but the values to uncomplicate things
            if weight.isnull().any():
                raise ValueError("Weights cannot be nan. ")
            new_weights = weight.values
        elif isinstance(weight, pd.Categorical):
            new_weights = list(weight)
        else:
            raise TypeError("Expected one of [Number, str, Sequence, SeriesBase].")

        # final check that we're actually adding numbers as weights
        if not (isinstance(new_weights, Number) or all(isinstance(w, Number) for w in new_weights)):
            raise ValueError("All weight values must be numbers.")
        self[self._WEIGHTS_COLUMN] = new_weights

    @property
    def has_weights(self):
        """Check if this Table has any weights.

        Weight presence is determined by all weights not being identical.

        Returns
        -------
        bool
            True if the weights are not all identical, False otherwise.
        """
        return len(self) and len(self[self._WEIGHTS_COLUMN].unique()) != 1

    @classmethod
    def _new_id(cls, num=1, force_list=False):
        """Generate new application-wide globally-unique numbers.

        Parameters
        ----------
        num : int, optional, default 1
            The number of new IDs to generate.
        force_list : bool, optional, default False
            Whether to always force the result to be a list,
            even if only one number is generated.

        Returns
        -------
        int or list[int]
            New globally-unique identifiers.
            If generating a single number and not forcing into a list, output an integer.
            Otherwise output a list of integers of len(num).
        """
        with cls._next_instance_lock:
            out = np.arange(cls._next_instance_id, cls._next_instance_id + num)
            cls._next_instance_id += num
            return out[0] if num == 1 and not force_list else out

    def _transform_discrete_values_to_descriptors(self):
        """Transform discrete variables given in index form into descriptors.

        The conversion is performed in-place.
        """
        for var in chain(self.domain.variables, self.domain.metas):
            if isinstance(var, DiscreteVariable):
                # only transform the values if all of them appear to be integers and
                # could act as a variable value index
                # otherwise we're dealing with numeric discretes
                values = self[var.name].values
                try:
                    ints = values.astype(np.uint16)
                except ValueError:
                    continue
                if (ints == values).all() and ints.max() < len(var.values):
                    self[var.name] = np.asanyarray(var.values)[ints]

    def _transform_discrete_into_categorical(self):
        """Transform columns with discrete variables into pandas' categoricals.

        The operation is performed in-place.
        This must be done after replacing null values because those aren't values,
        and also after transforming discretes to descriptors.
        """
        for var in chain(self.domain.variables, self.domain.metas):
            if isinstance(var, DiscreteVariable):
                self[var.name] = pd.Categorical(self[var.name], categories=var.values, ordered=var.ordered)

    def _transform_timevariable_into_datetime(self):
        """Transform columns with TimeVariables into pandas' datetime columns (in-place)."""
        for var in chain(self.domain.variables, self.domain.metas):
            if isinstance(var, TimeVariable):
                self[var.name] = var.column_to_datetime(self[var.name])

    def save(self, filename):
        """Save a Table to a file.

        Parameters
        ----------
        filename : str
            The destination file path, absolute or relative.
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
        if isinstance(item, (Sequence, pd.Index, np.ndarray)) and not isinstance(item, str) \
                and all(isinstance(i, str) for i in item) \
                and any(ic not in item for ic in self._INTERNAL_COLUMN_NAMES):
            item = list(item) + [ic for ic in self._INTERNAL_COLUMN_NAMES if ic not in item]
        return super().__getitem__(item)

    def __setitem__(self, key, value):
        # if the table has an empty index and we're inserting a the first row,
        # the index would be created by pandas automatically.
        # we want to maintain unique indices, so we override the index manually.
        # we also need to set default weights, lest they be NA
        new_index_and_weights = len(self.index) == 0

        # SOMEWHAT PANDAS CONTRACT BREAKAGE:
        # the pandas default behaviour when adding a new column as a Series is that
        # the indexes are inner-joined: only the elements at the indices that exist
        # in the current table are actually set, other elements are NA
        # for easier handling of Orange behaviour, we effectively ignore the index
        # on the series if the series is of the same length (like when adding a new column)
        # to just merge the column into the table
        if isinstance(value, pd.Series) and len(value) == len(self) and not new_index_and_weights:
            value.index = self.index
        super().__setitem__(key, value)

        if new_index_and_weights:
            new_id = self._new_id(len(self), force_list=True)
            self.index = new_id
            # super call because we'd otherwise recurse back into this
            super().__setitem__(self._WEIGHTS_COLUMN, 1)

    def __iter__(self):
        """Iterate over the rows of this TableBase as SeriesBase, breaking the pandas contract.

        Returns
        -------
        generator
            A generator of rows as SeriesBase objects.

        Examples
        --------
            >>> iris = Table('iris')
            >>> for row in iris.iloc[[0]]:
            ...     print(row)
            sepal length            5.1
            sepal width             3.5
            petal length            1.4
            petal width             0.2
            iris            Iris-setosa
            __weights__               1
            Name: iris, dtype: object

        Notes
        -----
        This breaks the pandas contract! Pandas iterates over column names by default.
        However, this only breaks __str__ and __repr__, which are reimplemented anyway.
        """
        for _, row in self.iterrows():
            yield row

    def __str__(self):
        """Augment the pandas representation to provide a more Orange-friendly one.

        This means including domain information in the output.
        """
        # get only visible columns, also in the proper x-y-metas order
        attrs = list(filter_visible(self.domain.attributes))
        class_vars = list(filter_visible(self.domain.class_vars))
        metas = list(filter_visible(self.domain.metas))
        visible_cols = attrs + class_vars + metas
        roles = ["attribute"] * len(attrs) + ["class"] * len(class_vars) + ["meta"] * len(metas)

        index_tuples = [(col, str(type(self.domain[col]))[:-len("Variable")].lower(), role, str(self[col].dtype))
                        for col, role in zip(visible_cols, roles)]
        index_tuples += [(icn, "", "", str(self[icn].dtype)) for icn in self._INTERNAL_COLUMN_NAMES]
        display_index = pd.MultiIndex.from_tuples(index_tuples, names=("name", "type", "role", "dtype"))

        # overwriting columns for display_df also overwrites them for self
        # to solve this without copying, we just replace them with the correct ones in the end
        saved_columns = self.columns
        display_df = super()._constructor(data=self)  # need supertype to avoid recursing back
        display_df.columns = display_index  # overwrite the index in-place, no reindexing
        result = str(display_df)
        display_df.columns = saved_columns
        return result


    @classmethod
    def concatenate(cls, tables, axis=1, reindex=True, colstack=True, rowstack=False):
        """Concatenate tables by rows or columns.

        The resulting table will always retain the properties (name etc.) of the first table in the sequence.

        Parameters
        ----------
        tables : list[TableBase]
            A list of tables to concatenate.
        axis : int, optional, default 1
            The axis by which to concatenate.
            Axis 0 are rows, axis 1 are columns.
        reindex : bool, optional, default True
            Whether to generate a new index for the resulting Table.
            If reindex is False
             - when concatenating rows: some rows may have the same index
             - when concatenating columns: the index of the first table is preserved
        colstack : bool, optional, default True
            Whether to stack columns when concatenating by columns.
            No two columns may have the same name.
            If colstack is True, the number of rows must match on all tables.
            If colstack is False, perform an outer join instead.
        rowstack : bool, optional, default False
            Whether to stack rows when concatenating by rows.
            If rowstack is True, the number of columns must match on all tables.
            If rowstack is False, perform an outer join on the columns with NA values
            in places without data.
        Returns
        -------
        TableBase
            The concatenated table.
        """
        def unique_by_name_preserve_order(iterable):
            s = set()
            return [item for item in iterable if not (item.name in s or s.add(item.name))]

        def copy_rename_vars(vars, suffix):
            return [v.copy(new_name=v.name + suffix) for v in vars]

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
                new_domain = tables[0].domain
            else:
                result = pd.concat(tables, axis=0)
                # merges columns (nans for rows without those)
                # domain must contain the uniques of all variables
                new_domain = Domain(
                    unique_by_name_preserve_order(flatten(t.domain.attributes for t in tables)),
                    unique_by_name_preserve_order(flatten(t.domain.class_vars for t in tables)),
                    unique_by_name_preserve_order(flatten(t.domain.metas for t in tables))
                )
            new_index = cls._new_id(len(result))
            result.index = new_index
        elif axis == CONCAT_COLS:
            # check for same name
            columns = [v for v in flatten([list(t.columns) for t in tables]) if v not in cls._INTERNAL_COLUMN_NAMES]
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
        tmpdomain = result.domain
        result.__finalize__(tables[0])  # pd.concat does not do this by itself
        result.domain = tmpdomain  # we don't want to transfer the domain, specifically
        return cls.from_dataframe(new_domain, result)

    def merge(self, right, *args, suffixes=('_left', '_right'), **kwargs):
        """Merge two Tables. pd.DataFrame.merge wrapper.

        Handles internal columns and domain merging. Renames duplicates appropriately.

        Parameters
        ----------
        right : TableBase
            The other Table to merge.
        args : tuple
            Other pandas.DataFrame.merge arguments.
        suffixes : tuple
            Overrides the pandas.DataFrame.merge duplicate column suffixes with _left and _right.
        kwargs : dict
            Other pandas.DataFrame.merge keyword arguments.

        Returns
        -------
        TableBase
            A new, merged Table.

        See Also
        --------
        pd.DataFrame.merge
        """
        # let pandas do its thing
        result = super().merge(right, *args, suffixes=suffixes, **kwargs)

        # transfer attrs from self
        result.__finalize__(self)

        # fix multiple identical columns
        for icn in self._INTERNAL_COLUMN_NAMES:
            # duplicated cols are appended with _x and _y by default
            matching_columns = [c for c in result.columns if icn in c]
            first_col = result[matching_columns[0]]
            result.drop(matching_columns, axis=1, inplace=True)
            result[icn] = first_col

        # process a list of variables, appending suffix if not found in the
        # resulting columns (that means it was a dup)
        def suffix_dups(varlist, suffix):
            return (v if v in result.columns else v.copy(new_name=v.name + suffix) for v in varlist)

        # dedup because the target join valriable doesn't get renamed and there is
        # only one column, wehile without this, the domain would have two
        def dedup_inorder(varlist):
            s = set()
            return [x for x in varlist if not (x in s or s.add(x))]

        # merge domain
        new_domain = Domain(
            dedup_inorder(chain(suffix_dups(self.domain.attributes, suffixes[0]),
                                suffix_dups(right.domain.attributes, suffixes[1]))),
            dedup_inorder(chain(suffix_dups(self.domain.class_vars, suffixes[0]),
                                suffix_dups(right.domain.class_vars, suffixes[1]))),
            dedup_inorder(chain(suffix_dups(self.domain.metas, suffixes[0]),
                                suffix_dups(right.domain.metas, suffixes[1])))
        )
        result.domain = new_domain

        # always reindex
        result.index = self._new_id(len(result), force_list=True)

        return result

    def approx_len(self):
        """Return the approximate length of the table."""
        # TODO: remove, here solely for SQL compatibility, which would ideally be replaced with Spark
        # https://github.com/biolab/orange3/wiki/pandas-migration#future-todos-and-architectural-wishlist
        return len(self)

    def exact_len(self):
        """Return the exact length of the table."""
        # TODO: remove, here solely for SQL compatibility, which would ideally be replaced with Spark
        # https://github.com/biolab/orange3/wiki/pandas-migration#future-todos-and-architectural-wishlist
        return len(self)

    def has_missing(self):
        """Return `True` if there are any missing attribute or class values."""
        # manual access to columns because dumping to a numpy array (with self.X) is slower
        return self[self.domain.attributes].isnull().any().any() or self.has_missing_class()

    def has_missing_class(self):
        """Return `True` if there are any missing class values."""
        return self[self.domain.class_vars].isnull().any().any()

    def iterrows(self):
        """An override to return TableSeries instead of Series (pandas doesn't do that by default)."""
        # TODO: remove this in next pandas release, when it works by default (pydata/pandas#13978)
        # super here is the next item in the MRO, e.g. pd.DataFrame or pd.SparseDataFrame
        gen = super().iterrows()
        for item in gen:
            yield (item[0], self._constructor_sliced(item[1]))

    def __hash__(self):
        # TODO: inconsistent when dtype=object
        return hash(bytes(self._to_numpy(X=True, Y=True)))

    @deprecated('pandas-style column access: t[["colname1", "colname2"]]')
    def get_column_view(self, index):
        """Get a single column view of the table.

        Return a vector - as a view, not a copy - with a column of the table,
        and a bool flag telling whether this column is sparse.

        Parameters
        ----------
        index : int or str
            The index of the column or the column name.

        Returns
        -------
        (np.ndarray, bool)
            A tuple of the column values and a flag indicating the sparsity of the column.
        """
        if isinstance(index, str):
            col = self[index]
        else:
            col = self[self.columns[index]]
        return col.values, isinstance(col, pd.SparseSeries)

    def _compute_basic_stats(self, columns=None, include_metas=False):
        """Compute basic stats for each of the columns.

        Parameters
        ----------
        columns : list[str or Variable], optional, default None
            A list of columns to compute the stats for, None selects all.
        include_metas : bool, optional, default False
            Whether to include meta attributes in the statistics computations.

        Returns
        -------
        np.ndarray
            The resulting array has len(columns) rows, where each row
            contains (min, max, mean, var, #nans, #non-nans) in that order.

        Notes
        -----
        This is a legacy method and should be avoided and/or replaced where possible.
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
        """Compute the distribution of values for the given columns.

        Parameters
        ----------
        columns : list[str or Variable], optional, default None
            A list of columns to compute the stats for, None selects all.

        Returns
        -------
        list[(np.ndarray, int)]
            A list of distribution tuples. Type of distribution depends on the
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
        """Compute contingency matrices for one or more discrete or continuous variables
        against the specified discrete variable.

        Parameters
        ----------
        col_vars : list[str or Variable], optional, default None
            The variables whose values will correspond to columns in the result items.
            If None, selects all attributes from the domain.
        row_var : str or DiscreteVariable, optional, default None
            The variable whose values will correspond to rows in the result.
            Limited to discrete variables.

        Returns
        -------
        (list[(np.ndarray or (list[Number], np.ndarray), np.ndarray)], int)
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
                pivot = pd.pivot_table(self, values=self._WEIGHTS_COLUMN, index=[row_var],
                                       columns=[var], aggfunc=np.sum).fillna(0)
                unknowns = unknown_grouper[var].agg(lambda x: x.isnull().sum())
                if var.is_discrete:
                    contingencies.append((pivot.values, unknowns.values))
                else:
                    contingencies.append(((pivot.columns.values, pivot.values), unknowns.values))
        return contingencies, self[row_var].isnull().sum()

    @property
    def density(self):
        """Return the density of the current table."""
        raise NotImplementedError

    @property
    def is_sparse(self):
        """Return True if the current table is sparse."""
        raise NotImplementedError

    @property
    def is_dense(self):
        """Return True if the current table is dense."""
        return not self.is_sparse


class SeriesBase:
    """
    A common superclass for Series (as in pd.Series or pd.SparseSeries) objects.
    Transfers Table x/y/metas/weights functionality to the Series.
    """
    _WEIGHTS_COLUMN = TableBase._WEIGHTS_COLUMN
    _metadata = ['domain']

    # use the same functions as in Table for this
    # WARNING: depends on TableSeries having a domain, which is ensured
    # in Table._constructor_sliced
    _to_numpy = TableBase._to_numpy
    X = TableBase.X
    Y = TableBase.Y
    metas = TableBase.metas
    weights = w = TableBase.weights

# no PanelBase because pandas only has one (pd.Panel), SparsePanel was removed
# use TablePanel instead
