import warnings

from Orange.data import ContinuousVariable, Domain, StringVariable, DiscreteVariable
from Orange.data.table.base import _transferer

from .base import *
import pandas as pd
from pandas.sparse.array import BlockIndex


class Table(TableBase, pd.DataFrame):
    """A dense implementation of an Orange table."""

    @property
    def _constructor(self):
        """Proper pandas extension as per http://pandas.pydata.org/pandas-docs/stable/internals.html"""
        return Table

    @property
    def _constructor_sliced(self):
        """
        An ugly workaround for the fact that pandas doesn't transfer _metadata to Series objects.
        Where this property should return a constructor callable, we instead return a
        proxy function, which sets the necessary properties from _metadata using a closure
        to ensure thread-safety.

        This enables TableSeries to use .X/.Y/.metas because it has a Domain.
        """
        attrs = {k: getattr(self, k, None) for k in Table._metadata}
        return _transferer(TableSeries, attrs)

    @property
    def _constructor_expanddim(self):
        return TablePanel

    @property
    def density(self):
        """
        Compute the table density.
        Return the ratio of null values (pandas interpretation of null)
        """
        return 1 - self.isnull().sum().sum() / self.size

    @property
    def is_sparse(self):
        return False


class TableSeries(SeriesBase, pd.Series):
    @property
    def _constructor(self):
        return TableSeries

    @property
    def _constructor_expanddim(self):
        return Table


class TablePanel(pd.Panel):
    @property
    def _constructor(self):
        return TablePanel

    @property
    def _constructor_sliced(self):
        return Table


class SparseTable(TableBase, pd.SparseDataFrame):
    """A sparse implementation of an Orange table."""

    @property
    def _constructor(self):
        """Proper pandas extension as per http://pandas.pydata.org/pandas-docs/stable/internals.html"""
        return SparseTable

    @property
    def _constructor_sliced(self):
        """
        An ugly workaround for the fact that pandas doesn't transfer _metadata to Series objects.
        Where this property should return a constructor callable, we instead return a
        proxy function, which sets the necessary properties from _metadata using a closure
        to ensure thread-safety.

        This enables TableSeries to use .X/.Y/.metas because it has a Domain.
        """
        attrs = {k: getattr(self, k, None) for k in self._metadata}
        return _transferer(SparseTableSeries, attrs)

    @property
    def _constructor_expanddim(self):
        return TablePanel

    def __setitem__(self, key, value):
        # we don't want any special handling, use pandas directly
        return pd.SparseDataFrame.__setitem__(self, key, value)

    def _to_numpy(self, X=False, Y=False, meta=False, writable=False):
        """Generate a sparse matrix representation

        Unlike the default _to_numpy, this creates scipy.sparse matrices.
        Does not generate a dense matrix in memory.

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
        scipy.sparse.coo_matrix or np.ndarray
            The sparse matrix of the selected and transformed table data,
            or a dense ndarray if the data contains any non-numeric entries.

        """
        cols = []
        cols += self.domain.attributes if X else []
        cols += self.domain.class_vars if Y else []
        cols += self.domain.metas if meta else []
        n_rows = 1 if isinstance(self, SeriesBase) else len(self)

        # return empty if no columns: concatenation below needs some data
        # and in this case, there is none
        if not cols:
            return sp.coo_matrix((len(self), 0), dtype=np.float64)

        if any(v.is_string or v.is_discrete and
                any(not isinstance(val, Number) for val in v.values) for v in cols):
            # if there are any textual features, we must return a dense matrix,
            # as scipy.sparse only supports floats
            # just use the default implementation here instead of duplicating code
            return super()._to_numpy(X, Y, meta, writable)
        else:
            # the normal case, returning a sparse matrix
            # adapted from https://stackoverflow.com/a/37417084
            #  - does not handle dense columns (we have none)
            # in a nutshell, gets the coo_matrix building components directly
            # from each column of the SparseTable
            result_data = []
            result_row = []
            result_col = []
            for i, col in enumerate(cols):
                column_index = self[col.name].sp_index
                if isinstance(column_index, BlockIndex):
                    column_index = column_index.to_int_index()
                result_data.append(self[col.name].sp_values)
                result_row.append(column_index.indices)
                result_col.append(len(self[col.name].sp_values) * [i])
            return sp.coo_matrix((np.concatenate(result_data), (np.concatenate(result_row), np.concatenate(result_col))),
                                 (n_rows, len(cols)), dtype=np.float64)

    @property
    def Y(self):
        result = self._to_numpy(Y=True)
        # see _to_numpy for why there are two cases
        if sp.issparse(result):
            # subscripting sparse matrices doesn't work, so get a copy
            return result.getcol(0).tocoo() if result.shape[1] == 1 else result
        else:
            return result[:, 0] if result.shape[1] == 1 else result

    @classmethod
    def _coo_to_sparse_dataframe(cls, coo_matrix, column_index_start):
        """Convert a scipy.sparse.coo_matrix into a sparse dataframe.

        Indices start at 0.

        This constructs a single SparseTableSeries, which is then used to fill up
        the resulting SparseTable column-by-column. This is, counter-intuitively,
        a good way of doing things.

        We can't create a SparseDataFrame from a SparseSeries and then unstack
        the MultiIndex, because that gives us a dense DataFrame.

        We can't pass a list of SparseSeries rows to the SparseDataFrame constructor,
        as we would with a dense DataFrame, because that doesn't (yet?) work.

        Parameters
        ----------
        coo_matrix : scipy.sparse.coo_matrix
            The origin COOrdinate matrix.
        column_index_start : int
            Where to start counting the columns from.
            Useful for later concatenating multiple DataFrames.

        Returns
        -------
        pd.SparseDataFrame
            The resulting sparse data frame.
            Not a SparseTable yet because we don't have any Orange-specific data associated with it.
        """
        # convert into a multiindex sparse series
        # transposed because it's easier to get rows from ss
        # use a dense index so this works when a row is all-nan
        ss = pd.SparseSeries.from_coo(coo_matrix.T, dense_index=True)

        # create a new, completely empty sparse container
        # and fill its columns up sequentially
        # use a SparseDataFrame so we don't do any weights
        result = pd.SparseDataFrame(index=list(range(coo_matrix.shape[0])),
                                    columns=list(range(column_index_start, column_index_start + coo_matrix.shape[1])))
        for data_column_index, result_column_label in zip(range(coo_matrix.shape[1]), result.columns):
            if coo_matrix.shape[1] == 1:
                # an SS generated from a column vector is different
                col = ss
            else:
                # use only the first level index (we have two)
                col = ss.loc[data_column_index, :]
            # rewrite index from tuples of (selected_level, col) into just (col)
            # because selected_level is the same (we've selected it)
            col.index = [i[1] for i in col.index]
            result[result_column_label] = col
        return result

    @classmethod
    def _from_sparse_numpy(cls, domain, X, Y=None, metas=None, weights=None):
        """Construct a SparseTable from scipy.sparse matrices.

        This accepts X/Y/metas-like matrices (as in no strings) because
        scipy.sparse doesn't support anything other than numbers.

        If domain is None, all columns are assumed to be continuous.

        Parameters
        ----------
        domain : Domain
            If None, the domain is inferred from the data. Otherwise, specifies
            the column assignment to the new SparseTable.
        X : scipy.sparse.spmatrix or np.ndarray
            The X component of the data (or undetermined, depending on the domain).
        Y : scipy.sparse.spmatrix or np.ndarray, optional, default None
            The Y component of the data.
        metas : scipy.sparse.spmatrix or np.ndarray, optional, default None
            The meta attributes of the data.
        weights : scipy.sparse.spmatrix or np.ndarray, optional, default None
            The meta attributes of the data.

        Returns
        -------
        SparseTable
            A new SparseTable consturcted from the given data.
        """
        if domain is None:
            # legendary inference: everything is continuous! :D
            # TODO: much better inference, maybe just reuse existing
            domain = Domain(
                [ContinuousVariable("Feature " + str(i)) for i in list(range(X.shape[1] if X is not None else 0))],
                [ContinuousVariable("Target " + str(i)) for i in list(range(Y.shape[1] if Y is not None else 0))],
                [ContinuousVariable("Meta " + str(i)) for i in list(range(metas.shape[1] if metas is not None else 0))]
            )

        # convert everything to coo because that's what pandas works with currently (0.18.1)
        # converting csc and csr to coo has linear complexity (per the scipy docs)
        def _any_to_coo(mat):
            if mat is None or sp.isspmatrix_coo(mat):
                return mat
            elif mat.size == 0:
                # we don't use this either way, might as well skip it altogether
                return None
            elif not sp.issparse(mat):
                # the result of this is always two-dimensional,
                # but we still need to ensure it's transposed correctly
                return mat if len(mat.shape) == 2 else np.array([mat]).T
            else:
                return mat.tocoo()
        X = _any_to_coo(X)
        Y = _any_to_coo(Y)
        metas = _any_to_coo(metas)
        weights = _any_to_coo(weights)

        columns = [v.name for v in domain.attributes + domain.class_vars + domain.metas or []]
        partial_sdfs = []
        col_idx_start = 0
        for role_array in (X, Y, metas):
            if role_array is None:
                continue
            # unstack to convert the 2-level index (one for each coordinate dimension)
            # into 2 indexes - rows and columns
            if sp.issparse(role_array):
                partial_sdfs.append(cls._coo_to_sparse_dataframe(role_array, col_idx_start))
            else:
                partial_sdfs.append(pd.SparseDataFrame(role_array))
            col_idx_start += role_array.shape[1]
        # instruct pandas not to copy unnecessarily
        # and coerce SparseDataFrame into SparseTable
        result = cls(data=pd.concat(partial_sdfs, axis=1, copy=False))
        # rename the columns from column indices to proper names
        # and the rows into globally unique labels
        result.columns = columns + cls._INTERNAL_COLUMN_NAMES
        result.index = cls._new_id(len(result), force_list=True)
        result.domain = domain
        result.set_weights(weights or 1)  # weights can be None
        return result

    def _compute_distributions(self, columns=None):
        # this needs reimplementing because of possible columns without values
        if columns is None:
            columns = self.domain.attributes + self.domain.class_vars + self.domain.metas
        distributions = []
        for col in columns:
            var = self.domain[col]
            if var.is_discrete:
                # so we correctly process columns where a value doesn't appear, we need to
                # process each value separately
                vals = var.values
            else:
                # if we're using the thing above, might as well consolidate
                # the behaviour and not use groupby
                vals = self[col].unique()
            weighed_counts = pd.Series({v: self[self[col] == v][self._WEIGHTS_COLUMN].sum() for v in vals})
            unknowns = self[col].isnull().sum()
            if var.is_discrete:
                if var.ordered:
                    distributions.append((np.array([weighed_counts.loc[val] for val in var.values]), unknowns))
                else:
                    distributions.append((weighed_counts.values, unknowns))
            else:
                # explicitly return a 2D if the column is all-zeros
                distributions.append((np.array(sorted(weighed_counts.iteritems())).T
                                      if len(weighed_counts) != 0 else np.empty((2, 0)), unknowns))
        return distributions

    def _compute_contingency(self, col_vars=None, row_var=None):
        # pivot table doesn't work properly for sparse (who would have thought),
        # so reimplement in a really inefficient manner
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

        def _get_vals(var):
            if var.is_discrete:
                return var.values
            else:
                return self[var].unique()

        contingencies = []
        for var in col_vars:
            if var is row_var:
                dist_unks = self._compute_distributions(columns=[row_var])
                contingencies.append(dist_unks[0])
            else:
                # beware the snail
                col_vals = _get_vals(var)
                row_vals = _get_vals(row_var)
                cont = np.zeros((len(row_vals), len(col_vals)))
                unks = np.zeros(len(row_vals))
                for i, rv in enumerate(row_vals):
                    rvfilter = self[self[row_var] == rv]
                    for j, cv in enumerate(col_vals):
                        cont[i, j] = rvfilter[self[var] == cv][self._WEIGHTS_COLUMN].sum()
                    unks[i] = rvfilter[var].isnull().sum()
                if var.is_discrete:
                    contingencies.append((cont, unks))
                else:
                    contingencies.append(((col_vals, cont), unks))
        return contingencies, self[row_var].isnull().sum()

    @property
    def density(self):
        """Return the density as reported by pd.SparseDataFrame.density"""
        # density is a property
        return pd.SparseDataFrame.density.fget(self)

    @property
    def is_sparse(self):
        return True


class SparseTableSeries(SeriesBase, pd.SparseSeries):
    @property
    def _constructor(self):
        return SparseTableSeries

    @property
    def _constructor_expanddim(self):
        return SparseTable
