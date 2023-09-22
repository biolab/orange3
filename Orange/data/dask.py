import contextlib
import pickle
import warnings
from numbers import Integral
from os import path
import json

import h5py
import dask.delayed
import dask.array as da
import numpy as np
import pandas

from Orange.data import Table, RowInstance, Domain, \
    ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable
from Orange.data.table import _FromTableConversion, _ArrayConversion
from Orange.statistics.util import contingency
from Orange.util import OrangeDeprecationWarning


class DaskRowInstance(RowInstance):

    def __init__(self, table, row_index):
        """
        Construct a data instance representing the given row of the table.
        """
        warnings.warn("Creating Instances from DaskTables is inefficient", stacklevel=2)
        super().__init__(table, row_index)
        if isinstance(self._x, dask.array.Array):
            self._x = self._x.compute()
        if isinstance(self._y, dask.array.Array):
            self._y = self._y.compute()


class _ArrayConversionDask(_ArrayConversion):

    def join_partial_results(self, parts):
        if self.is_dask:
            return dask.array.vstack(parts)
        return super().join_partial_results(parts)

    def join_columns(self, data):
        if self.is_dask:
            return dask.array.hstack(data)
        return super().join_columns(data)


class _FromTableConversionDask(_FromTableConversion):

    # set very high to make the compute graph smaller, because
    # for dask operations it does not matter how high it is
    max_rows_at_once = 5000*1000

    _array_conversion_class = _ArrayConversionDask

    def __init__(self, source, destination):
        super().__init__(source, destination)
        self.X.is_dask = True
        self.Y.is_dask = True
        self.metas.is_dask = False
        self.X.results_inplace = False
        self.Y.results_inplace = False


class DaskTable(Table):

    _array_interface = da
    _from_table_conversion_class = _FromTableConversionDask

    # dask requires file to remain open
    __h5file: h5py.File

    def __new__(cls, *args, **kwargs):
        if not args and not kwargs:
            return super().__new__(cls)
        elif isinstance(args[0], DaskTable):
            if len(args) > 1:
                raise TypeError("DaskTable(table: DaskTable) expects just one argument")
            return cls.from_table(args[0].domain, args[0])
        return cls.from_arrays(*args, **kwargs)

    @classmethod
    def from_arrays(cls, domain, X=None, Y=None, metas=None):
        self = cls()

        size = None
        # get size from X, Y, or metas
        for array in [X, Y, metas]:
            if array is not None:
                size = len(array)
                break

        assert size is not None

        if X is None:
            X = da.zeros((size, len(domain.attributes)), chunks=(size, -1))

        if Y is None:
            Y = da.zeros((size, len(domain.class_vars)), chunks=(size, -1))

        if metas is None:
            metas = np.zeros((size, 0))

        assert isinstance(X, da.Array)
        assert isinstance(Y, da.Array)
        assert isinstance(metas, np.ndarray)

        self.domain = domain
        self._X = X
        self._Y = Y
        self._metas = metas
        self._W = np.ones((len(X), 0))  # weights are unsupported
        self._init_ids(self)

        return self

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
        h5file = f = h5py.File(filename, "r")

        def read_domain(sub):
            subdomain = f[f'domain_{sub}'].asstr() if f'domain_{sub}' in f else []
            subdomain_args = f[f'domain_{sub}_args'].asstr() \
                if f'domain_{sub}_args' in f else ['{}'] * len(subdomain)
            for attr, args in zip(subdomain, subdomain_args):
                yield attr[0], attr[1], json.loads(args)

        def make_var(name, header, args):
            var_cls = [var for var in (ContinuousVariable,
                                       DiscreteVariable,
                                       StringVariable,
                                       TimeVariable) if header in var.TYPE_HEADERS][0]
            new_var = var_cls(name, **{key: val for key, val in args.items()
                                       if key != "attributes"})
            new_var.attributes = args.get("attributes", {})
            return new_var

        def read_format_da(name, as_str=False):
            # dask's automatic chunking has problems with 0-dimension arrays
            if name in f:
                chunks = 'auto' if 0 not in f[name].shape else f[name].shape
                if as_str:
                    return da.from_array(f[name].asstr()[:], chunks=f[name].shape)
                return da.from_array(f[name], chunks=chunks)
            return None

        if 'domain' in f and f['domain'].shape == tuple():
            domain = pickle.loads(np.array(f['domain']).tobytes())
            warnings.warn("Use the newer DaskTable format (perhaps just run save() again).",
                          OrangeDeprecationWarning, stacklevel=2)
        else:
            domain = Domain(*[[make_var(*args) for args in read_domain(subdomain)]
                            for subdomain in ['attributes', 'class_vars', 'metas']])

        X = read_format_da("X")
        Y = read_format_da("Y")

        if len(domain.metas) and 'metas' in f and not isinstance(f['metas'], h5py.Group):
            metas = pickle.loads(np.array(f['metas']).tobytes())
            warnings.warn("Use the newer DaskTable format (perhaps just run save() again).",
                          OrangeDeprecationWarning, stacklevel=2)
        else:
            if len(domain.metas) > 1:
                metas = np.hstack([read_format_da(f'metas/{i}',
                                                  isinstance(attr, StringVariable))
                                   for i, attr in enumerate(domain.metas)])
                # metas are in memory
                metas = metas.compute()
            elif len(domain.metas) == 1:
                metas = read_format_da('metas/0',
                                       isinstance(domain.metas[0], StringVariable)
                                       ).compute()
            else:
                metas = None

        self = DaskTable.from_arrays(domain, X, Y, metas)
        self.__h5file = h5file
        if isinstance(filename, str):
            self.name = path.splitext(path.split(filename)[-1])[0]

        return self

    def close(self):
        self.__h5file.close()

    def set_weights(self, weight=1):
        raise NotImplementedError()

    def __getitem__(self, key):
        if isinstance(key, Integral):
            return DaskRowInstance(self, key)
        else:
            return super().__getitem__(key)

    def save(self, filename):
        # the following code works with both Table and DaskTable
        def parse(attr):
            params = (attr.name, attr.TYPE_HEADERS[1], {"attributes": attr.attributes})
            if isinstance(attr, DiscreteVariable):
                params[2].update(values=attr.values)
            elif isinstance(attr, TimeVariable):
                params[2].update(have_date=attr.have_date,
                                 have_time=attr.have_time)
            elif isinstance(attr, ContinuousVariable):
                params[2].update(number_of_decimals=attr.number_of_decimals)
            return params

        with h5py.File(filename, 'w') as f:
            for subdomain in ['attributes', 'class_vars', 'metas']:
                parsed = [parse(feature) for feature in getattr(self.domain, subdomain)]
                domain = np.array([[name, header] for name, header, _ in parsed], 'S')
                domain_args = np.array([json.dumps(args) for *_, args in parsed], 'S')
                f.create_dataset(f'domain_{subdomain}', data=domain)
                f.create_dataset(f'domain_{subdomain}_args', data=domain_args)

        if isinstance(self.X, da.Array):
            da.to_hdf5(filename, '/X', self.X)
        else:
            with h5py.File(filename, 'r+') as f:
                f.create_dataset("X", data=self.X)

        if self.Y.size:
            if isinstance(self.Y, da.Array):
                da.to_hdf5(filename, '/Y', self.Y)
            else:
                with h5py.File(filename, 'r+') as f:
                    f.create_dataset("Y", data=self.Y)

        if self.metas.size:
            if isinstance(self.metas, da.Array):
                for i, attr in enumerate(self.domain.metas):
                    col_type = 'S' if isinstance(attr, StringVariable) else 'f'
                    col_data = self.metas[:, [i]].astype(col_type)
                    da.to_hdf5(filename, f'/metas/{i}', col_data)
            else:
                with h5py.File(filename, 'r+') as f:
                    for i, attr in enumerate(self.domain.metas):
                        col_type = 'S' if isinstance(attr, StringVariable) else 'f'
                        col_data = self.metas[:, [i]].astype(col_type)
                        f.create_dataset(f'metas/{i}', data=col_data)

    def has_missing_attribute(self):
        raise NotImplementedError()

    def checksum(self, include_metas=True):
        raise NotImplementedError()

    def _filter_values(self, filter):
        selection = self._values_filter_to_indicator(filter)
        return self[selection.compute()]

    def compute(self) -> Table:
        X, Y = da.compute(self.X, self.Y)
        table = Table.from_numpy(self.domain, X, Y,
                                 metas=self.metas, W=self.W,
                                 attributes=self.attributes, ids=self.ids)
        table.name = self.name
        return table

    def _compute_basic_stats(self, columns=None,
                             include_metas=False, compute_variance=False):
        rr = []
        stats = []
        if not columns:
            if self.domain.attributes:
                rr.append(dask_stats(self._X,
                                     compute_variance=compute_variance))
            if self.domain.class_vars:
                rr.append(dask_stats(self._Y,
                                     compute_variance=compute_variance))
            if include_metas and self.domain.metas:
                rr.append(dask_stats(self.metas,
                                     compute_variance=compute_variance))
            if len(rr):
                stats = da.concatenate(rr, axis=0)
        else:
            nattrs = len(self.domain.attributes)
            for column in columns:
                c = self.domain.index(column)
                if 0 <= c < nattrs:
                    S = dask_stats(self._X[:, [c]],
                                   compute_variance=compute_variance)
                elif c >= nattrs:
                    if self._Y.ndim == 1 and c == nattrs:
                        S = dask_stats(self._Y[:, None],
                                       compute_variance=compute_variance)
                    else:
                        S = dask_stats(self._Y[:, [c - nattrs]],
                                       compute_variance=compute_variance)
                else:
                    S = dask_stats(self._metas[:, [-1 - c]],
                                   compute_variance=compute_variance)
                stats.append(S)
            stats = da.concatenate(stats, axis=0)
        stats = stats.compute()
        return stats

    def _update_locks(self, *args, **kwargs):
        return

    def ensure_copy(self):
        self._X = self._X.copy()
        self._Y = self._Y.copy()
        self._metas = self._metas.copy()
        self._W = self._W.copy()

    def get_nan_frequency_attribute(self):
        if self.X.size == 0:
            return 0
        return np.isnan(self.X).sum().compute() / self.X.size

    def unlocked(self, *parts):
        # table locking is currently disabled
        return contextlib.nullcontext()

    def __len__(self):
        if not isinstance(self.X.shape[0], int):
            self.X.compute_chunk_sizes()
        return self.X.shape[0]

    def _filter_has_class(self, negate=False):
        if self._Y.ndim == 1:
            retain = np.isnan(self._Y)
        else:
            retain = np.any(np.isnan(self._Y), axis=1)
        if not negate:
            retain = np.logical_not(retain)
        return self.from_table_rows(self, np.asarray(retain))

    def _compute_contingency(self, col_vars=None, row_var=None):
        if row_var is None:
            row_var = self.domain.class_var
            if row_var is None:
                raise ValueError("No row variable")

        row_indi = self.domain.index(row_var)
        row_var = self.domain[row_indi]

        if not row_var.is_discrete:
            raise TypeError("Row variable must be discrete")

        if col_vars is None:
            col_indi = range(len(self.domain.variables))
        else:
            col_indi = [self.domain.index(var) for var in col_vars]
        col_vars = [self.domain[ind] for ind in col_indi]

        if any(not var.is_discrete for var in col_vars):
            raise NotImplementedError("Contingency can only be computed for categorical values.")

        @dask.delayed
        def delayed_contingency(*args, **kwargs):
            return contingency(*args, **kwargs)

        n_atts = self.X.shape[1]
        contingencies = [None] * len(col_vars)
        for arr, f_cond, f_ind in (
                (self.X, lambda i: 0 <= i < n_atts, lambda i: i),
                (self._Y, lambda i: i >= n_atts, lambda i: i - n_atts),
                (self.metas, lambda i: i < 0, lambda i: -1 - i)):

            for e, ind in enumerate(col_indi):
                if f_cond(ind):
                    col_i, arr_i, var = e, f_ind(col_indi[e]), col_vars[e]
                    col = arr if arr.ndim == 1 else arr[:, arr_i]
                    contingencies[col_i] = delayed_contingency(
                        col.astype(float),
                        self._get_column_view(row_indi),
                        max_X=len(var.values) - 1,
                        max_y=len(row_var.values) - 1,
                        weights=self.W if self.has_weights() else None)

        return da.compute(contingencies)[0]


def dask_stats(X, compute_variance=False):
    is_numeric = np.issubdtype(X.dtype, np.number)

    if X.size and is_numeric:
        nans = da.isnan(X).sum(axis=0).reshape(-1, 1)
        return da.concatenate((
            da.nanmin(X, axis=0).reshape(-1, 1),
            da.nanmax(X, axis=0).reshape(-1, 1),
            da.nanmean(X, axis=0).reshape(-1, 1),
            da.nanvar(X, axis=0).reshape(-1, 1) if compute_variance else \
                da.zeros(nans.shape),
            nans,
            X.shape[0] - nans), axis=1)
    else:
        # metas is currently a numpy table
        nans = (pandas.isnull(X).sum(axis=0) + (X == "").sum(axis=0)) \
            if X.size else np.zeros(X.shape[1])
        return np.column_stack((
            np.tile(np.inf, X.shape[1]),
            np.tile(-np.inf, X.shape[1]),
            np.zeros(X.shape[1]),
            np.zeros(X.shape[1]),
            nans,
            X.shape[0] - nans))


def table_to_dask(table, filename):

    with h5py.File(filename, 'w') as f:
        f.create_dataset("X", data=table.X)
        f.create_dataset("Y", data=table.Y)
        f.create_dataset("domain", data=np.void(pickle.dumps(table.domain)))
        f.create_dataset("metas", data=np.void(pickle.dumps(table.metas)))
