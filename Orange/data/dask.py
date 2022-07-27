import contextlib
import pickle

import h5py
import dask
import dask.array as da
import numpy as np
import pandas

from Orange.data import Table


class DaskTable(Table):

    _array_interface = da

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
        self = cls()

        self.__h5file = f = h5py.File(filename, "r")

        def read_format_da(name):
            # dask's automatic chunking has problems with 0-dimension arrays
            if name in f and 0 not in f[name].shape:
                return da.from_array(f[name])
            return None

        self._X = read_format_da("X")
        self._Y = read_format_da("Y")

        # metas are in memory
        if "metas" in f:
            self._metas = pickle.loads(np.array(f['metas']).tobytes())
        else:
            self._metas = None

        size = None
        # get size from X, Y, or metas
        for el in ("_X", "_Y", "_metas"):
            array = getattr(self, el)
            if array is not None:
                size = len(array)
                break

        if self._X is None:
            self._X = da.zeros((size, 0), chunks=(size, 0))

        if self._Y is None:
            self._Y = da.zeros((size, 0), chunks=(size, 0))

        if self._metas is None:
            self._metas = np.zeros((size, 0))

        self._W = np.ones((size, 0))  # weights are unsupported

        self.domain = pickle.loads(np.array(f['domain']).tobytes())

        cls._init_ids(self)

        return self

    def close(self):
        self.__h5file.close()

    def set_weights(self, weight=1):
        raise NotImplementedError()

    def save(self, filename):
        # the following code works with both Table and DaskTable
        with h5py.File(filename, 'w') as f:
            f.create_dataset("domain", data=np.void(pickle.dumps(self.domain)))
            f.create_dataset("metas", data=np.void(pickle.dumps(self.metas)))

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

    def has_missing_attribute(self):
        raise NotImplementedError()

    def checksum(self, include_metas=True):
        raise NotImplementedError()

    def _filter_values(self, filter):
        selection = self._values_filter_to_indicator(filter)
        return self[selection.compute()]

    def compute(self) -> Table:
        X, Y = dask.compute(self.X, self.Y)
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
