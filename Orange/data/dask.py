import pickle

import h5py
import dask.array as da
import numpy as np

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

    def _update_locks(self, *args, **kwargs):
        return


def table_to_dask(table, filename):

    with h5py.File(filename, 'w') as f:
        f.create_dataset("X", data=table.X)
        f.create_dataset("Y", data=table.Y)
        f.create_dataset("domain", data=np.void(pickle.dumps(table.domain)))
        f.create_dataset("metas", data=np.void(pickle.dumps(table.metas)))
