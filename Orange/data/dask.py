import pickle

import h5py
import dask.array as da
import numpy as np

from Orange.data import Table


class DaskTable(Table):

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

        f = h5py.File(filename, "r")
        X = f['X']
        Y = f['Y']
        if 'W' in f:
            self.W = da.from_array(f['W'])
        else:
            self.W = np.ones((len(X), 0))
        self.X = da.from_array(X)
        self.Y = da.from_array(Y)

        # TODO for now, metas are in memory
        if "metas" in f:
            self.metas = pickle.loads(np.array(f['metas']).tobytes())
        else:
            self.metas = np.ones((len(self.X), 0))

        self.domain = pickle.loads(np.array(f['domain']).tobytes())

        cls._init_ids(self)

        return self

    def has_missing_attribute(self):
        raise NotImplementedError()

    def checksum(self, include_metas=True):
        raise NotImplementedError()


def table_to_dask(table, filename):

    with h5py.File(filename, 'w') as f:
        f.create_dataset("X", data=table.X)
        f.create_dataset("Y", data=table.Y)
        f.create_dataset("domain", data=np.void(pickle.dumps(table.domain)))
        f.create_dataset("metas", data=np.void(pickle.dumps(table.metas)))
