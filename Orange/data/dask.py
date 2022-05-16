import pickle

import h5py
import dask.array as da
import numpy as np

from Orange.data import Table, Domain


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
            self.W = self.metas = np.ones((len(X), 0))

        # TODO ids need to be set

        self.X = da.from_array(X)
        self.Y = da.from_array(Y)
        self.metas = np.ones((len(self.X), 0))  # TODO

        self.domain = pickle.loads(np.array(f['domain']).tobytes())

        cls._init_ids(self)

        return self


def table_to_dask(table, filename):

    with h5py.File(filename, 'w') as f:
        f.create_dataset("X", data=table.X)
        f.create_dataset("Y", data=table.Y)
        domain = Domain(table.domain.attributes, table.domain.class_vars, None)
        f.create_dataset("domain", data=np.void(pickle.dumps(domain)))
        #f.create_dataset("metas", data=table.metas)  # TODO object
