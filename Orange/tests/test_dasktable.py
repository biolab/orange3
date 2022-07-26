import unittest
from contextlib import contextmanager

import numpy as np
import numpy.testing
import dask.array as da

from Orange.data import Table, Domain, ContinuousVariable
from Orange.data.dask import DaskTable
from Orange.tests import named_file


@contextmanager
def open_as_dask(table):
    if isinstance(table, str):
        table = Table(table)
    with named_file('') as fn:
        DaskTable.save(table, fn)
        dasktable = DaskTable.from_file(fn)
        yield dasktable


class TableTestCase(unittest.TestCase):

    def same_tables(self, table, dasktable):
        self.assertEqual(dasktable.domain, table.domain)
        self.assertIsInstance(dasktable.X, da.Array)
        self.assertIsInstance(dasktable.Y, da.Array)
        self.assertNotIsInstance(dasktable.metas, da.Array)  # metas are in numpy
        numpy.testing.assert_equal(dasktable.X, table.X)
        numpy.testing.assert_equal(dasktable.Y, table.Y)
        numpy.testing.assert_equal(dasktable.metas, table.metas)

    def test_zero_size_dask_arrays(self):
        empty = Table.from_numpy(Domain([], [ContinuousVariable("y")]),
                                X=np.ones((10**5, 0)),
                                Y=np.ones((10**5, 1)))

        with open_as_dask(empty) as data:
            self.assertIsInstance(data, DaskTable)
            self.assertIsInstance(data.X, da.Array)
            self.assertEqual(data.X.shape, (10**5, 0))
            self.assertEqual(data._Y.size, 10**5)

        empty = Table.from_numpy(Domain([ContinuousVariable("x")], []),
                                 X=np.ones((10**5, 1)),
                                 Y=np.ones((10**5, 0)))

        with open_as_dask(empty) as data:
            self.assertIsInstance(data, DaskTable)
            self.assertIsInstance(data.Y, da.Array)
            self.assertEqual(data.X.shape, (10**5, 1))
            self.assertEqual(data._Y.shape, (10**5, 0))

    def test_save_table(self):
        zoo = Table('zoo')
        with named_file('', suffix='.hdf5') as fn:
            DaskTable.save(zoo, fn)
            dzoo = DaskTable.from_file(fn)
            self.same_tables(zoo, dzoo)
            dzoo.close()

    def test_save_dasktable(self):
        zoo = Table('zoo')
        with named_file('', suffix='.hdf5') as fn:
            DaskTable.save(zoo, fn)
            dzoo = DaskTable.from_file(fn)
            with named_file('', suffix='.hdf5') as fn2:
                dzoo.save(fn2)
                dzoo2 = DaskTable.from_file(fn)
                self.same_tables(zoo, dzoo2)
                dzoo2.close()
            dzoo.close()

    def test_save_dasktable_empty(self):
        zoo = Table('zoo')
        zoo = zoo.transform(Domain(zoo.domain.attributes, None, zoo.domain.metas))
        with named_file('', suffix='.hdf5') as fn:
            DaskTable.save(zoo, fn)
            dzoo = DaskTable.from_file(fn)
            with named_file('', suffix='.hdf5') as fn2:
                dzoo.save(fn2)
                dzoo2 = DaskTable.from_file(fn)
                self.same_tables(zoo, dzoo2)
                dzoo2.close()
            dzoo.close()
