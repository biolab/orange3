import unittest

import numpy.testing
import dask.array as da

from Orange.data import Table, Domain
from Orange.data.dask import DaskTable
from Orange.tests import named_file


class TableTestCase(unittest.TestCase):

    def same_tables(self, table, dasktable):
        self.assertEqual(dasktable.domain, table.domain)
        self.assertIsInstance(dasktable.X, da.Array)
        self.assertIsInstance(dasktable.Y, da.Array)
        self.assertNotIsInstance(dasktable.metas, da.Array)  # metas are in numpy
        numpy.testing.assert_equal(dasktable.X, table.X)
        numpy.testing.assert_equal(dasktable.Y, table.Y)
        numpy.testing.assert_equal(dasktable.metas, table.metas)

    def test_save_table(self):
        zoo = Table('zoo')
        with named_file('', suffix='.hdf5') as fn:
            DaskTable.save(zoo, fn)
            dzoo = DaskTable.from_file(fn)
            self.same_tables(zoo, dzoo)

    def test_save_dasktable(self):
        zoo = Table('zoo')
        with named_file('', suffix='.hdf5') as fn:
            DaskTable.save(zoo, fn)
            dzoo = DaskTable.from_file(fn)
            with named_file('', suffix='.hdf5') as fn2:
                dzoo.save(fn2)
                dzoo2 = DaskTable.from_file(fn)
                self.same_tables(zoo, dzoo2)

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
