import unittest
from contextlib import contextmanager

import numpy.testing
import dask.array as da

from Orange.data import Table, Domain
from Orange.data.dask import DaskTable, dask_stats
from Orange.statistics.basic_stats import DomainBasicStats
from Orange.tests import named_file


@contextmanager
def open_as_dask(table):
    if isinstance(table, str):
        table = Table(table)
    with named_file('') as fn:
        try:
            DaskTable.save(table, fn)
            dasktable = DaskTable.from_file(fn)
            yield dasktable
        finally:
            dasktable.close()


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

    def test_dask_stats(self):
        with open_as_dask("zoo") as data:
            stats = dask_stats(data.X)
            self.assertIsInstance(stats, da.Array)
            s = stats.compute()
            self.assertEqual(s.shape, (data.X.shape[1], 6))
            numpy.testing.assert_almost_equal(s[0],
                                              [0., 1., 0.4257426, 0., 0., 101.])
            stats = dask_stats(data.X, compute_variance=True)
            s = stats.compute()
            numpy.testing.assert_almost_equal(s[0],
                                              [0., 1., 0.4257426, 0.2444858, 0., 101.])

    def test_dask_stats_1d_matrix(self):
        with open_as_dask("zoo") as data:
            stats = dask_stats(data.Y)
            self.assertIsInstance(stats, da.Array)
            s = stats.compute()
            self.assertEqual(s.shape, (1, 6))
            numpy.testing.assert_almost_equal(s[0],
                                              [0., 6., 3.41584158, 0., 0., 101.])

    def test_domain_basic_stats(self):
        with open_as_dask("housing") as data:
            print(len(data.domain))
            stats = DomainBasicStats(data, compute_variance=True)
            f = stats[0]
            self.assertEqual(f.min, 0.00632)
            self.assertEqual(f.max, 88.9762)
            self.assertAlmostEqual(f.mean, 3.6135236)
            self.assertAlmostEqual(f.var, 73.8403597)
            c = stats[13]  # class value
            self.assertEqual(c.min, 5)
            self.assertEqual(c.max, 50)
            self.assertAlmostEqual(c.mean, 22.5328063)
            self.assertAlmostEqual(c.var, 84.4195562)
