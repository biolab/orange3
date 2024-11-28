import os
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime
from functools import wraps

import numpy as np
import numpy.testing
import dask.array as da

from Orange.data import Table, Domain, ContinuousVariable
from Orange.data.dask import DaskTable, dask_stats, DaskRowInstance
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


class _TempDaskTableHandler:

    def __init__(self, table):
        # pylint: disable=consider-using-with
        if isinstance(table, str):
            table = Table(table)
        file = tempfile.NamedTemporaryFile("wt", delete=False)
        self.fn = fn = file.name
        file.close()
        DaskTable.save(table, fn)
        self.dasktable = DaskTable.from_file(fn)
        self.dasktable.name = table.name
        self.dasktable._keep_alive_until_dasktable_is_needed = self

    def __del__(self):
        self.dasktable.close()
        os.remove(self.fn)


def temp_dasktable(table):
    """ Return a DaskTable created from a Table through a temporary file.
    The temporary file is deleted after garbage collection. """
    table = _TempDaskTableHandler(table)
    return table.dasktable


def with_dasktable(test_case):
    # type: (Callable) -> Callable
    """Run a single test case on both Orange Tables and DaskTables.

    Examples
    --------
    >>> @with_dasktable
    ... def test_something(self, prepare_table):
    ...     data: Table  # The table you want to test on
    ...     data = prepare_table(data)  # This converts the table to DaskTable

    """

    @wraps(test_case)
    def _wrapper(self):
        # Make sure to call setUp and tearDown methods in between test runs so
        # any widget state doesn't interfere between tests
        test_case(self, lambda table: table)
        self.tearDown()
        self.setUp()
        test_case(self, temp_dasktable)

    return _wrapper


class TableTestCase(unittest.TestCase):

    def same_tables(self, table, dasktable):
        self.assertEqual(dasktable.domain, table.domain)
        self.assertIsInstance(dasktable.X, da.Array)
        self.assertIsInstance(dasktable.Y, da.Array)
        self.assertNotIsInstance(dasktable.metas, da.Array)  # metas are in numpy
        numpy.testing.assert_equal(dasktable.X, table.X)
        numpy.testing.assert_equal(dasktable.Y, table.Y)
        numpy.testing.assert_equal(dasktable.metas, table.metas)

    def test_zero_width_dask_arrays(self):
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

    def test_zero_len_dask_arrays(self):
        atts = [ContinuousVariable("x"), ContinuousVariable("y")]
        empty = Table.from_numpy(Domain([], atts),
                                 X=np.ones((0, 0)),
                                 Y=np.ones((0, 2)))

        with open_as_dask(empty) as data:
            self.assertIsInstance(data, DaskTable)
            self.assertIsInstance(data.X, da.Array)
            self.assertIsInstance(data._Y, da.Array)
            self.assertEqual(data.X.shape, (0, 0))
            self.assertEqual(data._Y.shape, (0, 2))

        empty = Table.from_numpy(Domain(atts, []),
                                 X=np.ones((0, 2)),
                                 Y=np.ones((0, 0)))

        with open_as_dask(empty) as data:
            self.assertIsInstance(data, DaskTable)
            self.assertIsInstance(data.X, da.Array)
            self.assertIsInstance(data._Y, da.Array)
            self.assertEqual(data.X.shape, (0, 2))
            self.assertEqual(data._Y.shape, (0, 0))

    def test_compute(self):
        zoo = Table('zoo')
        with named_file('', suffix='.hdf5') as fn:
            DaskTable.save(zoo, fn)
            dzoo = DaskTable.from_file(fn)
            self.same_tables(dzoo.compute(), dzoo)
            dzoo.close()

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

    def test_remind_us_of_instance_warning(self):
        now = datetime.now()
        if (now.year, now.month) >= (2025, 1):
            iris = temp_dasktable("iris")
            with self.assertWarns(expected_warning=UserWarning):
                iris[1]
                self.fail("This warning is used as a development aid. This is a friendly "
                          "reminder that we should think about changing the warning into "
                          "an OrangeDeprecationWarning (and explicitly handle it, where "
                          "needed).")

    def test_instance(self):
        iris = temp_dasktable("iris")
        with self.assertWarns(expected_warning=UserWarning):
            instance = iris[1]
        self.assertIsInstance(instance, DaskRowInstance)

    def test_str(self):
        iris = temp_dasktable("iris")
        with self.assertWarns(expected_warning=UserWarning):
            self.assertEqual("[5.1, 3.5, 1.4, 0.2 | Iris-setosa]", str(iris[0]))
        with self.assertWarns(expected_warning=UserWarning):
            table_str = str(iris)
        lines = table_str.split('\n')
        self.assertEqual(150, len(lines))
        self.assertEqual("[[5.1, 3.5, 1.4, 0.2 | Iris-setosa],", lines[0])
        self.assertEqual(" [5.9, 3.0, 5.1, 1.8 | Iris-virginica]]", lines[-1])

    def test_table_len(self):
        iris = temp_dasktable("iris")
        iris = iris[iris.X[:, 0] > 6]
        self.assertTrue(np.isnan(iris.X.shape[0]))
        self.assertEqual(61, len(iris))
