"""Test for miscellaneous sql queries in widgets

Please note that such use is deprecated.
"""
import unittest

from Orange.data.sql.table import SqlTable
from Orange.preprocess import Discretize
from Orange.preprocess.discretize import EqualFreq
from Orange.tests.sql.base import PostgresTest
try:
    from Orange.widgets.visualize.owmosaic import get_conditional_distribution
    from Orange.widgets.visualize.utils.lac import \
        create_sql_contingency, get_bin_centers
except ImportError:
    no_widgets = True
else:
    no_widgets = False


class MiscSqlTests(PostgresTest):
    def test_discretization(self):
        iris = SqlTable(self.conn, self.iris, inspect_values=True)
        sepal_length = iris.domain["sepal length"]
        EqualFreq(n=4)(iris, sepal_length)

    @unittest.skipIf(no_widgets, "Cannot import widgets")
    def test_get_conditional_distribution(self):
        iris = SqlTable(self.conn, self.iris, inspect_values=True)
        sepal_length = iris.domain["sepal length"]
        get_conditional_distribution(iris, [sepal_length])
        get_conditional_distribution(iris, list(iris.domain.variables))

    @unittest.skipIf(no_widgets, "Cannot import widgets")
    def test_create_sql_contingency(self):
        iris = SqlTable(self.conn, self.iris, inspect_values=True)
        d_iris = Discretize()(iris)
        create_sql_contingency(d_iris, [0, 1], get_bin_centers(d_iris))
