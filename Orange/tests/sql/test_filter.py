from Orange.data.sql.table import SqlTable
from Orange.data.sql import filter

from Orange.tests.sql.base import PostgresTest


class FilterTests(PostgresTest):
    def test_something(self):
        iris = SqlTable(self.iris_uri)
        print(iris[0])

