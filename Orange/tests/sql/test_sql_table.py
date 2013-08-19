import unittest
from mock import MagicMock
from psycopg2 import OperationalError
from Orange.data.sql import table as sql_table
from Orange import data
from Orange.tests.sql.base import PostgresTest


class SqlTableUnitTests(unittest.TestCase):
    def test_parses_server_in_uri_format(self):
        table = sql_table.SqlTable.__new__(sql_table.SqlTable)
        parameters = table._parse_uri(
            "sql://user:password@host:7678/database/table")

        self.assertIn("host", parameters)
        self.assertEqual(parameters["host"], "host")
        self.assertIn("user", parameters)
        self.assertEqual(parameters["user"], "user")
        self.assertIn("password", parameters)
        self.assertEqual(parameters["password"], "password")
        self.assertIn("port", parameters)
        self.assertEqual(parameters["port"], 7678)
        self.assertIn("database", parameters)
        self.assertEqual(parameters["database"], "database")
        self.assertIn("table", parameters)
        self.assertEqual(parameters["table"], "table")


class SqlTableTests(PostgresTest):
    def test_reads_attributes_from_database(self):
        table = sql_table.SqlTable("sql://localhost/test/iris")

        # Continuous
        sepal_length = table[0][0]
        self.assertAlmostEqual(float(sepal_length), 5.1)
        self.assertEqual(str(sepal_length), '5.100')

        # Discrete
        iris = table[0][4]
        self.assertAlmostEqual(float(iris), 0)
        self.assertEqual(str(iris), 'Iris-setosa')

    def test_can_connect_to_database(self):
        table = sql_table.SqlTable('/test/iris')
        self.assertEqual(table.table_name, 'iris')
        self.assertEqual(
            [attr.name for attr in table.domain],
            ['sepal length', 'sepal width', 'petal length', 'petal width',
             'iris']
        )
        self.assertSequenceEqual(
            table.domain['iris'].values,
            ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

    def test_query_all(self):
        table = sql_table.SqlTable('/test/iris')
        results = list(table)

        self.assertEqual(len(results), 150)

    def test_query_subset_of_attributes(self):
        table = sql_table.SqlTable('/test/iris')
        attributes = [
            self._mock_attribute("sepal length"),
            self._mock_attribute("sepal width"),
            self._mock_attribute("double width", '2 * "sepal width"')
        ]
        results = list(table._query(
            attributes
        ))

        self.assertSequenceEqual(
            results[:5],
            [(5.1, 3.5, 7.0),
             (4.9, 3.0, 6.0),
             (4.7, 3.2, 6.4),
             (4.6, 3.1, 6.2),
             (5.0, 3.6, 7.2)]
        )

    def test_query_subset_of_rows(self):
        table = sql_table.SqlTable('/test/iris')
        all_results = list(table._query())

        results = list(table._query(rows=range(10)))
        self.assertEqual(len(results), 10)
        self.assertSequenceEqual(results, all_results[:10])

        results = list(table._query(rows=range(10)))
        self.assertEqual(len(results), 10)
        self.assertSequenceEqual(results, all_results[:10])

        results = list(table._query(rows=slice(None, 10)))
        self.assertEqual(len(results), 10)
        self.assertSequenceEqual(results, all_results[:10])

        results = list(table._query(rows=slice(10, None)))
        self.assertEqual(len(results), 140)
        self.assertSequenceEqual(results, all_results[10:])

    def _mock_attribute(self, attr_name, formula=None):
        if formula is None:
            formula = '"%s"' % attr_name
        class attr:
            name = attr_name

            @staticmethod
            def to_sql():
                return formula
        return attr
