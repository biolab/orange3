import unittest
from mock import MagicMock
from psycopg2 import OperationalError
from Orange.data.sql import table as sql_table
from Orange import data
from Orange.tests.sql.base import PostgresTest


class SqlTableUnitTests(unittest.TestCase):
    def setUp(self):
        self.table = sql_table.SqlTable.__new__(sql_table.SqlTable)

    def test_parses_connection_uri(self):
        parameters = self.table._parse_uri(
            "sql://user:password@host:7678/database/table")

        self.assertDictContainsSubset(dict(
            host="host",
            user="user",
            password="password",
            port=7678,
            database="database",
            table="table"
        ), parameters)

    def test_parse_minimal_connection_uri(self):
        parameters = self.table._parse_uri(
            "sql://host/database/table")

        self.assertDictContainsSubset(
            dict(host="host", database="database", table="table"),
            parameters
        )

    def test_parse_schema(self):
        parameters = self.table._parse_uri(
            "sql://host/database/table?schema=schema")

        self.assertDictContainsSubset(
            dict(database="database",
                 table="table",
                 host="host",
                 schema="schema"),
            parameters
        )

    def assertDictContainsSubset(self, subset, dictionary, msg=None):
        """Checks whether dictionary is a superset of subset.

        This method has been copied from unittest/case.py and undeprecated.
        """
        from unittest.case import safe_repr
        missing = []
        mismatched = []
        for key, value in subset.items():
            if key not in dictionary:
                missing.append(key)
            elif value != dictionary[key]:
                mismatched.append('%s, expected: %s, actual: %s' %
                                  (safe_repr(key), safe_repr(value),
                                   safe_repr(dictionary[key])))

        if not (missing or mismatched):
            return

        standardMsg = ''
        if missing:
            standardMsg = 'Missing: %s' % ','.join(safe_repr(m) for m in
                                                   missing)
        if mismatched:
            if standardMsg:
                standardMsg += '; '
            standardMsg += 'Mismatched values: %s' % ','.join(mismatched)

        self.fail(self._formatMessage(msg, standardMsg))


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
