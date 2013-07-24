import unittest
from mock import MagicMock
from psycopg2 import OperationalError
from Orange.data.sql import table as sql_table
from Orange import data
from Orange.tests.sql.base import PostgresTest


class SqlTableMockedTests(unittest.TestCase):
    def setUp(self):
        self.backend = self._mock_iris_backend()
        self.uri = "sql://localhost/test/iris"

    def test_parses_server_in_uri_format(self):
        table = sql_table.SqlTable("sql://user:password@server/database/table",
                                   backend=self.backend)

        self.backend.connect.assert_called_once_with(
            hostname='server',
            username='user',
            password='password',
            database='database',
            table='table',
        )
        self.assertEqual(table.host, 'server')
        self.assertEqual(table.database, 'database')
        self.assertEqual(table.table_name, 'table')

    def test_raises_value_error_on_invalid_scheme(self):
        with self.assertRaises(OperationalError):
            sql_table.SqlTable("http://server/database/table")

    def test_can_construct_attributes(self):
        table = sql_table.SqlTable(self.uri, backend=self.backend)

        attributes, metas = table._create_attributes()

        self.assertEqual(len(attributes), 5)
        for attr in attributes[:4]:
            self.assertIsInstance(attr, data.ContinuousVariable)
        for attr in attributes[4:]:
            self.assertIsInstance(attr, data.DiscreteVariable)

    def test_constructs_correct_domain(self):
        table = sql_table.SqlTable(self.uri, backend=self.backend)

        self.assertEqual(len(table.domain), 5)
        for attr in table.domain[:4]:
            self.assertIsInstance(attr, data.ContinuousVariable)
        attr = table.domain[4]
        self.assertIsInstance(attr, data.DiscreteVariable)
        self.assertSequenceEqual(
            attr.values,
            ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        )

    def test_sets_nrows(self):
        table = sql_table.SqlTable(self.uri, backend=self.backend)

        self.assertEqual(table.nrows, 150)

    def test_responds_to_len(self):
        table = sql_table.SqlTable(self.uri, backend=self.backend)

        nrows = len(table)

        self.assertEqual(nrows, 150)

    def _mock_iris_backend(self):
        return MagicMock(
            connect=MagicMock(),
            table_info=MagicMock(
                fields=(('sepal_length', 'double precision', ()),
                        ('sepal_width', 'double precision', ()),
                        ('petal_length', 'double precision', ()),
                        ('petal_width', 'double precision', ()),
                        ('iris', 'character varying',
                         ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'))),
                nrows=150,
            ),
        )


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
