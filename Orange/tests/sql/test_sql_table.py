import unittest

import numpy as np

from Orange.data.sql import table as sql_table
from Orange.data import filter, ContinuousVariable, DiscreteVariable, \
    StringVariable, Table, Domain
from Orange.data.sql.parser import SqlParser
from Orange.tests.sql.base import PostgresTest, get_dburi, has_psycopg2


@unittest.skipIf(not has_psycopg2, "Psycopg2 is required for sql tests.")
class SqlTableUnitTests(unittest.TestCase):
    def setUp(self):
        self.table = sql_table.SqlTable.__new__(sql_table.SqlTable)

    def test_parses_connection_uri(self):
        parameters = self.table.parse_uri(
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
        parameters = self.table.parse_uri(
            "sql://host/database/table")

        self.assertDictContainsSubset(
            dict(host="host", database="database", table="table"),
            parameters
        )

    def test_parse_schema(self):
        parameters = self.table.parse_uri(
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


@unittest.skipIf(not has_psycopg2, "Psycopg2 is required for sql tests.")
class SqlTableTests(PostgresTest):
    def test_constructs_correct_attributes(self):
        data = list(zip(self.float_variable(21),
                        self.discrete_variable(21),
                        self.string_variable(21)))
        with self.sql_table_from_data(data) as table:
            self.assertEqual(len(table.domain), 2)
            self.assertEqual(len(table.domain.metas), 1)

            float_attr, discrete_attr = table.domain
            string_attr, = table.domain.metas

            self.assertIsInstance(float_attr, ContinuousVariable)
            self.assertEqual(float_attr.name, "col0")
            self.assertEqual(float_attr.to_sql(), '"col0"')

            self.assertIsInstance(discrete_attr, DiscreteVariable)
            self.assertEqual(discrete_attr.name, "col1")
            self.assertEqual(discrete_attr.to_sql(), '"col1"')
            self.assertEqual(discrete_attr.values, ['f', 'm'])

            self.assertIsInstance(string_attr, StringVariable)
            self.assertEqual(string_attr.name, "col2")
            self.assertEqual(string_attr.to_sql(), '"col2"')

    def test_len(self):
        with self.sql_table_from_data(zip(self.float_variable(26))) as table:
            self.assertEqual(len(table), 26)

        with self.sql_table_from_data(zip(self.float_variable(0))) as table:
            self.assertEqual(len(table), 0)

    def test_len_with_filter(self):
        with self.sql_table_from_data(
                zip(self.discrete_variable(26))) as table:
            self.assertEqual(len(table), 26)

            filtered_table = filter.SameValue(table.domain[0], 'm')(table)
            self.assertEqual(len(filtered_table), 13)

            table.domain[0].values.append('x')
            filtered_table = filter.SameValue(table.domain[0], 'x')(table)
            self.assertEqual(len(filtered_table), 0)

    def test_query_all(self):
        table = sql_table.SqlTable(self.iris_uri)
        results = list(table)

        self.assertEqual(len(results), 150)

    def test_query_subset_of_attributes(self):
        table = sql_table.SqlTable(self.iris_uri)
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
        table = sql_table.SqlTable(self.iris_uri)
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

    def test_type_hints(self):
        table = sql_table.SqlTable(self.iris_uri, guess_values=True)
        self.assertEqual(len(table.domain), 5)
        self.assertEqual(len(table.domain.metas), 0)
        table = sql_table.SqlTable(self.iris_uri, guess_values=True,
                   type_hints=Domain([], [], metas=[StringVariable("iris")])) 
        self.assertEqual(len(table.domain), 4)
        self.assertEqual(len(table.domain.metas), 1)

    def test_joins(self):
        table = sql_table.SqlTable.from_sql(
            get_dburi(),
            sql="""SELECT a."sepal length",
                          b. "petal length",
                          CASE WHEN b."petal length" < 3 THEN '<'
                               ELSE '>'
                           END AS "qualitative petal length"
                     FROM iris a
               INNER JOIN iris b ON a."sepal width" = b."sepal width"
                    WHERE a."petal width" < 1
                 ORDER BY a."petal width" ASC""",
            type_hints=Domain([DiscreteVariable( \
                name="qualitative petal length", values=['<', '>'])], []))

        self.assertEqual(len(table), 498)
        self.assertEqual(list(table[497]), [4.9, 5.1, 1.])

    def _mock_attribute(self, attr_name, formula=None):
        if formula is None:
            formula = '"%s"' % attr_name

        class attr:
            name = attr_name

            @staticmethod
            def to_sql():
                return formula

        return attr

    def test_get_attributes_from_sql_parse(self):
        sql = [
            "SELECT",
            "a.test AS a_test,",
            "CASE WHEN a.n = 1 THEN 1",
            "WHEN a.n = 2 THEN 2",
            "ELSE 0",
            "END AS c,",
            "b.test,",
            "c.\"another test\"",
            "FROM",
            '"table" a',
            'INNER JOIN "table" b ON a.x = b.x',
            'CROSS JOIN "table" c ON b.x = c.x',
            'LEFT OUTER JOIN "table" d on c.x = d.x',
            'RIGHT OUTER JOIN "table" e on d.x = e.x',
            'FULL OUTER JOIN "table" f on f.x = e.x',
            'WHERE',
            '1 = 1',
            'GROUP BY',
            'a.group',
            'HAVING',
            'a.haver = 1',
            'LIMIT',
            '1',
            'OFFSET',
            '1',
        ]

        p = SqlParser(" ".join(sql))
        self.assertEqual(
            p.from_.replace(" ", ""),
            "".join(sql[9:15]).replace(" ", ""))
        self.assertEqual(
            p.where.replace(" ", ""),
            "".join(sql[16:17]).replace(" ", ""))
        self.assertEqual(
            p.sql_without_limit.replace(" ", ""),
            "".join(sql[:21]).replace(" ", ""))

    def test_raises_on_unsupported_keywords(self):
        with self.assertRaises(ValueError):
            SqlParser("SELECT * FROM table FOR UPDATE")

    def test_universal_table(self):
        uri, table_name = self.construct_universal_table()

        table = sql_table.SqlTable.from_sql(uri, sql="""
            SELECT
                v1.col2 as v1,
                v2.col2 as v2,
                v3.col2 as v3,
                v4.col2 as v4,
                v5.col2 as v5
              FROM %(table_name)s v1
        INNER JOIN %(table_name)s v2 ON v2.col0 = v1.col0 AND v2.col1 = 2
        INNER JOIN %(table_name)s v3 ON v3.col0 = v2.col0 AND v3.col1 = 3
        INNER JOIN %(table_name)s v4 ON v4.col0 = v1.col0 AND v4.col1 = 4
        INNER JOIN %(table_name)s v5 ON v5.col0 = v1.col0 AND v5.col1 = 5
             WHERE v1.col1 = 1
          ORDER BY v1.col0
        """ % dict(table_name='"%s"' % table_name))

        self.drop_sql_table(table_name)


    def construct_universal_table(self):
        values = []
        for r in range(1, 6):
            for c in range(1, 6):
                values.extend((r, c, r * c))
        table = Table(np.array(values).reshape((-1, 3)))
        uri = self.create_sql_table(table)
        return uri.rsplit('/', 1)

    def test_class_var_type_hints(self):
        iris = sql_table.SqlTable(self.iris_uri, 
                    type_hints=Domain([], DiscreteVariable("iris", 
                        values=['Iris-setosa', 'Iris-virginica', 
                                'Iris-versicolor'])))

        self.assertEqual(len(iris.domain.class_vars), 1)
        self.assertEqual(iris.domain.class_vars[0].name, 'iris')

    def test_metas_type_hints(self):
        iris = sql_table.SqlTable(self.iris_uri,
                    type_hints=Domain([], [], metas=[DiscreteVariable("iris", 
                        values=['Iris-setosa', 'Iris-virginica', 
                                'Iris-versicolor'])]))

        self.assertEqual(len(iris.domain.metas), 1)
        self.assertEqual(iris.domain.metas[0].name, 'iris')

    def test_select_all(self):
        iris = sql_table.SqlTable.from_sql(
            self.iris_uri,
            sql='SELECT * FROM iris',
            type_hints=Domain([], DiscreteVariable("iris", 
                 values=['Iris-setosa', 'Iris-virginica', 
                'Iris-versicolor']))
            )

        self.assertEqual(len(iris.domain), 5)
