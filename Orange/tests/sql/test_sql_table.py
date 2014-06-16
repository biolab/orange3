import unittest
from Orange.data.sql import table as sql_table
from Orange.data import filter
from Orange.tests.sql.base import PostgresTest


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
            VarTypes = float_attr.VarTypes

            self.assertEqual(float_attr.var_type, VarTypes.Continuous)
            self.assertEqual(float_attr.name, "col0")
            self.assertEqual(float_attr.to_sql(), '"col0"')

            self.assertEqual(discrete_attr.var_type, VarTypes.Discrete)
            self.assertEqual(discrete_attr.name, "col1")
            self.assertEqual(discrete_attr.to_sql(), '"col1"')
            self.assertEqual(discrete_attr.values, ['f', 'm'])

            self.assertEqual(string_attr.var_type, VarTypes.String)
            self.assertEqual(string_attr.name, "col2")
            self.assertEqual(string_attr.to_sql(), '"col2"')

    def test_len(self):
        with self.sql_table_from_data(zip(self.float_variable(26))) as table:
            self.assertEqual(len(table), 26)

        with self.sql_table_from_data(zip(self.float_variable(0))) as table:
            self.assertEqual(len(table), 0)

    def test_len_with_filter(self):
        with self.sql_table_from_data(zip(self.discrete_variable(26))) as table:
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
            'INNER JOIN "table" c ON b.x = c.x',
            'WHERE',
            '1 = 1',
            'GROUP BY',
            'a.group',
            'HAVING',
            'a.haver = 1',
            'LIMIT',
            '1'
        ]

        p = SqlParser(" ".join(sql))
        self.assertEqual(
            p.from_.replace(" ", ""),
            "".join(sql[9:12]).replace(" ", ""))
        self.assertEqual(
            p.where.replace(" ", ""),
            "".join(sql[13:14]).replace(" ", ""))
        self.assertEqual(
            p.sql_without_limit.replace(" ", ""),
            "".join(sql[:18]).replace(" ", ""))
