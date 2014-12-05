import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from Orange.data.sql import table as sql_table
from Orange.data import filter, ContinuousVariable, DiscreteVariable, \
    StringVariable, Table, Domain
from Orange.data.sql.table import SqlTable
from Orange.tests.sql.base import PostgresTest, get_dburi, has_psycopg2, \
    server_version


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
            self.assertTrue('"col0"' in float_attr.to_sql())

            self.assertIsInstance(discrete_attr, DiscreteVariable)
            self.assertEqual(discrete_attr.name, "col1")
            print(discrete_attr.to_sql())
            self.assertTrue('"col1"' in discrete_attr.to_sql())
            self.assertEqual(discrete_attr.values, ['f', 'm'])

            self.assertIsInstance(string_attr, StringVariable)
            self.assertEqual(string_attr.name, "col2")
            self.assertTrue('"col2"' in string_attr.to_sql())

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

    def test_XY_small(self):
        mat = np.random.randint(0, 2, (20, 3))
        conn, table_name = self.create_sql_table(mat)
        sql_table = SqlTable(conn, table_name,
                             type_hints=Domain([], DiscreteVariable(
                                 name='col2', values=['0', '1', '2'])))
        assert_almost_equal(sql_table.X, mat[:, :2])
        assert_almost_equal(sql_table.Y.flatten(), mat[:, 2])

    def test_XY_large(self):
        mat = np.random.randint(0, 2, (1020, 3))
        conn, table_name = self.create_sql_table(mat)
        sql_table = SqlTable(conn, table_name,
                             type_hints=Domain([], DiscreteVariable(
                                 name='col2', values=['0', '1', '2'])))
        with self.assertRaises(ValueError):
            sql_table.X
        with self.assertRaises(ValueError):
            sql_table.Y
        with self.assertRaises(ValueError):
            sql_table.download_data(1019)
        sql_table.download_data()
        assert_almost_equal(sql_table.X, mat[:, :2])
        assert_almost_equal(sql_table.Y.flatten(), mat[:, 2])


    def test_query_all(self):
        table = sql_table.SqlTable(self.conn, self.iris, inspect_values=True)
        results = list(table)

        self.assertEqual(len(results), 150)

    def test_query_subset_of_attributes(self):
        table = sql_table.SqlTable(self.conn, self.iris)
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
        table = sql_table.SqlTable(self.conn, self.iris)
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
        table = sql_table.SqlTable(self.conn, self.iris, inspect_values=True)
        self.assertEqual(len(table.domain), 5)
        self.assertEqual(len(table.domain.metas), 0)
        table = sql_table.SqlTable(self.conn, self.iris, inspect_values=True,
                                   type_hints=Domain([], [], metas=[
                                       StringVariable("iris")]))
        self.assertEqual(len(table.domain), 4)
        self.assertEqual(len(table.domain.metas), 1)

    def test_joins(self):
        table = sql_table.SqlTable(
            self.conn,
            """SELECT a."sepal length",
                          b. "petal length",
                          CASE WHEN b."petal length" < 3 THEN '<'
                               ELSE '>'
                           END AS "qualitative petal length"
                     FROM iris a
               INNER JOIN iris b ON a."sepal width" = b."sepal width"
                    WHERE a."petal width" < 1
                 ORDER BY a."sepal length", b. "petal length" ASC""",
            type_hints=Domain([DiscreteVariable(
                name="qualitative petal length",
                values=['<', '>'])], []))

        self.assertEqual(len(table), 498)
        self.assertAlmostEqual(list(table[497]), [5.8, 1.2, 0.])

    def _mock_attribute(self, attr_name, formula=None):
        if formula is None:
            formula = '"%s"' % attr_name

        class attr:
            name = attr_name

            @staticmethod
            def to_sql():
                return formula

        return attr

    def test_universal_table(self):
        uri, table_name = self.construct_universal_table()

        table = sql_table.SqlTable(self.conn, """
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
        return self.create_sql_table(table)

    def test_class_var_type_hints(self):
        iris = sql_table.SqlTable(self.conn, self.iris,
                                  type_hints=Domain([],
                                                      DiscreteVariable("iris",
                                                                       values=[
                                                                           'Iris-setosa',
                                                                           'Iris-virginica',
                                                                           'Iris-versicolor'])))

        self.assertEqual(len(iris.domain.class_vars), 1)
        self.assertEqual(iris.domain.class_vars[0].name, 'iris')

    def test_metas_type_hints(self):
        iris = sql_table.SqlTable(self.conn, self.iris,
                                  type_hints=Domain([], [], metas=[
                                      DiscreteVariable("iris",
                                                       values=['Iris-setosa',
                                                               'Iris-virginica',
                                                               'Iris-versicolor'])]))

        self.assertEqual(len(iris.domain.metas), 1)
        self.assertEqual(iris.domain.metas[0].name, 'iris')

    def test_select_all(self):
        iris = sql_table.SqlTable(
            self.conn,
            "SELECT * FROM iris",
            type_hints=Domain(
                [], DiscreteVariable("iris", values=['Iris-setosa',
                                                     'Iris-virginica',
                                                     'Iris-versicolor']))
        )

        self.assertEqual(len(iris.domain), 5)

    def test_discrete_bigint(self):
        table = np.arange(6).reshape((-1, 1))
        conn, table_name = self.create_sql_table(table, ['bigint'])

        sql_table = SqlTable(conn, table_name, inspect_values=False)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

        sql_table = SqlTable(conn, table_name, inspect_values=True)
        self.assertFirstAttrIsInstance(sql_table, DiscreteVariable)

    def test_continous_bigint(self):
        table = np.arange(25).reshape((-1, 1))
        conn, table_name = self.create_sql_table(table, ['bigint'])

        sql_table = SqlTable(conn, table_name, inspect_values=False)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

        sql_table = SqlTable(conn, table_name, inspect_values=True)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

    def test_discrete_int(self):
        table = np.arange(6).reshape((-1, 1))
        conn, table_name = self.create_sql_table(table, ['int'])

        sql_table = SqlTable(conn, table_name, inspect_values=False)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

        sql_table = SqlTable(conn, table_name, inspect_values=True)
        self.assertFirstAttrIsInstance(sql_table, DiscreteVariable)

    def test_continous_int(self):
        table = np.arange(25).reshape((-1, 1))
        conn, table_name = self.create_sql_table(table, ['int'])

        sql_table = SqlTable(conn, table_name, inspect_values=False)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

        sql_table = SqlTable(conn, table_name, inspect_values=True)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

    def test_discrete_smallint(self):
        table = np.arange(6).reshape((-1, 1))
        conn, table_name = self.create_sql_table(table, ['smallint'])

        sql_table = SqlTable(conn, table_name, inspect_values=False)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

        sql_table = SqlTable(conn, table_name, inspect_values=True)
        self.assertFirstAttrIsInstance(sql_table, DiscreteVariable)

    def test_continous_smallint(self):
        table = np.arange(25).reshape((-1, 1))
        conn, table_name = self.create_sql_table(table, ['smallint'])

        sql_table = SqlTable(conn, table_name, inspect_values=False)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

        sql_table = SqlTable(conn, table_name, inspect_values=True)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

    def test_boolean(self):
        table = np.array(['F', 'T', 0, 1, 'False', 'True']).reshape(-1, 1)
        conn, table_name = self.create_sql_table(table, ['boolean'])

        sql_table = SqlTable(conn, table_name, inspect_values=False)
        self.assertFirstAttrIsInstance(sql_table, DiscreteVariable)

        sql_table = SqlTable(conn, table_name, inspect_values=True)
        self.assertFirstAttrIsInstance(sql_table, DiscreteVariable)

    def test_discrete_char(self):
        table = np.array(['M', 'F', 'M', 'F', 'M', 'F']).reshape(-1, 1)
        conn, table_name = self.create_sql_table(table, ['char(1)'])

        sql_table = SqlTable(conn, table_name, inspect_values=False)
        self.assertFirstMetaIsInstance(sql_table, StringVariable)

        sql_table = SqlTable(conn, table_name, inspect_values=True)
        self.assertFirstAttrIsInstance(sql_table, DiscreteVariable)

    def test_meta_char(self):
        table = np.array(list('ABCDEFGHIJKLMNOPQRSTUVW')).reshape(-1, 1)
        conn, table_name = self.create_sql_table(table, ['char(1)'])

        sql_table = SqlTable(conn, table_name, inspect_values=False)
        self.assertFirstMetaIsInstance(sql_table, StringVariable)

        sql_table = SqlTable(conn, table_name, inspect_values=True)
        self.assertFirstMetaIsInstance(sql_table, StringVariable)

    def test_discrete_varchar(self):
        table = np.array(['M', 'F', 'M', 'F', 'M', 'F']).reshape(-1, 1)
        conn, table_name = self.create_sql_table(table, ['varchar(1)'])

        sql_table = SqlTable(conn, table_name, inspect_values=False)
        self.assertFirstMetaIsInstance(sql_table, StringVariable)

        sql_table = SqlTable(conn, table_name, inspect_values=True)
        self.assertFirstAttrIsInstance(sql_table, DiscreteVariable)

    def test_meta_varchar(self):
        table = np.array(list('ABCDEFGHIJKLMNOPQRSTUVW')).reshape(-1, 1)
        conn, table_name = self.create_sql_table(table, ['varchar(1)'])

        sql_table = SqlTable(conn, table_name, inspect_values=False)
        self.assertFirstMetaIsInstance(sql_table, StringVariable)

        sql_table = SqlTable(conn, table_name, inspect_values=True)
        self.assertFirstMetaIsInstance(sql_table, StringVariable)

    def test_date(self):
        table = np.array(['2014-04-12', '2014-04-13', '2014-04-14',
                          '2014-04-15', '2014-04-16']).reshape(-1, 1)
        conn, table_name = self.create_sql_table(table, ['date'])

        sql_table = SqlTable(conn, table_name, inspect_values=False)
        self.assertFirstMetaIsInstance(sql_table, StringVariable)

        sql_table = SqlTable(conn, table_name, inspect_values=True)
        self.assertFirstMetaIsInstance(sql_table, StringVariable)

    def test_double_precision(self):
        table = np.arange(25).reshape((-1, 1))
        conn, table_name = self.create_sql_table(table, ['double precision'])

        sql_table = SqlTable(conn, table_name, inspect_values=False)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

        sql_table = SqlTable(conn, table_name, inspect_values=True)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

    def test_numeric(self):
        table = np.arange(25).reshape((-1, 1))
        conn, table_name = self.create_sql_table(table, ['numeric(15, 2)'])

        sql_table = SqlTable(conn, table_name, inspect_values=False)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

        sql_table = SqlTable(conn, table_name, inspect_values=True)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

    def test_real(self):
        table = np.arange(25).reshape((-1, 1))
        conn, table_name = self.create_sql_table(table, ['real'])

        sql_table = SqlTable(conn, table_name, inspect_values=False)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

        sql_table = SqlTable(conn, table_name, inspect_values=True)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

    def test_serial(self):
        table = np.arange(25).reshape((-1, 1))
        conn, table_name = self.create_sql_table(table, ['serial'])

        sql_table = SqlTable(conn, table_name, inspect_values=False)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

        sql_table = SqlTable(conn, table_name, inspect_values=True)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

    @unittest.skipIf(server_version() < 90200,
                     "Type not supported on this server version.")
    def test_smallserial(self):
        table = np.arange(25).reshape((-1, 1))
        conn, table_name = self.create_sql_table(table, ['smallserial'])

        sql_table = SqlTable(conn, table_name, inspect_values=False)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

        sql_table = SqlTable(conn, table_name, inspect_values=True)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

    @unittest.skipIf(server_version() < 90200,
                     "Type not supported on this server version.")
    def test_bigserial(self):
        table = np.arange(25).reshape((-1, 1))
        conn, table_name = self.create_sql_table(table, ['bigserial'])

        sql_table = SqlTable(conn, table_name, inspect_values=False)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

        sql_table = SqlTable(conn, table_name, inspect_values=True)
        self.assertFirstAttrIsInstance(sql_table, ContinuousVariable)

    def test_text(self):
        table = np.array(list('ABCDEFGHIJKLMNOPQRSTUVW')).reshape((-1, 1))
        conn, table_name = self.create_sql_table(table, ['text'])

        sql_table = SqlTable(conn, table_name, inspect_values=False)
        self.assertFirstMetaIsInstance(sql_table, StringVariable)

        sql_table = SqlTable(conn, table_name, inspect_values=True)
        self.assertFirstMetaIsInstance(sql_table, StringVariable)

    def test_recovers_connection_after_sql_error(self):
        import psycopg2

        conn, table_name = self.create_sql_table(
            np.arange(25).reshape((-1, 1)))
        sql_table = SqlTable(conn, table_name)

        try:
            broken_query = "SELECT 1/%s FROM %s" % (
                sql_table.domain.attributes[0].to_sql(), sql_table.table_name)
            with sql_table._execute_sql_query(broken_query) as cur:
                cur.fetchall()
        except psycopg2.DataError:
            pass

        working_query = "SELECT %s FROM %s" % (
            sql_table.domain.attributes[0].to_sql(), sql_table.table_name)
        with sql_table._execute_sql_query(working_query) as cur:
            cur.fetchall()


    def assertFirstAttrIsInstance(self, table, variable_type):
        self.assertGreater(len(table.domain), 0)
        attr = table.domain[0]
        self.assertIsInstance(attr, variable_type)

    def assertFirstMetaIsInstance(self, table, variable_type):
        self.assertGreater(len(table.domain.metas), 0)
        attr = table.domain[-1]
        self.assertIsInstance(attr, variable_type)
