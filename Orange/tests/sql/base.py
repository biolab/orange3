import contextlib
import os
import string
import unittest
import uuid
from Orange.data.sql.table import SqlTable

try:
    import psycopg2
    has_psycopg2 = True
except ImportError:
    has_psycopg2 = False

import Orange
from Orange.data.sql import table as sql_table


def connection_params():
    return SqlTable.parse_uri(get_dburi())


def get_dburi():
    dburi = os.environ.get('ORANGE_TEST_DB_URI')
    if dburi:
        return dburi
    else:
        return "postgres://localhost/test"


def create_iris():
    iris = Orange.data.Table("iris")
    with psycopg2.connect(**connection_params()) as conn:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS iris")
        cur.execute("""
            CREATE TABLE iris (
                "sepal length" float,
                "sepal width" float,
                "petal length" float,
                "petal width" float,
                "iris" varchar(15)
            )
        """)
        for row in iris:
            values = []
            for i, val in enumerate(row):
                if i != 4:
                    values.append(str(val))
                else:
                    values.append(iris.domain.class_var.values[int(val)])
            cur.execute("""INSERT INTO iris VALUES
            (%s, %s, %s, %s, '%s')""" % tuple(values))
    return get_dburi() + '/iris'


@unittest.skipIf(not has_psycopg2, "Psycopg2 is required for sql tests.")
class PostgresTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SqlTable.connection_pool = \
            psycopg2.pool.ThreadedConnectionPool(1, 1, **connection_params())
        cls.iris_uri = create_iris()

    @classmethod
    def tearDownClass(cls):
        SqlTable.connection_pool.closeall()
        SqlTable.connection_pool = None

    def create_sql_table(self, data):
        table_name = self._create_sql_table(data)
        self.table_name = str(table_name)
        return get_dburi() + '/' + str(table_name)

    @contextlib.contextmanager
    def sql_table_from_data(self, data, guess_values=True):
        assert SqlTable.connection_pool is not None

        table_name = self._create_sql_table(data)
        yield SqlTable(get_dburi() + '/' \
            + str(table_name), guess_values=guess_values)
        self.drop_sql_table(table_name)

    def _create_sql_table(self, data):
        data = list(data)
        column_size = self._get_column_types(data)
        sql_column_types = [
            'float' if size == 0 else 'varchar(%s)' % size
            for size in column_size
        ]
        table_name = uuid.uuid4()
        create_table_sql = """
            CREATE TEMPORARY TABLE "%(table_name)s" (
                %(columns)s
            )
        """ % dict(
            table_name=table_name,
            columns=",\n".join(
                'col%d %s' % (i, t)
                for i, t in enumerate(sql_column_types)
            )
        )
        conn = SqlTable.connection_pool.getconn()
        cur = conn.cursor()
        cur.execute(create_table_sql)
        for row in data:
            values = []
            for v, t in zip(row, sql_column_types):
                if v is None:
                    values.append('NULL')
                elif t == 0:
                    values.append(str(v))
                else:
                    values.append("'%s'" % v)
            insert_sql = """
            INSERT INTO "%(table_name)s" VALUES
                (%(values)s)
            """ % dict(
                table_name=table_name,
                values=', '.join(values)
            )
            cur.execute(insert_sql)
        conn.commit()
        SqlTable.connection_pool.putconn(conn)
        return table_name

    def _get_column_types(self, data):
        if not data:
            return [0] * 3
        column_size = [0] * len(data[0])
        for row in data:
            for i, value in enumerate(row):
                if isinstance(value, str):
                    column_size[i] = max(len(value), column_size[i])
        return column_size

    def drop_sql_table(self, table_name):
        conn = SqlTable.connection_pool.getconn()
        cur = conn.cursor()
        cur.execute("""DROP TABLE "%s" """ % table_name)
        conn.commit()
        SqlTable.connection_pool.putconn(conn)


    def float_variable(self, size):
        return [i*.1 for i in range(size)]

    def discrete_variable(self, size):
        return ["mf"[i % 2] for i in range(size)]

    def string_variable(self, size):
        return string.ascii_letters[:size]
