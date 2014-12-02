import contextlib
import os
import string
import unittest
from urllib import parse
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
    return parse_uri(get_dburi())


def get_dburi():
    dburi = os.environ.get('ORANGE_TEST_DB_URI')
    if dburi:
        return dburi
    else:
        return "postgres://localhost/test"


def parse_uri(uri):
    parsed_uri = parse.urlparse(uri)
    database = parsed_uri.path.strip('/')
    if "/" in database:
        database, table = database.split('/', 1)
    else:
        table = ""

    params = parse.parse_qs(parsed_uri.query)
    for key, value in params.items():
        if len(params[key]) == 1:
            params[key] = value[0]

    params.update(dict(
        host=parsed_uri.hostname,
        port=parsed_uri.port,
        user=parsed_uri.username,
        database=database,
        password=parsed_uri.password,
    ))
    if table:
        params['table'] = table
    return params


def server_version():
    if has_psycopg2:
        with psycopg2.connect(**connection_params()) as conn:
            return conn.server_version
    else:
        return 0


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
    return get_dburi(), 'iris'


class ParseUriTests(unittest.TestCase):
    def test_parses_connection_uri(self):
        parameters = parse_uri(
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
        parameters = parse_uri(
            "sql://host/database/table")

        self.assertDictContainsSubset(
            dict(host="host", database="database", table="table"),
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
class PostgresTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SqlTable.connection_pool = \
            psycopg2.pool.ThreadedConnectionPool(1, 1, **connection_params())
        cls.conn, cls.iris = create_iris()

    @classmethod
    def tearDownClass(cls):
        SqlTable.connection_pool.closeall()
        SqlTable.connection_pool = None

    def create_sql_table(self, data, columns=None):
        table_name = self._create_sql_table(data, columns)
        return connection_params(), table_name

    @contextlib.contextmanager
    def sql_table_from_data(self, data, guess_values=True):
        assert SqlTable.connection_pool is not None

        table_name = self._create_sql_table(data)
        yield SqlTable(connection_params(), table_name,
                       inspect_values=guess_values)
        self.drop_sql_table(table_name)

    def _create_sql_table(self, data, sql_column_types=None):
        data = list(data)
        if sql_column_types is None:
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
        return str(table_name)

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
