import contextlib
import os
import string
import unittest
from urllib import parse
import uuid
import random

import Orange
from Orange.data.sql.table import SqlTable


def postgresql_test(f):
    try:
        import psycopg2
        return unittest.skipIf(not postgressql_version, "Database is not running.")(f)
    except:
        return unittest.skip("Psycopg2 is required for sql tests.")(f)


def mssql_test(f):
    try:
        import pymssql
        return unittest.skipIf(not mssql_active, "Database is not running.")(f)
    except:
        return unittest.skip("pymssql is required for mssql tests.")(f)


def connection_params():
    return dict(parse_uri(uri) for uri in get_dburi())


def get_dburi():
    dburi = os.environ.get('ORANGE_TEST_DB_URI').split("|")
    if dburi:
        return dburi
    else:
        return ["postgres://localhost/test"]


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
    return parsed_uri.scheme, params


try:
    import psycopg2
    with psycopg2.connect(**connection_params()['postgres']) as conn:
        postgressql_version = conn.server_version
except:
    postgressql_version = 0

try:
    import pymssql
    with pymssql.connect(**connection_params()['mssql']) as conn:
        mssql_active = True
except:
    mssql_active = False


def create_iris(db, param):
    iris = Orange.data.Table("iris")
    with db.connect(**param) as conn:
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


class TestParseUri(unittest.TestCase):
    def test_parses_connection_uri(self):
        scheme, parameters = parse_uri(
            "sql://user:password@host:7678/database/table")

        self.assertEqual("sql", scheme)
        self.assertDictContainsSubset(dict(
            host="host",
            user="user",
            password="password",
            port=7678,
            database="database",
            table="table"
        ), parameters)

    def test_parse_minimal_connection_uri(self):
        scheme, parameters = parse_uri(
            "sql://host/database/table")

        self.assertEqual("sql", scheme)
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


class DBTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raise NotImplementedError

    @classmethod
    def tearDownClass(cls):
        raise NotImplementedError

    def create_sql_table(self, data, columns=None):
        table_name = self._create_sql_table(data, columns)
        return self.conn, table_name

    @contextlib.contextmanager
    def sql_table_from_data(self, data, guess_values=True):
        table_name = self._create_sql_table(data)
        yield SqlTable(self.conn, table_name,
                       inspect_values=guess_values)
        self.drop_sql_table(table_name)

    def _create_sql_table(self, data, sql_column_types=None):
        raise NotImplementedError

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
        raise NotImplementedError

    def float_variable(self, size):
        return [i * .1 for i in range(size)]

    def discrete_variable(self, size):
        return ["mf"[i % 2] for i in range(size)]

    def string_variable(self, size):
        return string.ascii_letters[:size]


class PostgresTest(DBTest):
    @classmethod
    def setUpClass(cls):
        from psycopg2.pool import ThreadedConnectionPool
        from Orange.data.sql.backend.postgres import Psycopg2Backend

        Psycopg2Backend.connection_pool = \
            ThreadedConnectionPool(1, 1, **connection_params()['postgres'])
        cls.backend = Psycopg2Backend(connection_params()['postgres'])
        create_iris(psycopg2, connection_params()['postgres'])
        cls.iris = 'iris'
        cls.conn = [uri for uri in get_dburi() if 'postgres' in uri][0]

    @classmethod
    def tearDownClass(cls):
        from Orange.data.sql.backend.postgres import Psycopg2Backend
        Psycopg2Backend.connection_pool.closeall()
        Psycopg2Backend.connection_pool = None

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
        with self.backend.execute_sql_query(create_table_sql): pass
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
            with self.backend.execute_sql_query(insert_sql): pass
        return str(table_name)

    def drop_sql_table(self, table_name):
        with self.backend.execute_sql_query("""DROP TABLE "%s" """ % table_name): pass


class MicrosoftSQLTest(DBTest):
    @classmethod
    def setUpClass(cls):
        cls.iris = 'iris'
        cls.conn = [uri for uri in get_dburi() if 'mssql' in uri][0]

    @classmethod
    def tearDownClass(cls):
        pass

    def _create_sql_table(self, data, sql_column_types=None):
        data = list(data)
        if sql_column_types is None:
            column_size = self._get_column_types(data)
            sql_column_types = [
                'float' if size == 0 else 'varchar(%s)' % size
                for size in column_size
            ]
        table_name = ''.join(random.choices(string.ascii_lowercase, k=16))
        create_table_sql = 'CREATE TABLE %(table_name)s (%(columns)s)' % dict(
            table_name=table_name,
            columns=", ".join(
                'col%d %s' % (i, t) for i, t in enumerate(sql_column_types)
            )
        )
        insert_sql = 'INSERT INTO %(table_name)s VALUES (%(values)s)' % dict(
            table_name=table_name,
            values=', '.join(
                "%d" if c == "float" else "%s" for c in sql_column_types
            )
        )
        _, conn_params = parse_uri(self.conn)
        with pymssql.connect(**conn_params) as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_table_sql)
                cursor.executemany(insert_sql, [tuple(d) for d in data])
            conn.commit()

        return str(table_name)

    def drop_sql_table(self, table_name):
        _, conn_params = parse_uri(self.conn)
        with pymssql.connect(**conn_params) as conn:
            with conn.cursor() as cursor:
                cursor.execute("DROP TABLE %s" % table_name)
            conn.commit()
