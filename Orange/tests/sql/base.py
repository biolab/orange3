import os
import string
import unittest
from urllib import parse
import random
import inspect

import numpy as np

from Orange.data import Table


def parse_uri(uri):
    """Parse uri to db type and dictionary of connection parameters."""
    if uri == "":
        return "", dict()
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


class TestParseUri(unittest.TestCase):
    def test_parses_connection_uri(self):
        parameters = parse_uri(
            "sql://user:password@host:7678/database/table")

        self.assertEqual("sql", parameters[0])
        self.assertDictContainsSubset(dict(
            host="host",
            user="user",
            password="password",
            port=7678,
            database="database",
            table="table"
        ), parameters[1])

    def test_parse_minimal_connection_uri(self):
        parameters = parse_uri(
            "sql://host/database/table")

        self.assertEqual("sql", parameters[0])
        self.assertDictContainsSubset(
            dict(host="host", database="database", table="table"),
            parameters[1]
        )

    def test_parse_empty(self):
        parameters = parse_uri("")

        self.assertEqual("", parameters[0])
        self.assertDictContainsSubset(
            dict(),
            parameters[1]
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


def connection_params():
    """
    Get environment variable that holds db connection parameters.
    You can assign multiple connections by concatenating them with '|'.
    """
    dburi = os.environ.get('ORANGE_TEST_DB_URI', "")
    return dict(parse_uri(uri) for uri in dburi.split("|"))


class DBTestConnection:
    """Used to prepare the connection to the database"""
    uri_name = ""
    module = ""

    def __init__(self, params):
        self._params = params
        self.is_module = False
        self.is_active = False
        self.try_connection()

    @property
    def params(self):
        return self._params.copy()

    def try_connection(self):
        """
        Test if db specific connection module is installed and if you can
        connect to db.
        """
        raise NotImplementedError()

    def create_sql_table(self, data, sql_column_types=None,
                         sql_column_names=None, table_name=None):
        """This creates a new sql table in the specific db."""
        raise NotImplementedError()

    @staticmethod
    def _get_column_types(data):
        if not data:
            return [0] * 3
        column_size = [0] * len(data[0])
        for row in data:
            for i, value in enumerate(row):
                if isinstance(value, str):
                    column_size[i] = max(len(value), column_size[i])

        return column_size

    def drop_sql_table(self, table_name):
        """This drops given sql table in the specific db."""
        raise NotImplementedError

    def get_backend(self):
        """This returns the db specific Orange Backend."""
        raise NotImplementedError


class PostgresTestConnection(DBTestConnection):
    uri_name = "postgres"
    module = "psycopg2"

    def try_connection(self):
        try:
            import psycopg2
            self.is_module = True
            with psycopg2.connect(**self.params) as conn:
                self.is_active = True
                self.version = conn.server_version
        except:
            pass

    def create_sql_table(self, data, sql_column_types=None,
                         sql_column_names=None, table_name=None):
        data = list(data)

        if table_name is None:
            table_name = ''.join(random.choices(string.ascii_lowercase, k=16))

        if sql_column_types is None:
            column_size = self._get_column_types(data)
            sql_column_types = [
                'float' if size == 0 else 'varchar({})'.format(size)
                for size in column_size
            ]

        if sql_column_names is None:
            sql_column_names = ["col{}".format(i)
                                for i in range(len(sql_column_types))]
        else:
            sql_column_names = map(lambda x: '"{}"'.format(x), sql_column_names)

        drop_table_sql = "DROP TABLE IF EXISTS {}".format(table_name)

        create_table_sql = "CREATE TABLE {} ({})".format(
            table_name,
            ", ".join('{} {}'.format(n, t)
                      for n, t in zip(sql_column_names, sql_column_types)))

        insert_values = ", ".join(
            "({})".format(
                ", ".join("NULL" if v is None else "'{}'".format(v)
                          for v, t in zip(row, sql_column_types))
            ) for row in data
        )

        insert_sql = "INSERT INTO {} VALUES {}".format(table_name,
                                                       insert_values)

        import psycopg2
        with psycopg2.connect(**self.params) as conn:
            with conn.cursor() as curs:
                curs.execute(drop_table_sql)
            with conn.cursor() as curs:
                curs.execute(create_table_sql)
            if insert_values:
                with conn.cursor() as curs:
                    curs.execute(insert_sql)

        return self.params, table_name

    def drop_sql_table(self, table_name):
        import psycopg2
        with psycopg2.connect(**self.params) as conn:
            with conn.cursor() as curs:
                curs.execute("DROP TABLE {}".format(table_name))

    def get_backend(self):
        from Orange.data.sql.backend import Psycopg2Backend
        return Psycopg2Backend(self.params)


class MicrosoftTestConnection(DBTestConnection):
    uri_name = "mssql"
    module = "pymssql"

    def try_connection(self):
        try:
            import pymssql
            self.is_module = True
            with pymssql.connect(**self.params):
                self.is_active = True
        except:
            pass

    def create_sql_table(self, data, sql_column_types=None,
                         sql_column_names=None, table_name=None):
        data = list(data)

        if table_name is None:
            table_name = ''.join(random.choices(string.ascii_lowercase, k=16))

        if sql_column_types is None:
            column_size = self._get_column_types(data)
            sql_column_types = [
                'float' if size == 0 else 'varchar({})'.format(size)
                for size in column_size
            ]

        if sql_column_names is None:
            sql_column_names = ["col{}".format(i)
                                for i in range(len(sql_column_types))]
        else:
            sql_column_names = map(lambda x: '"{}"'.format(x), sql_column_names)

        drop_table_sql = "DROP TABLE IF EXISTS {}".format(table_name)

        create_table_sql = "CREATE TABLE {} ({})".format(
            table_name,
            ", ".join('{} {}'.format(n, t)
                      for n, t in zip(sql_column_names, sql_column_types)))

        insert_values = ", ".join(
            "({})".format(
                ", ".join("NULL" if v is None else "'{}'".format(v)
                          for v, t in zip(row, sql_column_types))
            ) for row in data
        )

        insert_sql = "INSERT INTO {} VALUES {}".format(table_name,
                                                       insert_values)

        import pymssql
        with pymssql.connect(**self.params) as conn:
            with conn.cursor() as cursor:
                cursor.execute(drop_table_sql)
                cursor.execute(create_table_sql)
                if insert_values:
                    cursor.execute(insert_sql)
            conn.commit()

        return self.params, table_name

    def drop_sql_table(self, table_name):
        import pymssql
        with pymssql.connect(**self.params) as conn:
            with conn.cursor() as cursor:
                cursor.execute("DROP TABLE {}".format(table_name))
            conn.commit()

    def get_backend(self):
        from Orange.data.sql.backend import PymssqlBackend
        return PymssqlBackend(self.params)


test_connections = {
    PostgresTestConnection.uri_name: PostgresTestConnection,
    MicrosoftTestConnection.uri_name: MicrosoftTestConnection
}


def dbs():
    """Parse env variable and initialize connection to given dbs."""
    params = connection_params()

    db_conn = {}
    for c in params:
        if c and c in test_connections:
            db_conn[c] = test_connections[c](params[c])

    return db_conn


class DataBaseTest:
    db_conn = dbs()

    @classmethod
    def _check_db(cls, db):
        if ">" in db:
            i = db.find(">")
            if db[:i] in cls.db_conn and \
                    cls.db_conn[db[:i]].version <= int(db[i + 1:]):
                raise unittest.SkipTest(
                    "This test is only run database version higher then {}"
                    .format(db[i + 1:]))
            else:
                db = db[:i]
        elif "<" in db:
            i = db.find("<")
            if db[:i] in cls.db_conn and \
                    cls.db_conn[db[:i]].version >= int(db[i + 1:]):
                raise unittest.SkipTest(
                    "This test is only run on database version lower then {}"
                    .format(db[i + 1:]))
            else:
                db = db[:i]

        if db in cls.db_conn:
            if not cls.db_conn[db].is_module:
                raise unittest.SkipTest(
                    "{} module is required for this database".format(
                        cls.db_conn[db].module))

            elif not cls.db_conn[db].is_active:
                raise unittest.SkipTest("Database is not running")
        else:
            if db in test_connections.keys():
                raise unittest.SkipTest("No connection provided for {}".format(db))
            else:
                raise Exception("Unsupported database")

        return db

    @classmethod
    def _setup_test_with(cls, test, db):
        """This is used to setup the `db` database and run `test` on it"""
        def new_test(test_self, *args):
            new_db = cls._check_db(db)

            test_self.current_db = new_db

            error = None
            if hasattr(test_self, "setUpDB"):
                test_self.setUpDB()
            try:
                test(test_self, *args)
            except Exception as ex:
                error = ex
            finally:
                if hasattr(test_self, "tearDownDB"):
                    test_self.tearDownDB()

            if error is not None:
                raise error

        return new_test

    @classmethod
    def run_on(cls, dbs):
        """
        Decorator used to create new test for every given database.

        For every db in the list of dbs create a new test that:
            * changes create_sql_table, drop_sql_table and backend to those
              implemented in the db subclass of DBTestConnection
            * runs setUpDB if exists
            * runs original test
            * runs tearDownDB if exists
        """
        def decorator(function):
            if function.__name__.startswith("test_db_"):
                return function

            stack = inspect.stack()
            frame = stack[1]
            frame_locals = frame[0].f_locals

            for db in dbs:
                name = 'test_db_' + db + '_' + function.__name__[5:]
                frame_locals[name] = cls._setup_test_with(function, db)
                frame_locals[name].__name__ = name
                frame_locals[name].place_as = function
                if function.__doc__ is not None:
                    frame_locals[name].__doc__ = 'On ' + db + ' run: ' + \
                                                 function.__doc__

            function.__test__ = False

        return decorator

    @property
    def backend(self):
        """This is according to the db currently being tested"""
        return self.db_conn[self.current_db].get_backend()

    def create_sql_table(self, data, sql_column_types=None,
                         sql_column_names=None, table_name=None):
        """This is according to the db currently being tested"""
        return self.db_conn[self.current_db]\
            .create_sql_table(data, sql_column_types=sql_column_types,
                              sql_column_names=sql_column_names,
                              table_name=table_name)

    def drop_sql_table(self, table_name):
        """This is according to the db currently being tested"""
        return self.db_conn[self.current_db]\
            .drop_sql_table(table_name=table_name)

    def create_iris_sql_table(self):
        iris = Table("iris")
        cn = ["sepal length", "sepal width", "petal length", "petal width",
              "iris"]
        ct = ["float", "float", "float", "float", "varchar(15)"]
        Y = np.array(iris.domain.class_var.values)[iris.Y.astype(int)]
        data = [list(x) + [y] for x, y in zip(iris.X, Y)]
        return self.create_sql_table(data, table_name="iris",
                                     sql_column_names=cn, sql_column_types=ct)

    def drop_iris_sql_table(self):
        self.drop_sql_table("iris")
