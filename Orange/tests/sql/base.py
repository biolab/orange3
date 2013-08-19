from collections import defaultdict
import os
import unittest
import uuid

import psycopg2

import Orange


def get_dburi():
    dburi = os.environ.get('ORANGE_TEST_DB_URI')
    if dburi:
        pass
    else:
        return "postgres://localhost/test"


def create_iris(dburi):
    iris = Orange.data.Table("iris")
    conn = psycopg2.connect(dburi)
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
    conn.commit()
    conn.close()


class PostgresTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dburi = get_dburi()
        create_iris(dburi)

        cls.iris_uri = dburi + '/iris'

    @classmethod
    def tearDownClass(cls):
        pass

    def create_sql_table(self, data):
        column_size = self._get_column_types(data)
        sql_column_types = [
            'float' if size == 0 else 'varchar(%s)' % size
            for size in column_size
        ]
        table_name = uuid.uuid4()
        create_table_sql = """
            CREATE TABLE "%(table_name)s" (
                %(columns)s
            )
        """ % dict(
            table_name=table_name,
            columns=",\n".join(
                'col%d %s' % (i, t)
                for i, t in enumerate(sql_column_types)
            )
        )
        conn = psycopg2.connect(get_dburi())
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
        conn.close()
        self.table_name = str(table_name)
        return get_dburi() + '/' + str(table_name)

    def _get_column_types(self, data):
        assert len(data) > 0
        column_size = [0] * len(data[0])
        for row in data:
            for i, value in enumerate(row):
                if isinstance(value, str):
                    column_size[i] = max(len(value), column_size[i])
        return column_size

    def drop_sql_table(self, table_name):
        conn = psycopg2.connect(get_dburi())
        cur = conn.cursor()
        cur.execute("""DROP TABLE "%s" """ % table_name)
        conn.commit()
        conn.close()
