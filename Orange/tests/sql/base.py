import os
import unittest

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


def destroy_iris(dburi):
    pass


class PostgresTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dburi = get_dburi()
        create_iris(dburi)

        cls.iris_uri = dburi + '/iris'

    @classmethod
    def tearDownClass(cls):
        pass
