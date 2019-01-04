import unittest
from unittest.mock import patch, MagicMock

from Orange.data import DiscreteVariable, Domain
from Orange.data.sql.table import SqlTable
from Orange.projection.pca import RemotePCA
from Orange.tests.sql.base import DataBaseTest as dbt


class PCATest(unittest.TestCase, dbt):
    def setUpDB(self):
        self.conn, self.iris = self.create_iris_sql_tabel()

    def tearDownDB(self):
        self.drop_iris_sql_tabel()

    @dbt.run_on(["postgres"])
    @patch("Orange.projection.pca.save_state", MagicMock())
    def test_PCA(self):
        table = SqlTable(self.conn, self.iris,
                         type_hints=Domain([], DiscreteVariable("iris",
                                values=['Iris-setosa', 'Iris-virginica',
                                        'Iris-versicolor'])))
        for batch_size in (50, 500):
            rpca = RemotePCA(table, batch_size, 20)
            self.assertEqual(rpca.components_.shape, (4, 4))
