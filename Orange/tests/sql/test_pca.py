import unittest
from unittest.mock import patch, MagicMock

from Orange.data import DiscreteVariable, Domain
from Orange.data.sql.table import SqlTable
from Orange.projection.pca import RemotePCA
from Orange.tests.sql.base import sql_test, connection_params


@sql_test
class PCATest(unittest.TestCase):
    @patch("Orange.projection.pca.save_state", MagicMock())
    def test_PCA(self):
        table = SqlTable(connection_params(), 'iris',
                         type_hints=Domain([], DiscreteVariable("iris",
                                values=['Iris-setosa', 'Iris-virginica',
                                        'Iris-versicolor'])))
        for batch_size in (50, 500):
            rpca = RemotePCA(table, batch_size, 20)
            self.assertEqual(rpca.components_.shape, (4, 4))


if __name__ == '__main__':
    unittest.main()
