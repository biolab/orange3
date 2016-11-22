import unittest
import numpy as np

from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable, \
    StringVariable
from Orange.preprocess.transformation import Identity, Transformation


class TestTransformation(unittest.TestCase):
    class TransformationMock(Transformation):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.called_with = None

        def transform(self, col):
            self.called_with = col
            return np.arange(len(col))

    @classmethod
    def setUpClass(cls):
        cls.data = Table("zoo")

    def test_call(self):
        """Call passes the column to `transform` and returns its results"""
        data = self.data
        trans = self.TransformationMock(data.domain[2])
        np.testing.assert_almost_equal(trans(data), np.arange(len(data)))
        np.testing.assert_array_equal(trans.called_with, data.X[:, 2])

        np.testing.assert_almost_equal(trans(data[0]), np.array([0]))
        np.testing.assert_array_equal(trans.called_with, data.X[0, 2])

        trans = self.TransformationMock(data.domain.metas[0])
        np.testing.assert_almost_equal(trans(data), np.arange(len(data)))
        np.testing.assert_array_equal(trans.called_with, data.metas.flatten())

        np.testing.assert_almost_equal(trans(data[0]), np.array([0]))
        np.testing.assert_array_equal(trans.called_with, data.metas[0, 0])

    def test_transform_fails(self):
        trans = Transformation(self.data.domain[2])
        self.assertRaises(NotImplementedError, trans, self.data)

    def test_identity(self):
        domain = Domain([ContinuousVariable("X")],
                        [DiscreteVariable("C", values=["0", "1", "2"])],
                        [StringVariable("S")])
        X = np.random.normal(size=(4, 1))
        Y = np.random.randint(3, size=(4, 1))
        M = np.array(["A", "B", "C", "D"], dtype=object).reshape(-1, 1)

        D = Table.from_numpy(domain, X, Y, metas=M)
        X1 = domain[0].copy(compute_value=Identity(domain[0]))
        Y1 = domain[1].copy(compute_value=Identity(domain[1]))
        S1 = domain.metas[0].copy(compute_value=Identity(domain.metas[0]))
        domain_1 = Domain([X1], [Y1], [S1])
        D1 = Table.from_table(domain_1, D)

        np.testing.assert_equal(D1.X, D.X)
        np.testing.assert_equal(D1.Y, D.Y)
        np.testing.assert_equal(D1.metas, D.metas)
