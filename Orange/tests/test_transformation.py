import unittest
import numpy as np

from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable, \
    StringVariable
from Orange.preprocess.transformation import Identity


class TestTransformation(unittest.TestCase):
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
