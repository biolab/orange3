import unittest
import numpy as np
from Orange.modelling.scite import SciteTreeLearner, SciteVariable, generate_scite_data
from Orange.data import Domain, Table


class TestScite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        n, m = 10, 10
        cls.data = generate_scite_data(n, m, val=SciteVariable.MUT_HOM)
        cls.domain = cls.data.domain

        # Same data but with heterozygous mutations and shuffled order
        cls.test_order = np.arange(n, dtype=int)
        np.random.shuffle(cls.test_order)
        test_data = generate_scite_data(n, m, val=SciteVariable.MUT_HET)[cls.test_order]
        cls.test_data = Table.from_numpy(X=test_data.X, domain=cls.domain)

        # Data with all missing values
        U = np.zeros((n, m))
        U[:, :] = SciteVariable.__values__.index(SciteVariable.MUT_UNKNOWN)
        cls.unknown_data = Table.from_numpy(cls.domain, U, Y=None, W=None)

        # Data with all missing values
        N = np.zeros((n, m))
        N[:, :] = SciteVariable.__values__.index(SciteVariable.MUT_NONE)
        cls.none_data = Table.from_numpy(cls.domain, N, Y=None, W=None)

        # Infer one model for all tests
        cls.learner = SciteTreeLearner(loops=1000, rep=1)
        cls.model = cls.learner(cls.data)

    def test_tree_structure(self):
        self.assertEqual(self.model.node_count(), 2 * len(self.domain)+1)
        self.assertEqual(self.model.leaf_count(), len(self.data)+1)
        self.assertEqual(self.model.depth(), len(self.data))

        i = 0
        node = self.model.root
        while i < len(self.domain):
            self.assertTrue(node.attr == self.domain[i])
            self.assertTrue(node.children[0].attr is None)
            node = node.children[1]
            i = i + 1

    def test_prediction(self):
        """ Test prediction on training data. """
        predictions = self.model(self.data)
        self.assertTrue(all(predictions.ravel() == self.model.instances.Y.ravel()))

    def test_extrapolation(self):
        """ Test prediction on test data. """
        predictions = self.model(self.test_data)
        self.assertTrue(all(predictions.ravel() == self.model.instances.Y.ravel()[self.test_order]))

    def test_unknown(self):
        """ Test prediction on test data. With all values unknown, prediction defaults to last feature. """
        predictions = self.model(self.unknown_data)
        self.assertTrue(all(predictions.ravel() == len(self.domain)))

    def test_none(self):
        """ Test prediction on test data. With all values none, prediction defaults to root (0). """
        predictions = self.model(self.none_data)
        self.assertTrue(all(predictions.ravel() == 0))

    def test_big_data(self):
        """ Test successful execution of the wrapped C++ code with 'big' data."""
        n, m = 100, 100
        X = np.zeros((n, m))
        for i in range(m):
            X[i:, i] = SciteVariable.__values__.index(SciteVariable.MUT_HOM)
        big_domain = Domain([SciteVariable(name="g-%d" % i) for i in range(m)])
        big_data = Table.from_numpy(big_domain, X, Y=None, W=None)
        learner = SciteTreeLearner(loops=1000, rep=1)
        _ = learner(big_data)
