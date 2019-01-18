# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
from unittest.mock import Mock

import numpy as np
import scipy.sparse as sp

from Orange.classification import NaiveBayesLearner
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.evaluation import CrossValidation, CA


class TestNaiveBayesLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = data = Table('titanic')
        cls.learner = NaiveBayesLearner()
        cls.table = data[::20]

    def setUp(self):
        self.model = self.learner(self.data)

    def test_NaiveBayes(self):
        results = CrossValidation(self.table, [self.learner], k=10)
        ca = CA(results)
        self.assertGreater(ca, 0.7)
        self.assertLess(ca, 0.9)

        results = CrossValidation(Table("iris"), [self.learner], k=10)
        ca = CA(results)
        self.assertGreater(ca, 0.7)

    def test_predict_single_instance(self):
        for ins in self.table:
            self.model(ins)
            val, prob = self.model(ins, self.model.ValueProbs)

    def test_predict_table(self):
        self.model(self.table)
        vals, probs = self.model(self.table, self.model.ValueProbs)

    def test_predict_numpy(self):
        X = self.table.X[::20]
        self.model(X)
        vals, probs = self.model(X, self.model.ValueProbs)

    def test_degenerate(self):
        d = Domain((ContinuousVariable(name="A"),
                    ContinuousVariable(name="B"),
                    ContinuousVariable(name="C")),
                   DiscreteVariable(name="CLASS", values=["M", "F"]))
        t = Table(d, [[0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1]])
        nb = NaiveBayesLearner()
        model = nb(t)
        self.assertEqual(model.domain.attributes, ())
        self.assertEqual(model(t[0]), 1)
        self.assertTrue(all(model(t) == 1))

    def test_allnan_cv(self):
        # GH 2740
        data = Table('voting')
        results = CrossValidation(data, [self.learner])
        self.assertFalse(any(results.failed))

    def test_sparse(self):
        _, dense_p = self.model.predict(self.data.X)

        _, csc_p = self.model.predict(sp.csc_matrix(self.data.X))
        np.testing.assert_almost_equal(dense_p, csc_p)

        _, csr_p = self.model.predict(sp.csr_matrix(self.data.X))
        np.testing.assert_almost_equal(dense_p, csr_p)

    def test_prediction_routing(self):
        data = self.data
        predict = self.model.predict = Mock(return_value=(data.Y, None))

        self.model(data)
        predict.assert_called()
        predict.reset_mock()

        self.model(data.X)
        predict.assert_called()
        predict.reset_mock()

        self.model.predict_storage(data)
        predict.assert_called()
        predict.reset_mock()

        self.model.predict_storage(data[0])
        predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
