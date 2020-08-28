# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
from unittest.mock import Mock

import numpy as np
import scipy.sparse as sp

from Orange.classification import NaiveBayesLearner
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.evaluation import CrossValidation, CA


# This class is used to force predict_storage to fall back to the slower
# procedure instead of calling `predict`
from Orange.tests import test_filename


class NotATable(Table):  # pylint: disable=too-many-ancestors,abstract-method
    pass


class TestNaiveBayesLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = data = Table('titanic')
        cls.learner = NaiveBayesLearner()
        cls.table = data[::20]

    def setUp(self):
        self.model = self.learner(self.data)

    def test_NaiveBayes(self):
        cv = CrossValidation(k=10)
        results = cv(self.table, [self.learner])
        ca = CA(results)
        self.assertGreater(ca, 0.7)
        self.assertLess(ca, 0.9)

        cv = CrossValidation(k=10)
        results = cv(Table("iris"), [self.learner])
        ca = CA(results)
        self.assertGreater(ca, 0.7)

    def test_degenerate(self):
        d = Domain((ContinuousVariable(name="A"),
                    ContinuousVariable(name="B"),
                    ContinuousVariable(name="C")),
                   DiscreteVariable(name="CLASS", values=("M", "F")))
        t = Table.from_list(d, [[0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1]])
        nb = NaiveBayesLearner()
        model = nb(t)
        self.assertEqual(model.domain.attributes, ())
        self.assertEqual(model(t[0]), 1)
        self.assertTrue(all(model(t) == 1))

    def test_allnan_cv(self):
        # GH 2740
        data = Table(test_filename('datasets/lenses.tab'))
        cv = CrossValidation(stratified=False)
        results = cv(data, [self.learner])
        self.assertFalse(any(results.failed))

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
        predict.assert_called()

    def test_compare_results_of_predict_and_predict_storage(self):
        data2 = NotATable("titanic")

        self.model = self.learner(self.data[:50])
        predict = self.model.predict = Mock(side_effect=self.model.predict)
        values, probs = self.model.predict_storage(self.data[50:])
        predict.assert_called()
        predict.reset_mock()
        values2, probs2 = self.model.predict_storage(data2[50:])
        predict.assert_not_called()

        np.testing.assert_equal(values, values2)
        np.testing.assert_equal(probs, probs2)

    def test_predictions(self):
        self._test_predictions(sparse=None)
        self._test_predictions_with_absent_class(sparse=None)

    def test_predictions_csr_matrix(self):
        self._test_predictions(sparse=sp.csr_matrix)
        self._test_predictions_with_absent_class(sparse=sp.csr_matrix)

    def test_predictions_csc_matrix(self):
        self._test_predictions(sparse=sp.csc_matrix)
        self._test_predictions_with_absent_class(sparse=sp.csc_matrix)

    def _test_predictions(self, sparse):
        x = np.array([
            [1, 0, 0],
            [0, np.nan, 0],
            [0, 1, 0],
            [0, 0, 0],
            [1, 2, 0],
            [1, 1, 0],
            [1, 2, 0],
            [0, 1, 0]])
        if sparse is not None:
            x = sparse(x)

        y = np.array([0, 0, 0, 1, 1, 1, 2, 2])
        domain = Domain(
            [DiscreteVariable("a", values="ab"),
             DiscreteVariable("b", values="abc"),
             DiscreteVariable("c", values="a")],
            DiscreteVariable("y", values="abc"))
        data = Table.from_numpy(domain, x, y)

        model = self.learner(data)
        np.testing.assert_almost_equal(
            model.class_prob,
            [4/11, 4/11, 3/11]
        )
        np.testing.assert_almost_equal(
            np.exp(model.log_cont_prob[0]) * model.class_prob[:, None],
            [[3/7, 2/7], [2/7, 3/7], [2/7, 2/7]])
        np.testing.assert_almost_equal(
            np.exp(model.log_cont_prob[1]) * model.class_prob[:, None],
            [[2/5, 1/3, 1/5], [2/5, 1/3, 2/5], [1/5, 1/3, 2/5]])
        np.testing.assert_almost_equal(
            np.exp(model.log_cont_prob[2]) * model.class_prob[:, None],
            [[4/11], [4/11], [3/11]])

        test_x = np.array([[a, b, 0] for a in [0, 1] for b in [0, 1, 2]])
        # Classifiers reject csc matrices in the base class
        # Naive bayesian classifier supports them if predict_storage is
        # called directly, which we do below
        if sparse is not None and sparse is not sp.csc_matrix:
            test_x = sparse(test_x)
        test_y = np.full((6, ), np.nan)
        # The following was computed manually, too
        exp_probs = np.array([
            [0.47368421052632, 0.31578947368421, 0.21052631578947],
            [0.39130434782609, 0.26086956521739, 0.34782608695652],
            [0.24324324324324, 0.32432432432432, 0.43243243243243],
            [0.31578947368421, 0.47368421052632, 0.21052631578947],
            [0.26086956521739, 0.39130434782609, 0.34782608695652],
            [0.15000000000000, 0.45000000000000, 0.40000000000000]
        ])

        # Test the faster algorithm for Table (numpy matrices)
        test_data = Table.from_numpy(domain, test_x, test_y)
        probs = model(test_data, ret=model.Probs)
        np.testing.assert_almost_equal(exp_probs, probs)
        values = model(test_data)
        np.testing.assert_equal(values, np.argmax(exp_probs, axis=1))
        values, probs = model(test_data, ret=model.ValueProbs)
        np.testing.assert_almost_equal(exp_probs, probs)
        np.testing.assert_equal(values, np.argmax(exp_probs, axis=1))

        # Test the slower algorithm for non-Table data (iteration in Python)
        test_data = NotATable.from_numpy(domain, test_x, test_y)
        probs = model(test_data, ret=model.Probs)
        np.testing.assert_almost_equal(exp_probs, probs)
        values = model(test_data)
        np.testing.assert_equal(values, np.argmax(exp_probs, axis=1))
        values, probs = model(test_data, ret=model.ValueProbs)
        np.testing.assert_almost_equal(exp_probs, probs)
        np.testing.assert_equal(values, np.argmax(exp_probs, axis=1))

        # Test prediction directly on numpy
        probs = model(test_x, ret=model.Probs)
        np.testing.assert_almost_equal(exp_probs, probs)
        values = model(test_x)
        np.testing.assert_equal(values, np.argmax(exp_probs, axis=1))
        values, probs = model(test_x, ret=model.ValueProbs)
        np.testing.assert_almost_equal(exp_probs, probs)
        np.testing.assert_equal(values, np.argmax(exp_probs, axis=1))

        # Test prediction on instances
        for inst, exp_prob in zip(test_data, exp_probs):
            np.testing.assert_almost_equal(
                model(inst, ret=model.Probs),
                exp_prob)
            self.assertEqual(model(inst), np.argmax(exp_prob))
            value, prob = model(inst, ret=model.ValueProbs)
            np.testing.assert_almost_equal(prob, exp_prob)
            self.assertEqual(value, np.argmax(exp_prob))

        # Test prediction by directly calling predict. This is needed to test
        # csc_matrix, but doesn't hurt others
        if sparse is sp.csc_matrix:
            test_x = sparse(test_x)
        values, probs = model.predict(test_x)
        np.testing.assert_almost_equal(exp_probs, probs)
        np.testing.assert_equal(values, np.argmax(exp_probs, axis=1))

    def _test_predictions_with_absent_class(self, sparse):
        """Empty classes should not affect predictions"""
        x = np.array([
            [1, 0, 0],
            [0, np.nan, 0],
            [0, 1, 0],
            [0, 0, 0],
            [1, 2, 0],
            [1, 1, 0],
            [1, 2, 0],
            [0, 1, 0]])
        if sparse is not None:
            x = sparse(x)

        y = np.array([0, 0, 0, 2, 2, 2, 3, 3])
        domain = Domain(
            [DiscreteVariable("a", values="ab"),
             DiscreteVariable("b", values="abc"),
             DiscreteVariable("c", values="a")],
            DiscreteVariable("y", values="abcd"))
        data = Table.from_numpy(domain, x, y)

        model = self.learner(data)
        np.testing.assert_almost_equal(
            model.class_prob,
            [4/11, 0, 4/11, 3/11]
        )
        np.testing.assert_almost_equal(
            np.exp(model.log_cont_prob[0]) * model.class_prob[:, None],
            [[3/7, 2/7], [0, 0], [2/7, 3/7], [2/7, 2/7]])
        np.testing.assert_almost_equal(
            np.exp(model.log_cont_prob[1]) * model.class_prob[:, None],
            [[2/5, 1/3, 1/5], [0, 0, 0], [2/5, 1/3, 2/5], [1/5, 1/3, 2/5]])
        np.testing.assert_almost_equal(
            np.exp(model.log_cont_prob[2]) * model.class_prob[:, None],
            [[4/11], [0], [4/11], [3/11]])

        test_x = np.array([[a, b, 0] for a in [0, 1] for b in [0, 1, 2]])
        # Classifiers reject csc matrices in the base class
        # Naive bayesian classifier supports them if predict_storage is
        # called directly, which we do below
        if sparse is not None and sparse is not sp.csc_matrix:
            test_x = sparse(test_x)
        test_y = np.full((6, ), np.nan)
        # The following was computed manually, too
        exp_probs = np.array([
            [0.47368421052632, 0, 0.31578947368421, 0.21052631578947],
            [0.39130434782609, 0, 0.26086956521739, 0.34782608695652],
            [0.24324324324324, 0, 0.32432432432432, 0.43243243243243],
            [0.31578947368421, 0, 0.47368421052632, 0.21052631578947],
            [0.26086956521739, 0, 0.39130434782609, 0.34782608695652],
            [0.15000000000000, 0, 0.45000000000000, 0.40000000000000]
        ])

        # Test the faster algorithm for Table (numpy matrices)
        test_data = Table.from_numpy(domain, test_x, test_y)
        probs = model(test_data, ret=model.Probs)
        np.testing.assert_almost_equal(exp_probs, probs)
        values = model(test_data)
        np.testing.assert_equal(values, np.argmax(exp_probs, axis=1))
        values, probs = model(test_data, ret=model.ValueProbs)
        np.testing.assert_almost_equal(exp_probs, probs)
        np.testing.assert_equal(values, np.argmax(exp_probs, axis=1))

        # Test the slower algorithm for non-Table data (iteration in Python)
        test_data = NotATable.from_numpy(domain, test_x, test_y)
        probs = model(test_data, ret=model.Probs)
        np.testing.assert_almost_equal(exp_probs, probs)
        values = model(test_data)
        np.testing.assert_equal(values, np.argmax(exp_probs, axis=1))
        values, probs = model(test_data, ret=model.ValueProbs)
        np.testing.assert_almost_equal(exp_probs, probs)
        np.testing.assert_equal(values, np.argmax(exp_probs, axis=1))

        # Test prediction directly on numpy
        probs = model(test_x, ret=model.Probs)
        np.testing.assert_almost_equal(exp_probs, probs)
        values = model(test_x)
        np.testing.assert_equal(values, np.argmax(exp_probs, axis=1))
        values, probs = model(test_x, ret=model.ValueProbs)
        np.testing.assert_almost_equal(exp_probs, probs)
        np.testing.assert_equal(values, np.argmax(exp_probs, axis=1))

        # Test prediction on instances
        for inst, exp_prob in zip(test_data, exp_probs):
            np.testing.assert_almost_equal(
                model(inst, ret=model.Probs),
                exp_prob)
            self.assertEqual(model(inst), np.argmax(exp_prob))
            value, prob = model(inst, ret=model.ValueProbs)
            np.testing.assert_almost_equal(prob, exp_prob)
            self.assertEqual(value, np.argmax(exp_prob))

        # Test prediction by directly calling predict. This is needed to test
        # csc_matrix, but doesn't hurt others
        if sparse is sp.csc_matrix:
            test_x = sparse(test_x)
        values, probs = model.predict(test_x)
        np.testing.assert_almost_equal(exp_probs, probs)
        np.testing.assert_equal(values, np.argmax(exp_probs, axis=1))

    def test_no_attributes(self):
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2])
        domain = Domain([], DiscreteVariable("y", values="abc"))
        data = Table.from_numpy(domain, np.zeros((len(y), 0)), y.T)
        model = self.learner(data)
        np.testing.assert_almost_equal(
            model.predict_storage(np.zeros((5, 0)))[1],
            [[4/11, 4/11, 3/11]] * 5
        )

    def test_no_targets(self):
        x = np.array([[0], [1], [2]])
        y = np.full(3, np.nan)
        domain = Domain([DiscreteVariable("x", values="abc")],
                        DiscreteVariable("y", values="abc"))
        data = Table.from_numpy(domain, x, y)
        self.assertRaises(ValueError, self.learner, data)


if __name__ == "__main__":
    unittest.main()
