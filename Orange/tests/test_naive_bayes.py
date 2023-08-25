# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
from unittest.mock import Mock

import numpy as np
import scipy.sparse as sp

from Orange.classification import NaiveBayesLearner
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.evaluation import CrossValidation, CA
from Orange.tests import test_filename


# This class is used to force predict_storage to fall back to the slower
# procedure instead of calling `predict`
class NotATable(Table):  # pylint: disable=too-many-ancestors,abstract-method
    @classmethod
    def from_file(cls, *args, **kwargs):
        table = super().from_file(*args, **kwargs)
        return cls(table)


def assert_predictions_equal(data, model, exp_probs):
    exp_vals = np.argmax(np.atleast_2d(exp_probs), axis=1)
    np.testing.assert_almost_equal(model(data, ret=model.Probs), exp_probs)
    np.testing.assert_equal(model(data), exp_vals)
    values, probs = model(data, ret=model.ValueProbs)
    np.testing.assert_almost_equal(probs, exp_probs)
    np.testing.assert_equal(values, exp_vals)


def assert_model_equal(model, results):
    np.testing.assert_almost_equal(
        model.class_prob,
        results[0])
    np.testing.assert_almost_equal(
        np.exp(model.log_cont_prob[0]) * model.class_prob[:, None],
        results[1])
    np.testing.assert_almost_equal(
        np.exp(model.log_cont_prob[1]) * model.class_prob[:, None],
        results[2])
    np.testing.assert_almost_equal(
        np.exp(model.log_cont_prob[2]) * model.class_prob[:, None],
        results[3])


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
        self._test_predictions(sparse=None, absent_class=True)
        self._test_predict_missing_attributes(sparse=None)

    def test_predictions_csr_matrix(self):
        self._test_predictions(sparse=sp.csr_matrix)
        self._test_predictions(sparse=sp.csr_matrix, absent_class=True)
        self._test_predict_missing_attributes(sparse=sp.csr_matrix)

    def test_predictions_csc_matrix(self):
        self._test_predictions(sparse=sp.csc_matrix)
        self._test_predictions(sparse=sp.csc_matrix, absent_class=True)
        self._test_predict_missing_attributes(sparse=sp.csc_matrix)

    @staticmethod
    def _create_prediction_data(sparse, absent_class=False):
        """ The following was computed manually """
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
        class_var = DiscreteVariable("y", values="abc")
        results = [
            [4/11, 4/11, 3/11],
            [[3/7, 2/7], [2/7, 3/7], [2/7, 2/7]],
            [[2/5, 1/3, 1/5], [2/5, 1/3, 2/5], [1/5, 1/3, 2/5]],
            [[4/11], [4/11], [3/11]]
        ]

        test_x = np.array([[a, b, 0] for a in [0, 1] for b in [0, 1, 2]])
        # Classifiers reject csc matrices in the base class
        # Naive bayesian classifier supports them if predict_storage is
        # called directly, which we do below
        if sparse is not None and sparse is not sp.csc_matrix:
            test_x = sparse(test_x)
        test_y = np.full((6, ), np.nan)

        exp_probs = np.array([
            [0.47368421052632, 0.31578947368421, 0.21052631578947],
            [0.39130434782609, 0.26086956521739, 0.34782608695652],
            [0.24324324324324, 0.32432432432432, 0.43243243243243],
            [0.31578947368421, 0.47368421052632, 0.21052631578947],
            [0.26086956521739, 0.39130434782609, 0.34782608695652],
            [0.15000000000000, 0.45000000000000, 0.40000000000000]
        ])

        if absent_class:
            y = np.array([0, 0, 0, 2, 2, 2, 3, 3])
            class_var = DiscreteVariable("y", values="abcd")
            for i, row in enumerate(results):
                row.insert(1, i and [0]*len(row[0]))
            exp_probs = np.insert(exp_probs, 1, 0, axis=1)

        domain = Domain(
            [DiscreteVariable("a", values="ab"),
             DiscreteVariable("b", values="abc"),
             DiscreteVariable("c", values="a")],
            class_var)
        data = Table.from_numpy(domain, x, y)

        return data, domain, results, test_x, test_y, exp_probs

    def _test_predictions(self, sparse, absent_class=False):
        (data, domain, results,
         test_x, test_y, exp_probs) = self._create_prediction_data(sparse, absent_class)

        model = self.learner(data)
        assert_model_equal(model, results)

        # Test the faster algorithm for Table (numpy matrices)
        test_data = Table.from_numpy(domain, test_x, test_y)
        assert_predictions_equal(test_data, model, exp_probs)

        # Test the slower algorithm for non-Table data (iteration in Python)
        test_data = NotATable.from_numpy(domain, test_x, test_y)
        assert_predictions_equal(test_data, model, exp_probs)

        # Test prediction directly on numpy
        assert_predictions_equal(test_x, model, exp_probs)

        # Test prediction on instances
        for inst, exp_prob in zip(test_data, exp_probs):
            assert_predictions_equal(inst, model, exp_prob)

        # Test prediction by directly calling predict. This is needed to test
        # csc_matrix, but doesn't hurt others
        if sparse is sp.csc_matrix:
            test_x = sparse(test_x)
        values, probs = model.predict(test_x)
        np.testing.assert_almost_equal(probs, exp_probs)
        np.testing.assert_equal(values, np.argmax(exp_probs, axis=1))

    @staticmethod
    def _create_missing_attributes(sparse):
        x = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 2, 0],
            [1, 2, np.nan]])
        if sparse is not None:
            x = sparse(x)
        y = np.array([1, 0, 0, 0, 1, 1, 1])

        test_x = np.array([[np.nan, np.nan, np.nan],
                           [np.nan, 0, np.nan],
                           [0, np.nan, np.nan]])
        if sparse is not None and sparse is not sp.csc_matrix:
            test_x = sparse(test_x)
        exp_probs = np.array([[(3 + 1) / (7 + 2), (4 + 1) / (7 + 2)],
                              [(1 + 1) / (2 + 2), (1 + 1) / (2 + 2)],
                              [(3 + 1) / (3 + 2), (0 + 1) / (3 + 2)]])

        domain = Domain(
            [DiscreteVariable("a", values="ab"),
             DiscreteVariable("b", values="abc"),
             DiscreteVariable("c", values="a")],
            DiscreteVariable("y", values="AB"))
        return Table.from_numpy(domain, x, y), test_x, exp_probs

    def _test_predict_missing_attributes(self, sparse):
        data, test_x, exp_probs = self._create_missing_attributes(sparse)
        model = self.learner(data)
        probs = model(test_x, ret=model.Probs)
        np.testing.assert_almost_equal(probs, exp_probs)

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
