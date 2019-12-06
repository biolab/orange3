# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import numpy as np
from Orange.data import Table
from Orange.classification import SklTreeLearner
from Orange.regression import SklTreeRegressionLearner
from Orange.ensembles import (
    SklAdaBoostClassificationLearner,
    SklAdaBoostRegressionLearner,
)
from Orange.evaluation import CrossValidation, CA, RMSE


class TestSklAdaBoostLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table("iris")
        cls.housing = Table("housing")

    def test_adaboost(self):
        learn = SklAdaBoostClassificationLearner()
        cv = CrossValidation(k=3)
        results = cv(self.iris, [learn])
        ca = CA(results)
        self.assertGreater(ca, 0.9)
        self.assertLess(ca, 0.99)

    def test_adaboost_base_estimator(self):
        np.random.seed(0)
        stump_estimator = SklTreeLearner(max_depth=1)
        tree_estimator = SklTreeLearner()
        stump = SklAdaBoostClassificationLearner(
            base_estimator=stump_estimator, n_estimators=5)
        tree = SklAdaBoostClassificationLearner(
            base_estimator=tree_estimator, n_estimators=5)
        cv = CrossValidation(k=4)
        results = cv(self.iris, [stump, tree])
        ca = CA(results)
        self.assertLessEqual(ca[0], ca[1])

    def test_predict_single_instance(self):
        learn = SklAdaBoostClassificationLearner()
        m = learn(self.iris)
        ins = self.iris[0]
        m(ins)
        _, _ = m(ins, m.ValueProbs)

    def test_predict_table(self):
        learn = SklAdaBoostClassificationLearner()
        m = learn(self.iris)
        m(self.iris)
        _, _ = m(self.iris, m.ValueProbs)

    def test_predict_numpy(self):
        learn = SklAdaBoostClassificationLearner()
        m = learn(self.iris)
        _, _ = m(self.iris.X, m.ValueProbs)

    def test_adaboost_adequacy(self):
        learner = SklAdaBoostClassificationLearner()
        self.assertRaises(ValueError, learner, self.housing)

    def test_adaboost_reg(self):
        learn = SklAdaBoostRegressionLearner()
        cv = CrossValidation(k=3)
        results = cv(self.housing, [learn])
        _ = RMSE(results)

    def test_adaboost_reg_base_estimator(self):
        np.random.seed(0)
        stump_estimator = SklTreeRegressionLearner(max_depth=1)
        tree_estimator = SklTreeRegressionLearner()
        stump = SklAdaBoostRegressionLearner(base_estimator=stump_estimator)
        tree = SklAdaBoostRegressionLearner(base_estimator=tree_estimator)
        cv = CrossValidation(k=3)
        results = cv(self.housing, [stump, tree])
        rmse = RMSE(results)
        self.assertGreaterEqual(rmse[0], rmse[1])

    def test_predict_single_instance_reg(self):
        learn = SklAdaBoostRegressionLearner()
        m = learn(self.housing)
        ins = self.housing[0]
        pred = m(ins)
        self.assertGreaterEqual(pred, 0)

    def test_predict_table_reg(self):
        learn = SklAdaBoostRegressionLearner()
        m = learn(self.housing)
        pred = m(self.housing)
        self.assertEqual(len(self.housing), len(pred))
        self.assertGreater(all(pred), 0)

    def test_predict_numpy_reg(self):
        learn = SklAdaBoostRegressionLearner()
        m = learn(self.housing)
        pred = m(self.housing.X)
        self.assertEqual(len(self.housing), len(pred))
        self.assertGreater(all(pred), 0)

    def test_adaboost_adequacy_reg(self):
        learner = SklAdaBoostRegressionLearner()
        self.assertRaises(ValueError, learner, self.iris)
