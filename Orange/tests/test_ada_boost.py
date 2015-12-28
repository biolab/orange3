import unittest
import numpy as np
from Orange.data import Table
from Orange.classification import AdaBoostLearner, TreeLearner
from Orange.regression import AdaBoostRegressionLearner, TreeRegressionLearner
from Orange.evaluation import CrossValidation, CA, RMSE


class AdaBoostTest(unittest.TestCase):
    def test_adaboost(self):
        table = Table("iris")
        learn = AdaBoostLearner()
        results = CrossValidation(table, [learn], k=10)
        ca = CA(results)
        self.assertGreater(ca, 0.9)
        self.assertLess(ca, 0.99)

    def test_adaboost_base_estimator(self):
        np.random.seed(0)
        table = Table("iris")
        stump_estimator = TreeLearner(max_depth=1)
        tree_estimator = TreeLearner()
        stump = AdaBoostLearner(base_estimator=stump_estimator)
        tree = AdaBoostLearner(base_estimator=tree_estimator)
        results = CrossValidation(table, [stump, tree], k=10)
        ca = CA(results)
        self.assertTrue(ca[0] < ca[1])

    def test_predict_single_instance(self):
        table = Table('iris')
        learn = AdaBoostLearner()
        m = learn(table)
        for ins in table:
            m(ins)
            _, _ = m(ins, m.ValueProbs)

    def test_predict_table(self):
        table = Table('iris')
        learn = AdaBoostLearner()
        m = learn(table)
        m(table)
        _, _ = m(table, m.ValueProbs)

    def test_predict_numpy(self):
        table = Table('iris')
        learn = AdaBoostLearner()
        m = learn(table)
        _, _ = m(table.X, m.ValueProbs)

    def test_adaboost_reg(self):
        table = Table("housing")
        learn = AdaBoostRegressionLearner()
        results = CrossValidation(table, [learn], k=10)
        _ = RMSE(results)

    def test_adaboost_reg_base_estimator(self):
        np.random.seed(0)
        table = Table("housing")
        stump_estimator = TreeRegressionLearner(max_depth=1)
        tree_estimator = TreeRegressionLearner()
        stump = AdaBoostRegressionLearner(base_estimator=stump_estimator)
        tree = AdaBoostRegressionLearner(base_estimator=tree_estimator)
        results = CrossValidation(table, [stump, tree], k=10)
        rmse = RMSE(results)
        self.assertTrue(rmse[0] >= rmse[1])

    def test_predict_single_instance_reg(self):
        table = Table('housing')
        learn = AdaBoostRegressionLearner()
        m = learn(table)
        for ins in table:
            pred = m(ins)
            self.assertTrue(pred > 0)

    def test_predict_table_reg(self):
        table = Table('housing')
        learn = AdaBoostRegressionLearner()
        m = learn(table)
        pred = m(table)
        self.assertEqual(len(table), len(pred))
        self.assertTrue(all(pred) > 0)

    def test_predict_numpy_reg(self):
        table = Table('housing')
        learn = AdaBoostRegressionLearner()
        m = learn(table)
        pred = m(table.X)
        self.assertEqual(len(table), len(pred))
        self.assertTrue(all(pred) > 0)
