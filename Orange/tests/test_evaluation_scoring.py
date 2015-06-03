import unittest
from Orange.data import DiscreteVariable, Domain

import numpy as np

import Orange
from Orange.evaluation import AUC, CA, Results, F1
from Orange.preprocess import discretize


class Scoring_CA_Test(unittest.TestCase):
    def test_init(self):
        res = Results(nmethods=2, nrows=100)
        res.actual[:50] = 0
        res.actual[50:] = 1
        res.predicted = np.vstack((res.actual, res.actual))
        np.testing.assert_almost_equal(CA(res), [1, 1])

        res.predicted[0][0] = 1
        np.testing.assert_almost_equal(CA(res), [0.99, 1])

        res.predicted[1] = 1 - res.predicted[1]
        np.testing.assert_almost_equal(CA(res), [0.99, 0])

    def test_call(self):
        res = Results(nmethods=2, nrows=100)
        res.actual[:50] = 0
        res.actual[50:] = 1
        res.predicted = np.vstack((res.actual, res.actual))
        ca = CA()
        np.testing.assert_almost_equal(ca(res), [1, 1])

        res.predicted[0][0] = 1
        np.testing.assert_almost_equal(ca(res), [0.99, 1])

        res.predicted[1] = 1 - res.predicted[1]
        np.testing.assert_almost_equal(ca(res), [0.99, 0])

    def test_bayes(self):
        x = np.random.random_integers(1, 3, (100, 5))
        col = np.random.randint(5)
        y = x[:, col].copy().reshape(100, 1)
        t = Orange.data.Table(x, y)
        t = Orange.preprocess.Discretize(
            method=discretize.EqualWidth(n=3))(t)
        nb = Orange.classification.NaiveBayesLearner()
        res = Orange.evaluation.TestOnTrainingData(t, [nb])
        np.testing.assert_almost_equal(CA(res), [1])

        t.Y[-20:] = 4 - t.Y[-20:]
        res = Orange.evaluation.TestOnTrainingData(t, [nb])
        self.assertGreaterEqual(CA(res)[0], 0.75)
        self.assertLess(CA(res)[0], 1)


class Scoring_AUC_Test(unittest.TestCase):
    def test_tree(self):
        data = Orange.data.Table('iris')
        tree = Orange.classification.TreeLearner()
        res = Orange.evaluation.CrossValidation(data, [tree], k=2)
        self.assertTrue(0.8 < AUC(res)[0] < 1.)

    def test_constant_prob(self):
        data = Orange.data.Table('iris')
        maj = Orange.classification.MajorityLearner()
        res = Orange.evaluation.TestOnTrainingData(data, [maj])
        self.assertEqual(AUC(res)[0], 0.5)

    def test_multiclass_auc_multi_learners(self):
        data = Orange.data.Table('iris')
        learners = [Orange.classification.LogisticRegressionLearner(), Orange.classification.MajorityLearner()]
        res = Orange.evaluation.testing.CrossValidation(data, learners, k=10)
        self.assertTrue(AUC(res)[0] > 0.6 > AUC(res)[1] > 0.4)

    def test_auc_on_multiclass_data_returns_1d_array(self):
        titanic = Orange.data.Table('titanic')[:100]
        lenses = Orange.data.Table('lenses')[:100]
        majority = Orange.classification.MajorityLearner()

        results = Orange.evaluation.TestOnTrainingData(lenses, [majority])
        auc = Orange.evaluation.AUC(results)
        self.assertEqual(auc.ndim, 1)

        results = Orange.evaluation.TestOnTrainingData(titanic, [majority])
        auc = Orange.evaluation.AUC(results)
        self.assertEqual(auc.ndim, 1)

    def test_auc_scores(self):
        actual = np.array([0., 0., 0., 1., 1., 1.])

        # All wrong
        self.assertAlmostEqual(
            self.compute_auc(actual, [1., 1., 1., 0., 0., 0.]), 0.)
        # All with same probability
        self.assertAlmostEqual(
            self.compute_auc(actual, [0., 0., 0., 0., 0., 0.]), 0.5)
        # All correct
        self.assertAlmostEqual(
            self.compute_auc(actual, [0., 0., 0., 1., 1., 1.]), 1.)

        # One wrong
        self.assertAlmostEqual(
            self.compute_auc(actual, [0., 0., 0., 1., 1., 0.]), 5 / 6)
        # Two wrong
        self.assertAlmostEqual(
            self.compute_auc(actual, [1., 1., 0., 1., 1., 1.]), 4 / 6)
        # Three wrong
        self.assertAlmostEqual(
            self.compute_auc(actual, [1., 1., 0., 1., 1., 0.]), 3 / 6)

    def compute_auc(self, actual, predicted):
        predicted = np.array(predicted).reshape(1, -1)
        results = Orange.evaluation.Results(
            nmethods=1, domain=Domain([], [DiscreteVariable(values='01')]),
            actual=actual, predicted=predicted)
        return AUC(results)[0]


class Scoring_CD_Test(unittest.TestCase):
    def test_cd_score(self):
        avranks = [1.9, 3.2, 2.8, 3.3]
        cd = Orange.evaluation.scoring.compute_CD(avranks, 30)
        np.testing.assert_almost_equal(cd, 0.856344)

        cd = Orange.evaluation.scoring.compute_CD(avranks, 30, test="bonferroni-dunn")
        np.testing.assert_almost_equal(cd, 0.798)


class Scoring_LogLoss_Test(unittest.TestCase):
    def test_log_loss(self):
        data = Orange.data.Table('iris')
        majority = Orange.classification.MajorityLearner()
        results = Orange.evaluation.TestOnTrainingData(data, [majority])
        ll = Orange.evaluation.LogLoss(results)
        self.assertAlmostEqual(ll[0], - np.log(1 / 3))


class Scoring_F1_Test(unittest.TestCase):
    def test_F1_multiclass(self):
        results = Orange.evaluation.Results(
            domain=Domain([], DiscreteVariable(name="y", values="01234")),
            actual=[0, 4, 4, 1, 2, 0, 1, 2, 3, 2])
        results.predicted = np.array([[0, 1, 4, 1, 1, 0, 0, 2, 3, 1],
                                      [0, 4, 4, 1, 2, 0, 1, 2, 3, 2]])
        res = Orange.evaluation.F1(results)
        self.assertAlmostEqual(res[0], 0.61)
        self.assertEqual(res[1], 1.)

    def test_F1_target(self):
        results = Orange.evaluation.Results(
            domain=Domain([], DiscreteVariable(name="y", values="01234")),
            actual=[0, 4, 4, 1, 2, 0, 1, 2, 3, 2])
        results.predicted = np.array([[0, 1, 4, 1, 1, 0, 0, 2, 3, 1],
                                      [0, 4, 4, 1, 2, 0, 1, 2, 3, 2]])
        res = Orange.evaluation.F1(results, target=0)
        self.assertAlmostEqual(res[0], 4 / 5)
        self.assertEqual(res[1], 1.)
        res = Orange.evaluation.F1(results, target=1)
        self.assertEqual(res[0], 2 / 6)
        self.assertEqual(res[1], 1.)
        res = Orange.evaluation.F1(results, target=2)
        self.assertEqual(res[0], 2 / 4)
        self.assertEqual(res[1], 1.)
        res = Orange.evaluation.F1(results, target=3)
        self.assertEqual(res[0], 1.)
        self.assertEqual(res[1], 1.)
        res = Orange.evaluation.F1(results, target=4)
        self.assertEqual(res[0], 2 / 3)
        self.assertEqual(res[1], 1.)

    def test_F1_binary(self):
        results = Orange.evaluation.Results(
            domain=Domain([], DiscreteVariable(name="y", values="01")),
            actual=[0, 1, 1, 1, 0, 0, 1, 0, 0, 1])
        results.predicted = np.array([[0, 1, 1, 1, 0, 0, 1, 0, 0, 1],
                                      [0, 1, 1, 1, 0, 0, 1, 1, 1, 1]])
        res = Orange.evaluation.F1(results)
        self.assertEqual(res[0], 1.)
        self.assertAlmostEqual(res[1], 10 / 12)
        res_target = Orange.evaluation.F1(results, target=1)
        self.assertEqual(res[0], res_target[0])
        self.assertEqual(res[1], res_target[1])
        res_target = Orange.evaluation.F1(results, target=0)
        self.assertEqual(res_target[0], 1.)
        self.assertAlmostEqual(res_target[1], 3 / 4)
