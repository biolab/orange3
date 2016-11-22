# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import numpy as np

from Orange.data import DiscreteVariable, Domain
from Orange.data import Table
from Orange.classification import LogisticRegressionLearner, SklTreeLearner, NaiveBayesLearner,\
                                  MajorityLearner
from Orange.evaluation import AUC, CA, Results, Recall, \
    Precision, TestOnTrainingData, scoring, LogLoss, F1, CrossValidation
from Orange.preprocess import discretize, Discretize


class TestRecall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Table('iris')

    def test_recall(self):
        learner = LogisticRegressionLearner(preprocessors=[])
        results = TestOnTrainingData(self.data, [learner])
        self.assertAlmostEqual(Recall(results)[0], 0.960, 3)


class TestPrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Table('iris')

    def test_precision(self):
        learner = LogisticRegressionLearner(preprocessors=[])
        results = TestOnTrainingData(self.data, [learner])
        self.assertAlmostEqual(Precision(results)[0], 0.962, 3)


class TestCA(unittest.TestCase):
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
        x = np.random.randint(2, size=(100, 5))
        col = np.random.randint(5)
        y = x[:, col].copy().reshape(100, 1)
        t = Table(x, y)
        t = Discretize(
            method=discretize.EqualWidth(n=3))(t)
        nb = NaiveBayesLearner()
        res = TestOnTrainingData(t, [nb])
        np.testing.assert_almost_equal(CA(res), [1])

        t.Y[-20:] = 1 - t.Y[-20:]
        res = TestOnTrainingData(t, [nb])
        self.assertGreaterEqual(CA(res)[0], 0.75)
        self.assertLess(CA(res)[0], 1)


class TestAUC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')

    def test_tree(self):
        tree = SklTreeLearner()
        res = CrossValidation(self.iris, [tree], k=2)
        self.assertGreater(AUC(res)[0], 0.8)
        self.assertLess(AUC(res)[0], 1.)

    def test_constant_prob(self):
        maj = MajorityLearner()
        res = TestOnTrainingData(self.iris, [maj])
        self.assertEqual(AUC(res)[0], 0.5)

    def test_multiclass_auc_multi_learners(self):
        learners = [LogisticRegressionLearner(),
                    MajorityLearner()]
        res = CrossValidation(self.iris, learners, k=10)
        self.assertGreater(AUC(res)[0], 0.6)
        self.assertLess(AUC(res)[1], 0.6)
        self.assertGreater(AUC(res)[1], 0.4)

    def test_auc_on_multiclass_data_returns_1d_array(self):
        titanic = Table('titanic')[:100]
        lenses = Table('lenses')[:100]
        majority = MajorityLearner()

        results = TestOnTrainingData(lenses, [majority])
        auc = AUC(results)
        self.assertEqual(auc.ndim, 1)

        results = TestOnTrainingData(titanic, [majority])
        auc = AUC(results)
        self.assertEqual(auc.ndim, 1)

    def test_auc_scores(self):
        actual = np.array([0., 0., 0., 1., 1., 1.])

        for predicted, auc in (([1., 1., 1., 0., 0., 0.], 0.),      # All wrong
                               ([0., 0., 0., 0., 0., 0.], 0.5),     # All with same probability
                               ([0., 0., 0., 1., 1., 1.], 1.),      # All correct
                               ([0., 0., 0., 1., 1., 0.], 5 / 6),   # One wrong
                               ([1., 1., 0., 1., 1., 1.], 4 / 6),   # Two wrong
                               ([1., 1., 0., 1., 1., 0.], 3 / 6)):  # Three wrong
            self.assertAlmostEqual(self.compute_auc(actual, predicted), auc)

    def compute_auc(self, actual, predicted):
        predicted = np.array(predicted).reshape(1, -1)
        results = Results(
            nmethods=1, domain=Domain([], [DiscreteVariable(values='01')]),
            actual=actual, predicted=predicted)
        return AUC(results)[0]


class TestComputeCD(unittest.TestCase):
    def test_compute_CD(self):
        avranks = [1.9, 3.2, 2.8, 3.3]
        cd = scoring.compute_CD(avranks, 30)
        np.testing.assert_almost_equal(cd, 0.856344)

        cd = scoring.compute_CD(avranks, 30, test="bonferroni-dunn")
        np.testing.assert_almost_equal(cd, 0.798)


class TestLogLoss(unittest.TestCase):
    def test_log_loss(self):
        data = Table('iris')
        majority = MajorityLearner()
        results = TestOnTrainingData(data, [majority])
        ll = LogLoss(results)
        self.assertAlmostEqual(ll[0], - np.log(1 / 3))

    def _log_loss(self, act, prob):
        ll = np.dot(np.log(prob[:, 0]), act[:, 0]) + \
             np.dot(np.log(prob[:, 1]), act[:, 1])
        return - ll / len(act)

    def test_log_loss_calc(self):
        data = Table('titanic')
        learner = LogisticRegressionLearner()
        results = TestOnTrainingData(data, [learner])

        actual = np.copy(results.actual)
        actual = actual.reshape(actual.shape[0], 1)
        actual = np.hstack((1 - actual, actual))
        probab = results.probabilities[0]

        ll_calc = self._log_loss(actual, probab)
        ll_orange = LogLoss(results)
        self.assertAlmostEqual(ll_calc, ll_orange[0])


class TestF1(unittest.TestCase):
    def test_F1_multiclass(self):
        results = Results(
            domain=Domain([], DiscreteVariable(name="y", values="01234")),
            actual=[0, 4, 4, 1, 2, 0, 1, 2, 3, 2])
        results.predicted = np.array([[0, 1, 4, 1, 1, 0, 0, 2, 3, 1],
                                      [0, 4, 4, 1, 2, 0, 1, 2, 3, 2]])
        res = F1(results)
        self.assertAlmostEqual(res[0], 0.61)
        self.assertEqual(res[1], 1.)

    def test_F1_target(self):
        results = Results(
            domain=Domain([], DiscreteVariable(name="y", values="01234")),
            actual=[0, 4, 4, 1, 2, 0, 1, 2, 3, 2])
        results.predicted = np.array([[0, 1, 4, 1, 1, 0, 0, 2, 3, 1],
                                      [0, 4, 4, 1, 2, 0, 1, 2, 3, 2]])

        for target, prob in ((0, 4 / 5),
                             (1, 1 / 3),
                             (2, 1 / 2),
                             (3, 1.),
                             (4, 2 / 3)):
            res = F1(results, target=target)
            self.assertEqual(res[0], prob)
            self.assertEqual(res[1], 1.)

    def test_F1_binary(self):
        results = Results(
            domain=Domain([], DiscreteVariable(name="y", values="01")),
            actual=[0, 1, 1, 1, 0, 0, 1, 0, 0, 1])
        results.predicted = np.array([[0, 1, 1, 1, 0, 0, 1, 0, 0, 1],
                                      [0, 1, 1, 1, 0, 0, 1, 1, 1, 1]])
        res = F1(results)
        self.assertEqual(res[0], 1.)
        self.assertAlmostEqual(res[1], 5 / 6)
        res_target = F1(results, target=1)
        self.assertEqual(res[0], res_target[0])
        self.assertEqual(res[1], res_target[1])
        res_target = F1(results, target=0)
        self.assertEqual(res_target[0], 1.)
        self.assertAlmostEqual(res_target[1], 3 / 4)
