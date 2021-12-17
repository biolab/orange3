# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import numpy as np

from Orange.data import DiscreteVariable, ContinuousVariable, Domain
from Orange.data import Table
from Orange.classification import LogisticRegressionLearner, SklTreeLearner, NaiveBayesLearner,\
                                  MajorityLearner
from Orange.evaluation import AUC, CA, Results, Recall, \
    Precision, TestOnTrainingData, scoring, LogLoss, F1, CrossValidation
from Orange.evaluation.scoring import Specificity
from Orange.preprocess import discretize, Discretize
from Orange.tests import test_filename


class TestScoreMetaType(unittest.TestCase):
    class BaseScore(metaclass=scoring.ScoreMetaType):
        pass

    class Score1(BaseScore, abstract=True):
        class_types = (DiscreteVariable,)

    class Score2(Score1):
        pass

    class Score3(Score2):
        name = "foo"

    class Score4(Score2):
        pass

    class Score5(BaseScore):
        class_types = (DiscreteVariable, ContinuousVariable)

    def test_registry(self):
        """All non-abstract classes appear in the registry"""
        self.assertEqual(
            self.BaseScore.registry,
            {"Score2": self.Score2, "Score3": self.Score3,
             "Score4": self.Score4, "Score5": self.Score5})

    def test_names(self):
        """Attribute `name` defaults to class and is not inherited"""
        self.assertEqual(self.Score2.name, "Score2")
        self.assertEqual(self.Score3.name, "foo")
        self.assertEqual(self.Score4.name, "Score4")
        self.assertEqual(self.Score5.name, "Score5")


class TestPrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.score = Precision()

    def test_precision_iris(self):
        learner = LogisticRegressionLearner(preprocessors=[])
        res = TestOnTrainingData()(self.iris, [learner])
        self.assertGreater(self.score(res, average='weighted')[0], 0.95)
        self.assertGreater(self.score(res, target=1)[0], 0.95)
        self.assertGreater(self.score(res, target=1, average=None)[0], 0.95)
        self.assertGreater(self.score(res, target=1, average='weighted')[0], 0.95)
        self.assertGreater(self.score(res, target=0, average=None)[0], 0.99)
        self.assertGreater(self.score(res, target=2, average=None)[0], 0.94)

    def test_precision_multiclass(self):
        results = Results(
            domain=Domain([], DiscreteVariable(name="y", values="01234")),
            actual=[0, 4, 4, 1, 2, 0, 1, 2, 3, 2])
        results.predicted = np.array([[0, 4, 4, 1, 2, 0, 1, 2, 3, 2],
                                      [0, 1, 4, 1, 1, 0, 0, 2, 3, 1]])
        res = self.score(results, average='weighted')
        self.assertEqual(res[0], 1.)
        self.assertAlmostEqual(res[1], 0.78333, 5)

        for target, prob in ((0, 2 / 3),
                             (1, 1 / 4),
                             (2, 1 / 1),
                             (3, 1 / 1),
                             (4, 1 / 1)):
            res = self.score(results, target=target, average=None)
            self.assertEqual(res[0], 1.)
            self.assertEqual(res[1], prob)

    def test_precision_binary(self):
        results = Results(
            domain=Domain([], DiscreteVariable(name="y", values="01")),
            actual=[0, 1, 1, 1, 0, 0, 1, 0, 0, 1])
        results.predicted = np.array([[0, 1, 1, 1, 0, 0, 1, 0, 0, 1],
                                      [0, 1, 1, 1, 0, 0, 1, 1, 1, 0]])
        res = self.score(results)
        self.assertEqual(res[0], 1.)
        self.assertAlmostEqual(res[1], 4 / 6)
        res_target = self.score(results, target=1)
        self.assertEqual(res[0], res_target[0])
        self.assertEqual(res[1], res_target[1])
        res_target = self.score(results, target=0)
        self.assertEqual(res_target[0], 1.)
        self.assertAlmostEqual(res_target[1], 3 / 4)
        res_target = self.score(results, average='macro')
        self.assertEqual(res_target[0], 1.)
        self.assertAlmostEqual(res_target[1], (4 / 6 + 3 / 4) / 2)


class TestRecall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.score = Recall()

    def test_recall_iris(self):
        learner = LogisticRegressionLearner(preprocessors=[])
        res = TestOnTrainingData()(self.iris, [learner])
        self.assertGreater(self.score(res, average='weighted')[0], 0.96)
        self.assertGreater(self.score(res, target=1)[0], 0.9)
        self.assertGreater(self.score(res, target=1, average=None)[0], 0.9)
        self.assertGreater(self.score(res, target=1, average='weighted')[0], 0.9)
        self.assertGreater(self.score(res, target=0, average=None)[0], 0.99)
        self.assertGreater(self.score(res, target=2, average=None)[0], 0.97)

    def test_recall_multiclass(self):
        results = Results(
            domain=Domain([], DiscreteVariable(name="y", values="01234")),
            actual=[0, 4, 4, 1, 2, 0, 1, 2, 3, 2])
        results.predicted = np.array([[0, 4, 4, 1, 2, 0, 1, 2, 3, 2],
                                      [0, 1, 4, 1, 1, 0, 0, 2, 3, 1]])
        res = self.score(results, average='weighted')
        self.assertEqual(res[0], 1.)
        self.assertAlmostEqual(res[1], 0.6)

        for target, prob in ((0, 2 / 2),
                             (1, 1 / 2),
                             (2, 1 / 3),
                             (3, 1 / 1),
                             (4, 1 / 2)):
            res = self.score(results, target=target)
            self.assertEqual(res[0], 1.)
            self.assertEqual(res[1], prob)

    def test_recall_binary(self):
        results = Results(
            domain=Domain([], DiscreteVariable(name="y", values="01")),
            actual=[0, 1, 1, 1, 0, 0, 1, 0, 0, 1])
        results.predicted = np.array([[0, 1, 1, 1, 0, 0, 1, 0, 0, 1],
                                      [0, 1, 1, 1, 0, 0, 1, 1, 1, 0]])
        res = self.score(results)
        self.assertEqual(res[0], 1.)
        self.assertAlmostEqual(res[1], 4 / 5)
        res_target = self.score(results, target=1)
        self.assertEqual(res[0], res_target[0])
        self.assertEqual(res[1], res_target[1])
        res_target = self.score(results, target=0)
        self.assertEqual(res_target[0], 1.)
        self.assertAlmostEqual(res_target[1], 3 / 5)
        res_target = self.score(results, average='macro')
        self.assertEqual(res_target[0], 1.)
        self.assertAlmostEqual(res_target[1], (4 / 5 + 3 / 5) / 2)


class TestF1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.score = F1()

    def test_recall_iris(self):
        learner = LogisticRegressionLearner(preprocessors=[])
        res = TestOnTrainingData()(self.iris, [learner])
        self.assertGreater(self.score(res, average='weighted')[0], 0.95)
        self.assertGreater(self.score(res, target=1)[0], 0.95)
        self.assertGreater(self.score(res, target=1, average=None)[0], 0.95)
        self.assertGreater(self.score(res, target=1, average='weighted')[0], 0.95)
        self.assertGreater(self.score(res, target=0, average=None)[0], 0.99)
        self.assertGreater(self.score(res, target=2, average=None)[0], 0.95)

    def test_F1_multiclass(self):
        results = Results(
            domain=Domain([], DiscreteVariable(name="y", values="01234")),
            actual=[0, 4, 4, 1, 2, 0, 1, 2, 3, 2])
        results.predicted = np.array([[0, 4, 4, 1, 2, 0, 1, 2, 3, 2],
                                      [0, 1, 4, 1, 1, 0, 0, 2, 3, 1]])
        res = self.score(results, average='weighted')
        self.assertEqual(res[0], 1.)
        self.assertAlmostEqual(res[1], 0.61)

        for target, prob in ((0, 4 / 5),
                             (1, 1 / 3),
                             (2, 1 / 2),
                             (3, 1.),
                             (4, 2 / 3)):
            res = self.score(results, target=target)
            self.assertEqual(res[0], 1.)
            self.assertEqual(res[1], prob)

    def test_F1_binary(self):
        results = Results(
            domain=Domain([], DiscreteVariable(name="y", values="01")),
            actual=[0, 1, 1, 1, 0, 0, 1, 0, 0, 1])
        results.predicted = np.array([[0, 1, 1, 1, 0, 0, 1, 0, 0, 1],
                                      [0, 1, 1, 1, 0, 0, 1, 1, 1, 1]])
        res = self.score(results)
        self.assertEqual(res[0], 1.)
        self.assertAlmostEqual(res[1], 5 / 6)
        res_target = self.score(results, target=1)
        self.assertEqual(res[0], res_target[0])
        self.assertEqual(res[1], res_target[1])
        res_target = self.score(results, target=0)
        self.assertEqual(res_target[0], 1.)
        self.assertAlmostEqual(res_target[1], 3 / 4)


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
        y = x[:, col].reshape(100, 1).copy()
        t = Table.from_numpy(None, x, y)
        t = Discretize(
            method=discretize.EqualWidth(n=3))(t)
        nb = NaiveBayesLearner()
        res = TestOnTrainingData()(t, [nb])
        np.testing.assert_almost_equal(CA(res), [1])

        t = Table.from_numpy(None, t.X, t.Y.copy())
        with t.unlocked():
            t.Y[-20:] = 1 - t.Y[-20:]
        res = TestOnTrainingData()(t, [nb])
        self.assertGreaterEqual(CA(res)[0], 0.75)
        self.assertLess(CA(res)[0], 1)


class TestAUC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')

    def test_tree(self):
        tree = SklTreeLearner()
        res = CrossValidation(k=2)(self.iris, [tree])
        self.assertGreater(AUC(res)[0], 0.8)
        self.assertLess(AUC(res)[0], 1.)

    def test_constant_prob(self):
        maj = MajorityLearner()
        res = TestOnTrainingData()(self.iris, [maj])
        self.assertEqual(AUC(res)[0], 0.5)

    def test_multiclass_auc_multi_learners(self):
        learners = [LogisticRegressionLearner(),
                    MajorityLearner()]
        res = CrossValidation(k=10)(self.iris, learners)
        self.assertGreater(AUC(res)[0], 0.6)
        self.assertLess(AUC(res)[1], 0.6)
        self.assertGreater(AUC(res)[1], 0.4)

    def test_auc_on_multiclass_data_returns_1d_array(self):
        titanic = Table('titanic')[:100]
        lenses = Table(test_filename('datasets/lenses.tab'))[:100]
        majority = MajorityLearner()

        results = TestOnTrainingData()(lenses, [majority])
        auc = AUC(results)
        self.assertEqual(auc.ndim, 1)

        results = TestOnTrainingData()(titanic, [majority])
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
        probabilities = np.zeros((1, predicted.shape[1], 2))
        probabilities[0, :, 1] = predicted[0]
        probabilities[0, :, 0] = 1 - predicted[0]
        results = Results(
            nmethods=1, domain=Domain([], [DiscreteVariable("x", values='01')]),
            actual=actual, predicted=predicted)
        results.probabilities = probabilities
        return AUC(results)[0]


class TestComputeCD(unittest.TestCase):
    def test_compute_CD(self):
        avranks = [1.9, 3.2, 2.8, 3.3]
        cd = scoring.compute_CD(avranks, 30)
        np.testing.assert_almost_equal(cd, 0.856344)

        cd = scoring.compute_CD(avranks, 30, test="bonferroni-dunn")
        np.testing.assert_almost_equal(cd, 0.798)

        # Do what you will, just don't crash
        scoring.graph_ranks(avranks, "abcd", cd)
        scoring.graph_ranks(avranks, "abcd", cd, cdmethod=0)


class TestLogLoss(unittest.TestCase):
    def test_log_loss(self):
        data = Table('iris')
        majority = MajorityLearner()
        results = TestOnTrainingData()(data, [majority])
        ll = LogLoss(results)
        self.assertAlmostEqual(ll[0], - np.log(1 / 3))

    def _log_loss(self, act, prob):
        ll = np.dot(np.log(prob[:, 0]), act[:, 0]) + \
             np.dot(np.log(prob[:, 1]), act[:, 1])
        return - ll / len(act)

    def test_log_loss_calc(self):
        data = Table('titanic')
        learner = LogisticRegressionLearner()
        results = TestOnTrainingData()(data, [learner])

        actual = np.copy(results.actual)
        actual = actual.reshape(actual.shape[0], 1)
        actual = np.hstack((1 - actual, actual))
        probab = results.probabilities[0]

        ll_calc = self._log_loss(actual, probab)
        ll_orange = LogLoss(results)
        self.assertAlmostEqual(ll_calc, ll_orange[0])


class TestSpecificity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.score = Specificity()

    def test_specificity_iris(self):
        learner = LogisticRegressionLearner(preprocessors=[])
        res = TestOnTrainingData()(self.iris, [learner])
        self.assertGreaterEqual(
            self.score(res, average='weighted')[0], (1 + 0.99 + 0.95) / 3
        )
        self.assertGreaterEqual(
            self.score(res, target=1)[0], 99 / (99 + 1)
        )
        self.assertGreaterEqual(
            self.score(res, target=1, average=None)[0],  99 / (99 + 1)
        )
        self.assertGreaterEqual(
            self.score(res, target=1, average='weighted')[0], 99 / (99 + 1)
        )
        self.assertGreaterEqual(
            self.score(res, target=0, average=None)[0], 1
        )
        self.assertGreaterEqual(
            self.score(res, target=2, average=None)[0], 95 / (95 + 5)
        )

    def test_precision_multiclass(self):
        results = Results(
            domain=Domain([], DiscreteVariable(name="y", values="01234")),
            actual=[0, 4, 4, 1, 2, 0, 1, 2, 3, 2])
        results.predicted = np.array([[0, 4, 4, 1, 2, 0, 1, 2, 3, 2],
                                      [0, 1, 4, 1, 1, 0, 0, 2, 3, 1]])
        res = self.score(results, average='weighted')
        self.assertEqual(res[0], 1.)
        self.assertAlmostEqual(res[1], 0.9, 5)

        for target, prob in ((0, 7 / 8),
                             (1, 5 / 8),
                             (2, 1),
                             (3, 1),
                             (4, 1)):
            res = self.score(results, target=target, average=None)
            self.assertEqual(res[0], 1.)
            self.assertEqual(res[1], prob)

    def test_precision_binary(self):
        results = Results(
            domain=Domain([], DiscreteVariable(name="y", values="01")),
            actual=[0, 1, 1, 1, 0, 0, 1, 0, 0, 1])
        results.predicted = np.array([[0, 1, 1, 1, 0, 0, 1, 0, 0, 1],
                                      [0, 1, 1, 1, 0, 0, 1, 1, 1, 0]])
        res = self.score(results)
        self.assertEqual(res[0], 1.)
        self.assertAlmostEqual(res[1], 3 / 5)
        res_target = self.score(results, target=1)
        self.assertEqual(res[0], res_target[0])
        self.assertEqual(res[1], res_target[1])
        res_target = self.score(results, target=0)
        self.assertEqual(res_target[0], 1.)
        self.assertAlmostEqual(res_target[1], 4 / 5)

    def test_errors(self):
        learner = LogisticRegressionLearner(preprocessors=[])
        res = TestOnTrainingData()(self.iris, [learner])

        # binary average does not work for number of classes different than 2
        self.assertRaises(ValueError, self.score, res, average="binary")

        # implemented only weighted and binary averaging
        self.assertRaises(ValueError, self.score, res, average="abc")


if __name__ == '__main__':
    unittest.main()
    del TestScoreMetaType
