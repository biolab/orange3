import unittest
import Orange
import numpy as np

from Orange.classification import NaiveBayesLearner, MajorityLearner
from Orange.classification.majority import ConstantModel
from Orange.classification.naive_bayes import NaiveBayesModel
from Orange.data import Table
from Orange.evaluation import *
from Orange.preprocess import discretize, preprocess


def random_data(nrows, ncols):
    np.random.seed(42)
    x = np.random.random_integers(1, 3, (nrows, ncols))
    col = np.random.randint(ncols)
    y = x[:nrows, col].reshape(nrows, 1)
    table = Table(x, y)
    table = preprocess.Discretize(discretize.EqualWidth(n=3))(table)
    return table


class TestingTestCase(unittest.TestCase):
    def test_no_data(self):
        self.assertRaises(TypeError, CrossValidation,
                          learners=[NaiveBayesLearner()])


class CrossValidationTestCase(unittest.TestCase):
    def test_results(self):
        nrows, ncols = 1000, 10
        t = random_data(nrows, ncols)
        res = CrossValidation(t, [NaiveBayesLearner()])
        y = t.Y
        np.testing.assert_equal(res.actual, y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(res.predicted[0],
                                y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(np.argmax(res.probabilities[0], axis=1),
                                y[res.row_indices].reshape(nrows))
        self.assertEqual(len(res.folds), 10)
        for i, fold in enumerate(res.folds):
            self.assertAlmostEqual(fold.start, i * 100, delta=3)
            self.assertAlmostEqual(fold.stop, (i + 1) * 100, delta=3)

    def test_folds(self):
        nrows, ncols = 1000, 10
        t = random_data(nrows, ncols)
        res = CrossValidation(t, [NaiveBayesLearner()], k=5)
        self.assertEqual(len(res.folds), 5)
        for i, fold in enumerate(res.folds):
            self.assertAlmostEqual(fold.start, i * 200, delta=3)
            self.assertAlmostEqual(fold.stop, (i + 1) * 200, delta=3)

    def test_call_5(self):
        nrows, ncols = 1000, 10
        t = random_data(nrows, ncols)
        res = CrossValidation(t, [NaiveBayesLearner()], k=5)
        y = t.Y
        np.testing.assert_equal(res.actual, y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(res.predicted[0],
                                y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(np.argmax(res.probabilities[0], axis=1),
                                y[res.row_indices].reshape(nrows))
        self.assertEqual(len(res.folds), 5)
        for i, fold in enumerate(res.folds):
            self.assertAlmostEqual(fold.start, i * 200, delta=3)
            self.assertAlmostEqual(fold.stop, (i + 1) * 200, delta=3)

    def test_store_data(self):
        nrows, ncols = 100, 10
        t = random_data(nrows, ncols)
        learners = [NaiveBayesLearner()]

        res = CrossValidation(t, learners)
        self.assertIsNone(res.data)

        res = CrossValidation(t, learners, store_data=True)
        self.assertIs(res.data, t)

        res = CrossValidation(t, learners)
        self.assertIsNone(res.data)

        res = CrossValidation(t, learners, store_data=True)
        self.assertIs(res.data, t)

    def test_store_models(self):
        nrows, ncols = 100, 10
        t = random_data(nrows, ncols)
        learners = [NaiveBayesLearner(), MajorityLearner()]

        res = CrossValidation(t, learners, k=5)
        self.assertIsNone(res.models)

        res = CrossValidation(t, learners, k=5, store_models=True)
        self.assertEqual(len(res.models), 5)
        for models in res.models:
            self.assertEqual(len(models), 2)
            self.assertIsInstance(models[0], NaiveBayesModel)
            self.assertIsInstance(models[1], ConstantModel)

        res = CrossValidation(t, learners, k=5)
        self.assertIsNone(res.models)

        res = CrossValidation(t, learners, k=5, store_models=True)
        self.assertEqual(len(res.models), 5)
        for models in res.models:
            self.assertEqual(len(models), 2)
            self.assertIsInstance(models[0], NaiveBayesModel)
            self.assertIsInstance(models[1], ConstantModel)

    def test_10_fold_probs(self):
        data = Table('iris')[30:130]
        learners = [MajorityLearner(), MajorityLearner()]

        results = CrossValidation(data, learners, k=10)

        self.assertEqual(results.predicted.shape, (2, len(data)))
        np.testing.assert_equal(results.predicted, np.ones((2, 100)))
        probs = results.probabilities
        self.assertTrue((probs[:, :, 0] < probs[:, :, 2]).all())
        self.assertTrue((probs[:, :, 2] < probs[:, :, 1]).all())

    def test_miss_majority(self):
        x = np.zeros((50, 3))
        y = x[:, -1]
        x[-4:] = np.ones((4, 3))
        data = Table(x, y)
        res = CrossValidation(data, [MajorityLearner()], k=3)
        np.testing.assert_equal(res.predicted[0][:49], 0)

        x[-4:] = np.zeros((4, 3))
        res = CrossValidation(data, [MajorityLearner()], k=3)
        np.testing.assert_equal(res.predicted[0][:49], 0)


class LeaveOneOutTestCase(unittest.TestCase):
    def test_results(self):
        nrows, ncols = 100, 10
        t = random_data(nrows, ncols)
        res = LeaveOneOut(t, [NaiveBayesLearner()])
        y = t.Y
        np.testing.assert_equal(res.actual, y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(res.predicted[0],
                                y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(np.argmax(res.probabilities[0], axis=1),
                                y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(res.row_indices, np.arange(nrows))

    def test_call(self):
        nrows, ncols = 100, 10
        t = random_data(nrows, ncols)
        res = LeaveOneOut(t, [NaiveBayesLearner()])
        y = t.Y
        np.testing.assert_equal(res.actual, y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(res.predicted[0],
                                y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(np.argmax(res.probabilities[0], axis=1),
                                y[res.row_indices].reshape(nrows))

    def test_store_data(self):
        nrows, ncols = 50, 10
        t = random_data(nrows, ncols)
        learners = [NaiveBayesLearner()]

        res = LeaveOneOut(t, learners)
        self.assertIsNone(res.data)

        res = LeaveOneOut(t, learners, store_data=True)
        self.assertIs(res.data, t)

        res = LeaveOneOut(t, learners)
        self.assertIsNone(res.data)

        res = LeaveOneOut(t, learners, store_data=True)
        self.assertIs(res.data, t)

    def test_store_models(self):
        nrows, ncols = 50, 10
        t = random_data(nrows, ncols)
        learners = [NaiveBayesLearner(), MajorityLearner()]

        res = LeaveOneOut(t, learners)
        self.assertIsNone(res.models)

        res = LeaveOneOut(t, learners, store_models=True)
        self.assertEqual(len(res.models), 50)
        for models in res.models:
            self.assertEqual(len(models), 2)
            self.assertIsInstance(models[0], NaiveBayesModel)
            self.assertIsInstance(models[1], ConstantModel)

        res = LeaveOneOut(t, learners)
        self.assertIsNone(res.models)

        res = LeaveOneOut(t, learners, store_models=True)
        self.assertEqual(len(res.models), 50)
        for models in res.models:
            self.assertEqual(len(models), 2)
            self.assertIsInstance(models[0], NaiveBayesModel)
            self.assertIsInstance(models[1], ConstantModel)

    def test_probs(self):
        data = Table('iris')[30:130]
        learners = [MajorityLearner(), MajorityLearner()]

        results = LeaveOneOut(data, learners)

        self.assertEqual(results.predicted.shape, (2, len(data)))
        np.testing.assert_equal(results.predicted, np.ones((2, 100)))
        probs = results.probabilities
        self.assertTrue((probs[:, :, 0] < probs[:, :, 2]).all())
        self.assertTrue((probs[:, :, 2] < probs[:, :, 1]).all())

    def test_miss_majority(self):
        x = np.zeros((50, 3))
        y = x[:, -1]
        x[49] = 1
        data = Table(x, y)
        res = LeaveOneOut(data, [MajorityLearner()])
        np.testing.assert_equal(res.predicted[0][:49], 0)

        x[49] = 0
        res = LeaveOneOut(data, [MajorityLearner()])
        np.testing.assert_equal(res.predicted[0][:49], 0)

        x[25:] = 1
        y = x[:, -1]
        data = Table(x, y)
        res = LeaveOneOut(data, [MajorityLearner()])
        np.testing.assert_equal(res.predicted[0],
                                1 - data.Y[res.row_indices].flatten())



class TestOnTrainingTestCase(unittest.TestCase):
    def test_results(self):
        nrows, ncols = 50, 10
        t = random_data(nrows, ncols)
        res = TestOnTrainingData(t, [NaiveBayesLearner()])
        y = t.Y
        np.testing.assert_equal(res.actual, y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(res.predicted[0],
                                y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(np.argmax(res.probabilities[0], axis=1),
                                y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(res.row_indices, np.arange(nrows))

    def test_store_data(self):
        nrows, ncols = 50, 10
        t = random_data(nrows, ncols)
        learners = [NaiveBayesLearner()]

        res = TestOnTrainingData(t, learners)
        self.assertIsNone(res.data)

        res = TestOnTrainingData(t, learners, store_data=True)
        self.assertIs(res.data, t)

        res = TestOnTrainingData(t, learners)
        self.assertIsNone(res.data)

        res = TestOnTrainingData(t, learners, store_data=True)
        self.assertIs(res.data, t)

    def test_store_models(self):
        nrows, ncols = 50, 10
        t = random_data(nrows, ncols)
        learners = [NaiveBayesLearner(), MajorityLearner()]

        res = TestOnTrainingData(t, learners)
        self.assertIsNone(res.models)

        res = TestOnTrainingData(t, learners, store_models=True)
        self.assertEqual(len(res.models), 1)
        for models in res.models:
            self.assertEqual(len(models), 2)
            self.assertIsInstance(models[0], NaiveBayesModel)
            self.assertIsInstance(models[1], ConstantModel)

        res = TestOnTrainingData(t, learners)
        self.assertIsNone(res.models)

        res = TestOnTrainingData(t, learners, store_models=True)
        self.assertEqual(len(res.models), 1)
        for models in res.models:
            self.assertEqual(len(models), 2)
            self.assertIsInstance(models[0], NaiveBayesModel)
            self.assertIsInstance(models[1], ConstantModel)

    def test_probs(self):
        data = Table('iris')[30:130]
        learners = [MajorityLearner(), MajorityLearner()]

        results = TestOnTrainingData(data, learners)

        self.assertEqual(results.predicted.shape, (2, len(data)))
        np.testing.assert_equal(results.predicted, np.ones((2, 100)))
        probs = results.probabilities
        self.assertTrue((probs[:, :, 0] < probs[:, :, 2]).all())
        self.assertTrue((probs[:, :, 2] < probs[:, :, 1]).all())

    def test_miss_majority(self):
        x = np.zeros((50, 3))
        y = x[:, -1]
        x[49] = 1
        data = Table(x, y)
        res = TestOnTrainingData(data, [MajorityLearner()])
        np.testing.assert_equal(res.predicted[0][:49], 0)

        x[49] = 0
        res = TestOnTrainingData(data, [MajorityLearner()])
        np.testing.assert_equal(res.predicted[0][:49], 0)

        x[25:] = 1
        y = x[:, -1]
        data = Table(x, y)
        res = TestOnTrainingData(data, [MajorityLearner()])
        np.testing.assert_equal(res.predicted[0], res.predicted[0][0])


class TestTrainTestSplit(unittest.TestCase):
    def test_fixed_training_size(self):
        data = Orange.data.Table("iris")
        train, test = Orange.evaluation.sample(data, 100)
        self.assertEqual(len(train), 100)
        self.assertEqual(len(train) + len(test), len(data))

        train, test = Orange.evaluation.sample(data, 0.1)
        self.assertEqual(len(train), 15)
        self.assertEqual(len(train) + len(test), len(data))

        train, test = Orange.evaluation.sample(data, 0.1, stratified=True)
        self.assertEqual(len(train), 15)
        self.assertEqual(len(train) + len(test), len(data))

        train, test = Orange.evaluation.sample(data, 0.2, replace=True)
        self.assertEqual(len(train), 30)

        train, test = Orange.evaluation.sample(data, 0.9, replace=True)
        self.assertEqual(len(train), 135)
        self.assertGreater(len(train) + len(test), len(data))
