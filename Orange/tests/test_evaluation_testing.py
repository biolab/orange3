import unittest
import numpy as np

from Orange.classification import naive_bayes, majority
from Orange.data import discretization, Table
from Orange.evaluation import testing
from Orange.feature.discretization import EqualWidth


def random_data(nrows, ncols):
    np.random.seed(42)
    x = np.random.random_integers(1, 3, (nrows, ncols))
    col = np.random.randint(ncols)
    y = x[:nrows, col].reshape(nrows, 1)
    table = Table(x, y)
    table = discretization.DiscretizeTable(table, method=EqualWidth(n=3))
    return table


class TestingTestCase(unittest.TestCase):
    def test_no_data(self):
        self.assertRaises(TypeError, testing.CrossValidation,
                          learners=[naive_bayes.NaiveBayesLearner()])


class CrossValidationTestCase(unittest.TestCase):
    def test_results(self):
        nrows, ncols = 1000, 10
        t = random_data(nrows, ncols)
        res = testing.CrossValidation(t, [naive_bayes.NaiveBayesLearner()])
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
        res = testing.CrossValidation(t, [naive_bayes.NaiveBayesLearner()], k=5)
        self.assertEqual(len(res.folds), 5)
        for i, fold in enumerate(res.folds):
            self.assertAlmostEqual(fold.start, i * 200, delta=3)
            self.assertAlmostEqual(fold.stop, (i + 1) * 200, delta=3)

    def test_call_5(self):
        nrows, ncols = 1000, 10
        t = random_data(nrows, ncols)
        res = testing.CrossValidation(t, [naive_bayes.NaiveBayesLearner()], k=5)
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
        nrows, ncols = 1000, 10
        t = random_data(nrows, ncols)
        learners = [naive_bayes.NaiveBayesLearner()]

        res = testing.CrossValidation(t, learners)
        self.assertIsNone(res.data)

        res = testing.CrossValidation(t, learners, store_data=True)
        self.assertIs(res.data, t)

        res = testing.CrossValidation(t, learners)
        self.assertIsNone(res.data)

        res = testing.CrossValidation(t, learners, store_data=True)
        self.assertIs(res.data, t)

    def test_store_models(self):
        nrows, ncols = 1000, 10
        t = random_data(nrows, ncols)
        learners = [naive_bayes.NaiveBayesLearner(), majority.MajorityLearner()]

        res = testing.CrossValidation(t, learners, k=5)
        self.assertIsNone(res.models)

        res = testing.CrossValidation(t, learners, k=5, store_models=True)
        self.assertEqual(len(res.models), 5)
        for models in res.models:
            self.assertEqual(len(models), 2)
            self.assertIsInstance(models[0], naive_bayes.NaiveBayesModel)
            self.assertIsInstance(models[1], majority.ConstantModel)

        res = testing.CrossValidation(t, learners, k=5)
        self.assertIsNone(res.models)

        res = testing.CrossValidation(t, learners, k=5, store_models=True)
        self.assertEqual(len(res.models), 5)
        for models in res.models:
            self.assertEqual(len(models), 2)
            self.assertIsInstance(models[0], naive_bayes.NaiveBayesModel)
            self.assertIsInstance(models[1], majority.ConstantModel)

    def test_10_fold_probs(self):
        data = Table('iris')[30:130]
        learners = [majority.MajorityLearner(), majority.MajorityLearner()]

        results = testing.CrossValidation(data, learners, k=10)

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
        res = testing.CrossValidation(data, [majority.MajorityLearner()])
        np.testing.assert_equal(res.predicted[0][:49], 0)

        x[49] = 0
        res = testing.CrossValidation(data, [majority.MajorityLearner()])
        np.testing.assert_equal(res.predicted[0][:49], 0)


class LeaveOneOutTestCase(unittest.TestCase):
    def test_results(self):
        nrows, ncols = 100, 10
        t = random_data(nrows, ncols)
        res = testing.LeaveOneOut(t, [naive_bayes.NaiveBayesLearner()])
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
        res = testing.LeaveOneOut(t, [naive_bayes.NaiveBayesLearner()])
        y = t.Y
        np.testing.assert_equal(res.actual, y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(res.predicted[0],
                                y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(np.argmax(res.probabilities[0], axis=1),
                                y[res.row_indices].reshape(nrows))

    def test_store_data(self):
        nrows, ncols = 50, 10
        t = random_data(nrows, ncols)
        learners = [naive_bayes.NaiveBayesLearner()]

        res = testing.LeaveOneOut(t, learners)
        self.assertIsNone(res.data)

        res = testing.LeaveOneOut(t, learners, store_data=True)
        self.assertIs(res.data, t)

        res = testing.LeaveOneOut(t, learners)
        self.assertIsNone(res.data)

        res = testing.LeaveOneOut(t, learners, store_data=True)
        self.assertIs(res.data, t)

    def test_store_models(self):
        nrows, ncols = 50, 10
        t = random_data(nrows, ncols)
        learners = [naive_bayes.NaiveBayesLearner(), majority.MajorityLearner()]

        res = testing.LeaveOneOut(t, learners)
        self.assertIsNone(res.models)

        res = testing.LeaveOneOut(t, learners, store_models=True)
        self.assertEqual(len(res.models), 50)
        for models in res.models:
            self.assertEqual(len(models), 2)
            self.assertIsInstance(models[0], naive_bayes.NaiveBayesModel)
            self.assertIsInstance(models[1], majority.ConstantModel)

        res = testing.LeaveOneOut(t, learners)
        self.assertIsNone(res.models)

        res = testing.LeaveOneOut(t, learners, store_models=True)
        self.assertEqual(len(res.models), 50)
        for models in res.models:
            self.assertEqual(len(models), 2)
            self.assertIsInstance(models[0], naive_bayes.NaiveBayesModel)
            self.assertIsInstance(models[1], majority.ConstantModel)

    def test_probs(self):
        data = Table('iris')[30:130]
        learners = [majority.MajorityLearner(), majority.MajorityLearner()]

        results = testing.LeaveOneOut(data, learners)

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
        res = testing.LeaveOneOut(data, [majority.MajorityLearner()])
        np.testing.assert_equal(res.predicted[0][:49], 0)

        x[49] = 0
        res = testing.LeaveOneOut(data, [majority.MajorityLearner()])
        np.testing.assert_equal(res.predicted[0][:49], 0)

        x[25:] = 1
        y = x[:, -1]
        data = Table(x, y)
        res = testing.LeaveOneOut(data, [majority.MajorityLearner()])
        np.testing.assert_equal(res.predicted[0],
                                1 - data.Y[res.row_indices].flatten())



class TestOnTrainingTestCase(unittest.TestCase):
    def test_results(self):
        nrows, ncols = 50, 10
        t = random_data(nrows, ncols)
        res = testing.TestOnTrainingData(t, [naive_bayes.NaiveBayesLearner()])
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
        learners = [naive_bayes.NaiveBayesLearner()]

        res = testing.TestOnTrainingData(t, learners)
        self.assertIsNone(res.data)

        res = testing.TestOnTrainingData(t, learners, store_data=True)
        self.assertIs(res.data, t)

        res = testing.TestOnTrainingData(t, learners)
        self.assertIsNone(res.data)

        res = testing.TestOnTrainingData(t, learners, store_data=True)
        self.assertIs(res.data, t)

    def test_store_models(self):
        nrows, ncols = 50, 10
        t = random_data(nrows, ncols)
        learners = [naive_bayes.NaiveBayesLearner(), majority.MajorityLearner()]

        res = testing.TestOnTrainingData(t, learners)
        self.assertIsNone(res.models)

        res = testing.TestOnTrainingData(t, learners, store_models=True)
        self.assertEqual(len(res.models), 1)
        for models in res.models:
            self.assertEqual(len(models), 2)
            self.assertIsInstance(models[0], naive_bayes.NaiveBayesModel)
            self.assertIsInstance(models[1], majority.ConstantModel)

        res = testing.TestOnTrainingData(t, learners)
        self.assertIsNone(res.models)

        res = testing.TestOnTrainingData(t, learners, store_models=True)
        self.assertEqual(len(res.models), 1)
        for models in res.models:
            self.assertEqual(len(models), 2)
            self.assertIsInstance(models[0], naive_bayes.NaiveBayesModel)
            self.assertIsInstance(models[1], majority.ConstantModel)

    def test_probs(self):
        data = Table('iris')[30:130]
        learners = [majority.MajorityLearner(), majority.MajorityLearner()]

        results = testing.TestOnTrainingData(data, learners)

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
        res = testing.TestOnTrainingData(data, [majority.MajorityLearner()])
        np.testing.assert_equal(res.predicted[0][:49], 0)

        x[49] = 0
        res = testing.TestOnTrainingData(data, [majority.MajorityLearner()])
        np.testing.assert_equal(res.predicted[0][:49], 0)

        x[25:] = 1
        y = x[:, -1]
        data = Table(x, y)
        res = testing.TestOnTrainingData(data, [majority.MajorityLearner()])
        np.testing.assert_equal(res.predicted[0], res.predicted[0][0])
