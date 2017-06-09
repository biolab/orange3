# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import numpy as np

from Orange.classification import NaiveBayesLearner, MajorityLearner
from Orange.regression import LinearRegressionLearner, MeanLearner
from Orange.data import Table, Domain, DiscreteVariable
from Orange.evaluation import (Results, CrossValidation, LeaveOneOut, TestOnTrainingData,
                               TestOnTestData, ShuffleSplit, sample, RMSE,
                               CrossValidationFeature)
from Orange.preprocess import discretize, preprocess


def random_data(nrows, ncols):
    np.random.seed(42)
    x = np.random.randint(0, 2, (nrows, ncols))
    col = np.random.randint(ncols)
    y = x[:nrows, col].reshape(nrows, 1)
    table = Table(x, y)
    table = preprocess.Discretize(discretize.EqualWidth(n=3))(table)
    return table


class TestingTestCase(unittest.TestCase):
    def test_no_data(self):
        self.assertRaises(TypeError, CrossValidation,
                          learners=[NaiveBayesLearner()])


class _ParameterTuningLearner(MajorityLearner):
    def __call__(self, data):
        learner = MajorityLearner()
        CrossValidation(data, [learner], k=2)
        return learner(data)


# noinspection PyUnresolvedReferences
class TestSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.nrows = 200
        cls.ncols = 5
        cls.random_table = random_data(cls.nrows, cls.ncols)

    def run_test_failed(self, method, succ_calls):
        # Can't use mocking helpers here (wrong result type for Majority,
        # exception caught for fails)
        def major(*args):
            nonlocal major_call
            major_call += 1
            return MajorityLearner()(*args)

        def fails(_):
            nonlocal fail_calls
            fail_calls += 1
            raise SystemError("failing learner")

        major_call = 0
        fail_calls = 0
        res = method(random_data(50, 4), [major, fails, major])
        self.assertFalse(res.failed[0])
        self.assertIsInstance(res.failed[1], Exception)
        self.assertFalse(res.failed[2])
        self.assertEqual(major_call, succ_calls)
        self.assertEqual(fail_calls, succ_calls / 2)

    def run_test_callback(self, method, expected_progresses):
        def record_progress(p):
            progress.append(p)
        progress = []
        method(self.random_table, [MajorityLearner(), MajorityLearner()],
               callback=record_progress)
        np.testing.assert_almost_equal(np.array(progress), expected_progresses)

    def run_test_preprocessor(self, method, expected_sizes):
        def preprocessor(data):
            data_sizes.append(len(data))
            return data
        data_sizes = []
        method(Table('iris'), [MajorityLearner(), MajorityLearner()],
               preprocessor=preprocessor)
        self.assertEqual(data_sizes, expected_sizes)

    def check_folds(self, result, folds_count, rows):
        self.assertEqual(len(result.folds), folds_count)
        fold_size = rows / folds_count
        for i, fold in enumerate(result.folds):
            self.assertAlmostEqual(fold.start, i * fold_size, delta=3)
            self.assertAlmostEqual(fold.stop, (i + 1) * fold_size, delta=3)

    def check_models(self, result, learners, folds):
        self.assertEqual(result.models.shape, (folds, len(learners)))
        for models in result.models:
            for model, learner in zip(models, learners):
                self.assertIsInstance(model, learner.__returns__)

    def _callback_values(self, iterations):
        return np.hstack((np.linspace(0., .99, iterations + 1)[1:], [1]))


class TestCrossValidation(TestSampling):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.housing = Table('housing')
        cls.nrows = 50
        cls.ncols = 5
        cls.random_table = random_data(cls.nrows, cls.ncols)

    def test_results(self):
        nrows, ncols = self.random_table.X.shape
        res = CrossValidation(self.random_table, [NaiveBayesLearner()], k=10,
                              stratified=False)
        y = self.random_table.Y
        np.testing.assert_equal(res.actual, y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(res.predicted[0],
                                y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(np.argmax(res.probabilities[0], axis=1),
                                y[res.row_indices].reshape(nrows))
        self.check_folds(res, 10, nrows)

    def test_continuous(self):
        res = CrossValidation(self.housing, [LinearRegressionLearner()], k=3, n_jobs=1)
        self.assertLess(RMSE(res), 5)

    def test_folds(self):
        res = CrossValidation(self.random_table, [NaiveBayesLearner()], k=5)
        self.check_folds(res, 5, self.nrows)

    def test_call_5(self):
        nrows, ncols = self.random_table.X.shape
        res = CrossValidation(self.random_table, [NaiveBayesLearner()], k=5,
                              stratified=False)
        y = self.random_table.Y
        np.testing.assert_equal(res.actual, y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(res.predicted[0],
                                y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(np.argmax(res.probabilities[0], axis=1),
                                y[res.row_indices].reshape(nrows))
        self.check_folds(res, 5, nrows)

    def test_store_data(self):
        learners = [NaiveBayesLearner()]
        res = CrossValidation(self.random_table, learners, store_data=False)
        self.assertIsNone(res.data)

        res = CrossValidation(self.random_table, learners, store_data=True)
        self.assertIs(res.data, self.random_table)

    def test_store_models(self):
        learners = [NaiveBayesLearner(), MajorityLearner()]

        res = CrossValidation(self.random_table, learners, k=5, store_models=False)
        self.assertIsNone(res.models)

        res = CrossValidation(self.random_table, learners, k=5, store_models=True)
        self.assertEqual(len(res.models), 5)
        self.check_models(res, learners, 5)

    def test_split_by_model(self):
        learners = [NaiveBayesLearner(), MajorityLearner()]
        res = CrossValidation(self.random_table, learners, k=5, store_models=True)

        for i, result in enumerate(res.split_by_model()):
            self.assertIsInstance(result, Results)
            self.assertTrue((result.predicted == res.predicted[i]).all())
            self.assertTrue((result.probabilities == res.probabilities[i]).all())
            self.assertEqual(len(result.models), 5)
            for model in result.models:
                self.assertIsInstance(model, learners[i].__returns__)
            self.assertSequenceEqual(result.learners, [res.learners[i]])

    def test_10_fold_probs(self):
        learners = [MajorityLearner(), MajorityLearner()]

        results = CrossValidation(self.iris[30:130], learners, k=10)

        self.assertEqual(results.predicted.shape, (2, len(self.iris[30:130])))
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

    def test_too_many_folds(self):
        w = []
        res = CrossValidation(self.iris, [MajorityLearner()], k=len(self.iris)/2, warnings=w)
        self.assertGreater(len(w), 0)

    def test_failed(self):
        self.run_test_failed(CrossValidation, 20)

    def test_callback(self):
        self.run_test_callback(CrossValidation, self._callback_values(20))

    def test_preprocessor(self):
        self.run_test_preprocessor(CrossValidation, [135] * 10)

    def test_augmented_data_classification(self):
        data = Table("iris")
        n_classes = len(data.domain.class_var.values)
        res = CrossValidation(data, [NaiveBayesLearner()], store_data=True)
        table = res.get_augmented_data(['Naive Bayes'])

        self.assertEqual(len(table), len(data))
        self.assertEqual(len(table.domain.attributes), len(data.domain.attributes))
        self.assertEqual(len(table.domain.class_vars), len(data.domain.class_vars))
        # +1 for class, +n_classes for probabilities, +1 for fold
        self.assertEqual(len(table.domain.metas), len(data.domain.metas) + 1 + n_classes + 1)
        self.assertEqual(table.domain.metas[len(data.domain.metas)].values, data.domain.class_var.values)

        res = CrossValidation(data, [NaiveBayesLearner(), MajorityLearner()], store_data=True)
        table = res.get_augmented_data(['Naive Bayes', 'Majority'])

        self.assertEqual(len(table), len(data))
        self.assertEqual(len(table.domain.attributes), len(data.domain.attributes))
        self.assertEqual(len(table.domain.class_vars), len(data.domain.class_vars))
        self.assertEqual(len(table.domain.metas), len(data.domain.metas) + 2*(n_classes+1) + 1)
        self.assertEqual(table.domain.metas[len(data.domain.metas)].values, data.domain.class_var.values)
        self.assertEqual(table.domain.metas[len(data.domain.metas)+1].values, data.domain.class_var.values)

    def test_augmented_data_regression(self):
        data = Table("housing")
        res = CrossValidation(data, [LinearRegressionLearner(), ], store_data=True)
        table = res.get_augmented_data(['Linear Regression'])

        self.assertEqual(len(table), len(data))
        self.assertEqual(len(table.domain.attributes), len(data.domain.attributes))
        self.assertEqual(len(table.domain.class_vars), len(data.domain.class_vars))
        # +1 for class, +1 for fold
        self.assertEqual(len(table.domain.metas), len(data.domain.metas) + 1 + 1)

        res = CrossValidation(data, [LinearRegressionLearner(), MeanLearner()], store_data=True)
        table = res.get_augmented_data(['Linear Regression', 'Mean Learner'])

        self.assertEqual(len(table), len(data))
        self.assertEqual(len(table.domain.attributes), len(data.domain.attributes))
        self.assertEqual(len(table.domain.class_vars), len(data.domain.class_vars))
        # +2 for class, +1 for fold
        self.assertEqual(len(table.domain.metas), len(data.domain.metas) + 2 + 1)


class TestCrossValidationFeature(TestSampling):

    def add_meta_fold(self, data, f):
        fat = DiscreteVariable(name="fold", values=[str(a) for a in range(f)])
        domain = Domain(data.domain.attributes, data.domain.class_var, metas=[fat])
        ndata = Table(domain, data)
        vals = np.tile(range(f), len(data)//f + 1)[:len(data)]
        vals = vals.reshape((-1, 1))
        ndata[:, fat] = vals
        return ndata

    def test_call(self):
        t = self.random_table
        t = self.add_meta_fold(t, 3)
        res = CrossValidationFeature(t, [NaiveBayesLearner()], feature=t.domain.metas[0])
        y = t.Y
        np.testing.assert_equal(res.actual, y[res.row_indices].reshape(len(t)))
        np.testing.assert_equal(res.predicted[0],
                                y[res.row_indices].reshape(len(t)))
        np.testing.assert_equal(np.argmax(res.probabilities[0], axis=1),
                                y[res.row_indices].reshape(len(t)))

    def test_unknown(self):
        t = self.random_table
        t = self.add_meta_fold(t, 3)
        fat = t.domain.metas[0]
        t[0][fat] = float("nan")
        res = CrossValidationFeature(t, [NaiveBayesLearner()], feature=fat)
        self.assertNotIn(0, res.row_indices)


class TestLeaveOneOut(TestSampling):
    def test_results(self):
        nrows = self.nrows
        t = self.random_table
        res = LeaveOneOut(t, [NaiveBayesLearner()])
        y = t.Y
        np.testing.assert_equal(res.actual, y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(res.predicted[0],
                                y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(np.argmax(res.probabilities[0], axis=1),
                                y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(res.row_indices, np.arange(nrows))

    def test_call(self):
        nrows = self.nrows
        t = self.random_table
        res = LeaveOneOut(t, [NaiveBayesLearner()])
        y = t.Y
        np.testing.assert_equal(res.actual, y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(res.predicted[0],
                                y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(np.argmax(res.probabilities[0], axis=1),
                                y[res.row_indices].reshape(nrows))

    def test_store_data(self):
        t = self.random_table
        learners = [NaiveBayesLearner()]

        res = LeaveOneOut(t, learners, store_data=False)
        self.assertIsNone(res.data)

        res = LeaveOneOut(t, learners, store_data=True)
        self.assertIs(res.data, t)

    def test_store_models(self):
        t = self.random_table
        learners = [NaiveBayesLearner(), MajorityLearner()]

        res = LeaveOneOut(t, learners)
        self.assertIsNone(res.models)

        res = LeaveOneOut(t, learners, store_models=True)
        self.check_models(res, learners, self.nrows)

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
        data = Table(x, y)
        res = LeaveOneOut(data, [MajorityLearner()])
        np.testing.assert_equal(res.predicted[0],
                                1 - data.Y[res.row_indices].flatten())

    def test_failed(self):
        self.run_test_failed(LeaveOneOut, 100)

    def test_callback(self):
        self.run_test_callback(LeaveOneOut, self._callback_values(2 * self.nrows))

    def test_preprocessor(self):
        self.run_test_preprocessor(LeaveOneOut, [149] * 150)


class TestTestOnTrainingData(TestSampling):
    def test_results(self):
        nrows, ncols = self.random_table.X.shape
        t = self.random_table
        res = TestOnTrainingData(t, [NaiveBayesLearner()])
        y = t.Y
        np.testing.assert_equal(res.actual, y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(res.predicted[0],
                                y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(np.argmax(res.probabilities[0], axis=1),
                                y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(res.row_indices, np.arange(nrows))

    def test_store_data(self):
        t = self.random_table
        learners = [NaiveBayesLearner()]
        res = TestOnTrainingData(t, learners)
        self.assertIsNone(res.data)
        res = TestOnTrainingData(t, learners, store_data=True)
        self.assertIs(res.data, t)

    def test_store_models(self):
        t = self.random_table
        learners = [NaiveBayesLearner(), MajorityLearner()]

        res = TestOnTrainingData(t, learners)
        self.assertIsNone(res.models)

        res = TestOnTrainingData(t, learners, store_models=True)
        self.check_models(res, learners, 1)

    def test_probs(self):
        data = self.iris[30:130]
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
        data = Table(x, y)
        res = TestOnTrainingData(data, [MajorityLearner()])
        np.testing.assert_equal(res.predicted[0], res.predicted[0][0])

    def test_failed(self):
        self.run_test_failed(TestOnTrainingData, 2)

    def test_callback(self):
        self.run_test_callback(TestOnTrainingData, self._callback_values(2))

    def test_preprocessor(self):
        self.run_test_preprocessor(TestOnTrainingData, [150])


class TestTestOnTestData(TestSampling):
    def test_results(self):
        nrows, ncols = self.random_table.X.shape
        t = self.random_table
        res = TestOnTestData(t, t, [NaiveBayesLearner()])
        y = t.Y
        np.testing.assert_equal(res.actual, y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(res.predicted[0],
                                y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(np.argmax(res.probabilities[0], axis=1),
                                y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(res.row_indices, np.arange(nrows))

    def test_probs(self):
        data = self.iris[30:130]
        learners = [MajorityLearner(), MajorityLearner()]
        results = TestOnTestData(data, data, learners)

        self.assertEqual(results.predicted.shape, (2, len(data)))
        np.testing.assert_equal(results.predicted, np.ones((2, 100)))
        probs = results.probabilities
        self.assertTrue((probs[:, :, 0] < probs[:, :, 2]).all())
        self.assertTrue((probs[:, :, 2] < probs[:, :, 1]).all())

        train = self.iris[50:120]
        test = self.iris[:50]
        results = TestOnTestData(train, test, learners)
        self.assertEqual(results.predicted.shape, (2, len(test)))
        np.testing.assert_equal(results.predicted, np.ones((2, 50)))
        probs = results.probabilities
        self.assertTrue((probs[:, :, 0] == 0).all())

    def test_store_data(self):
        data = self.random_table
        train = data[:int(self.nrows*.75)]
        test = data[int(self.nrows*.75):]
        learners = [MajorityLearner()]

        res = TestOnTestData(train, test, learners)
        self.assertIsNone(res.data)

        res = TestOnTestData(train, test, learners, store_data=True)
        self.assertIs(res.data, test)

    def test_store_models(self):
        data = self.random_table
        train = data[:int(self.nrows*.75)]
        test = data[int(self.nrows*.75):]
        learners = [NaiveBayesLearner(), MajorityLearner()]

        res = TestOnTestData(train, test, learners)
        self.assertIsNone(res.models)

        res = TestOnTestData(train, test, learners, store_models=True)
        self.check_models(res, learners, 1)

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

    def run_test_failed(self, method, succ_calls):
        # Can't use mocking helpers here (wrong result type for Majority,
        # exception caught for fails)
        def major(*args):
            nonlocal major_call
            major_call += 1
            return MajorityLearner()(*args)

        def fails(_):
            nonlocal fail_calls
            fail_calls += 1
            raise SystemError("failing learner")

        major_call = 0
        fail_calls = 0
        data = random_data(50, 4)
        res = TestOnTestData(data, data, [major, fails, major])
        self.assertFalse(res.failed[0])
        self.assertIsInstance(res.failed[1], Exception)
        self.assertFalse(res.failed[2])
        self.assertEqual(major_call, 2)
        self.assertEqual(fail_calls, 1)

    def test_callback(self):
        def record_progress(p):
            progress.append(p)

        progress = []
        data = random_data(50, 4)
        TestOnTestData(data, data, [MajorityLearner(), MajorityLearner()],
                       callback=record_progress)
        np.testing.assert_almost_equal(progress, self._callback_values(2))

    def test_preprocessor(self):
        def preprocessor(data):
            data_sizes.append(len(data))
            return data

        data_sizes = []
        data = random_data(50, 5)
        TestOnTestData(data[:30], data[-20:],
                       [MajorityLearner(), MajorityLearner()],
                       preprocessor=preprocessor)
        self.assertEqual(data_sizes, [30])


class TestTrainTestSplit(unittest.TestCase):
    def test_fixed_training_size(self):
        data = Table("iris")
        train, test = sample(data, 100)
        self.assertEqual(len(train), 100)
        self.assertEqual(len(train) + len(test), len(data))

        train, test = sample(data, 0.1)
        self.assertEqual(len(train), 15)
        self.assertEqual(len(train) + len(test), len(data))

        train, test = sample(data, 0.1, stratified=True)
        self.assertEqual(len(train), 15)
        self.assertEqual(len(train) + len(test), len(data))

        train, test = sample(data, 0.2, replace=True)
        self.assertEqual(len(train), 30)

        train, test = sample(data, 0.9, replace=True)
        self.assertEqual(len(train), 135)
        self.assertGreater(len(train) + len(test), len(data))


class TestShuffleSplit(TestSampling):
    def test_results(self):
        data = self.random_table
        train_size, n_resamples = 0.6, 10
        res = ShuffleSplit(data, [NaiveBayesLearner()], train_size=train_size,
                           test_size=1 - train_size, n_resamples=n_resamples)
        self.assertEqual(len(res.predicted[0]),
                         n_resamples * self.nrows * (1 - train_size))

    def test_stratified(self):
        # strata size
        n = 50
        res = ShuffleSplit(self.iris, [NaiveBayesLearner()],
                           train_size=.5, test_size=.5,
                           n_resamples=3, stratified=True, random_state=0)

        strata_samples = []
        for train, test in res.indices:
            strata_samples.append(np.count_nonzero(train < n) == n/2)
            strata_samples.append(np.count_nonzero(train < 2 * n) == n)

        self.assertTrue(all(strata_samples))

    def test_not_stratified(self):
        # strata size
        n = 50
        res = ShuffleSplit(self.iris, [NaiveBayesLearner()],
                           train_size=.5, test_size=.5,
                           n_resamples=3, stratified=False, random_state=0)

        strata_samples = []
        for train, test in res.indices:
            strata_samples.append(np.count_nonzero(train < n) == n/2)
            strata_samples.append(np.count_nonzero(train < 2 * n) == n)

        self.assertTrue(not all(strata_samples))
