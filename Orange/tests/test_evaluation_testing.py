# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
from unittest.mock import Mock, patch

import numpy as np

from Orange.classification import NaiveBayesLearner, MajorityLearner
from Orange.evaluation.testing import Validation
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
    table = Table.from_numpy(None, x, y)
    table = preprocess.Discretize(discretize.EqualWidth(n=3))(table)
    return table


class TestingTestCase(unittest.TestCase):
    def test_no_data(self):
        self.assertRaises(ValueError, CrossValidation,
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

    def setUp(self):
        self.random_table = random_data(self.nrows, self.ncols)

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
        res = method()(random_data(50, 4), [major, fails, major])
        self.assertFalse(res.failed[0])
        self.assertIsInstance(res.failed[1], Exception)
        self.assertFalse(res.failed[2])
        self.assertEqual(major_call, succ_calls)
        self.assertEqual(fail_calls, succ_calls / 2)

    def run_test_callback(self, method, expected_progresses):
        def record_progress(p):
            progress.append(p)
        progress = []
        method()(
            self.random_table, [MajorityLearner(), MajorityLearner()],
            callback=record_progress)
        np.testing.assert_almost_equal(np.array(progress), expected_progresses)

    def run_test_preprocessor(self, method, expected_sizes):
        def preprocessor(data):
            data_sizes.append(len(data))
            return data
        data_sizes = []
        method()(Table('iris'), [MajorityLearner(), MajorityLearner()],
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

    @staticmethod
    def _callback_values(iterations):
        return np.hstack((np.linspace(0., .99, iterations + 1)[1:], [1]))


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.data = Table('iris')

    def test_invalid_argument_combination(self):
        self.assertRaises(ValueError, Validation, self.data)
        self.assertRaises(ValueError, Validation, None, [MajorityLearner()])
        self.assertRaises(ValueError, Validation, preprocessor=lambda x: x)
        self.assertRaises(ValueError, Validation, callback=lambda x: x)

    @patch("Orange.evaluation.testing.Validation.__call__")
    def test_warn_deprecations(self, _):
        self.assertWarns(
            DeprecationWarning,
            Validation, self.data, [MajorityLearner()])

        self.assertWarns(DeprecationWarning, Validation().fit)

    @patch("Orange.evaluation.testing.Validation.__call__")
    def test_obsolete_call_constructor(self, validation_call):

        class MockValidation(Validation):
            args = kwargs = None

            def __init__(self, *args, **kwargs):
                super().__init__()
                MockValidation.args = args
                MockValidation.kwargs = kwargs

            def get_indices(self, data):
                pass

        data = self.data
        learners = [MajorityLearner(), MajorityLearner()]
        kwargs = dict(foo=42, store_data=43, store_models=44, callback=45, n_jobs=46)
        self.assertWarns(
            DeprecationWarning,
            MockValidation, data, learners=learners,
            **kwargs)
        self.assertEqual(MockValidation.args, ())
        kwargs.pop("n_jobs")  # do not pass n_jobs and callback from __new__ to __init__
        kwargs.pop("callback")
        self.assertEqual(MockValidation.kwargs, kwargs)

        cargs, ckwargs = validation_call.call_args
        self.assertEqual(len(cargs), 1)
        self.assertIs(cargs[0], data)
        self.assertIs(ckwargs["learners"], learners)
        self.assertEqual(ckwargs["callback"], 45)


class TestCrossValidation(TestSampling):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table('iris')
        cls.housing = Table('housing')
        cls.nrows = 50
        cls.ncols = 5
        cls.random_table = random_data(cls.nrows, cls.ncols)

    def test_init(self):
        res = CrossValidation(k=42, stratified=False, random_state=43)
        self.assertEqual(res.k, 42)
        self.assertFalse(res.stratified)
        self.assertEqual(res.random_state, 43)

    def test_results(self):
        nrows, _ = self.random_table.X.shape
        res = CrossValidation(k=10, stratified=False)(
            self.random_table, [NaiveBayesLearner()])
        y = self.random_table.Y
        np.testing.assert_equal(res.actual, y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(res.predicted[0],
                                y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(np.argmax(res.probabilities[0], axis=1),
                                y[res.row_indices].reshape(nrows))
        self.check_folds(res, 10, nrows)

    def test_continuous(self):
        res = CrossValidation(k=3)(
            self.housing, [LinearRegressionLearner()])
        self.assertLess(RMSE(res), 5)

    def test_folds(self):
        res = CrossValidation(k=5)(self.random_table, [NaiveBayesLearner()])
        self.check_folds(res, 5, self.nrows)

    def test_call_5(self):
        nrows, _ = self.random_table.X.shape
        res = CrossValidation(k=5, stratified=False)(
            self.random_table, [NaiveBayesLearner()])
        y = self.random_table.Y
        np.testing.assert_equal(res.actual, y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(res.predicted[0],
                                y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(np.argmax(res.probabilities[0], axis=1),
                                y[res.row_indices].reshape(nrows))
        self.check_folds(res, 5, nrows)

    def test_store_data(self):
        learners = [NaiveBayesLearner()]
        res = CrossValidation(store_data=False)(self.random_table, learners)
        self.assertIsNone(res.data)

        res = CrossValidation(store_data=True)(self.random_table, learners)
        self.assertIs(res.data, self.random_table)

    def test_store_models(self):
        learners = [NaiveBayesLearner(), MajorityLearner()]

        res = CrossValidation(store_models=False, k=5)(
            self.random_table, learners)
        self.assertIsNone(res.models)

        res = CrossValidation(k=5, store_models=True)(self.random_table, learners)
        self.assertEqual(len(res.models), 5)
        self.check_models(res, learners, 5)

    def test_split_by_model(self):
        learners = [NaiveBayesLearner(), MajorityLearner()]
        res = CrossValidation(k=5, store_models=True)(self.random_table, learners)

        for i, result in enumerate(res.split_by_model()):
            self.assertIsInstance(result, Results)
            self.assertTrue((result.predicted == res.predicted[i]).all())
            self.assertTrue((result.probabilities == res.probabilities[i]).all())
            self.assertEqual(len(result.models), 5)
            for model in result.models[0]:
                self.assertIsInstance(model, learners[i].__returns__)
            self.assertSequenceEqual(result.learners, [res.learners[i]])

    def test_10_fold_probs(self):
        learners = [MajorityLearner(), MajorityLearner()]

        results = CrossValidation(k=10)(self.iris[30:130], learners)

        self.assertEqual(results.predicted.shape, (2, len(self.iris[30:130])))
        np.testing.assert_equal(results.predicted, np.ones((2, 100)))
        probs = results.probabilities
        self.assertTrue((probs[:, :, 0] < probs[:, :, 2]).all())
        self.assertTrue((probs[:, :, 2] < probs[:, :, 1]).all())

    @staticmethod
    def test_miss_majority():
        x = np.zeros((50, 3))
        y = x[:, -1]
        x[-4:] = np.ones((4, 3))
        data = Table.from_numpy(None, x, y)
        cv = CrossValidation(k=3)
        res = cv(data, [MajorityLearner()])
        np.testing.assert_equal(res.predicted[0][:49], 0)

        with data.unlocked(data.X):
            x[-4:] = np.zeros((4, 3))
        res = cv(data, [MajorityLearner()])
        np.testing.assert_equal(res.predicted[0][:49], 0)

    def test_too_many_folds(self):
        w = []
        CrossValidation(k=len(self.iris) // 2, warnings=w)(
            self.iris, [MajorityLearner()])
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
        res = CrossValidation(store_data=True)(data, [NaiveBayesLearner()])
        table = res.get_augmented_data(['Naive Bayes'])

        self.assertEqual(len(table), len(data))
        self.assertEqual(len(table.domain.attributes), len(data.domain.attributes))
        self.assertEqual(len(table.domain.class_vars), len(data.domain.class_vars))
        # +1 for class, +n_classes for probabilities, +1 for fold
        self.assertEqual(
            len(table.domain.metas), len(data.domain.metas) + 1 + n_classes + 1)
        self.assertEqual(
            table.domain.metas[len(data.domain.metas)].values, data.domain.class_var.values)

        res = CrossValidation(store_data=True)(data, [NaiveBayesLearner(), MajorityLearner()])
        table = res.get_augmented_data(['Naive Bayes', 'Majority'])

        self.assertEqual(len(table), len(data))
        self.assertEqual(len(table.domain.attributes), len(data.domain.attributes))
        self.assertEqual(len(table.domain.class_vars), len(data.domain.class_vars))
        self.assertEqual(
            len(table.domain.metas), len(data.domain.metas) + 2*(n_classes+1) + 1)
        self.assertEqual(
            table.domain.metas[len(data.domain.metas)].values, data.domain.class_var.values)
        self.assertEqual(
            table.domain.metas[len(data.domain.metas)+1].values, data.domain.class_var.values)

    def test_augmented_data_regression(self):
        data = Table("housing")
        res = CrossValidation(store_data=True)(data, [LinearRegressionLearner()])
        table = res.get_augmented_data(['Linear Regression'])

        self.assertEqual(len(table), len(data))
        self.assertEqual(len(table.domain.attributes), len(data.domain.attributes))
        self.assertEqual(len(table.domain.class_vars), len(data.domain.class_vars))
        # +1 for class, +1 for fold
        self.assertEqual(len(table.domain.metas), len(data.domain.metas) + 1 + 1)

        res = CrossValidation(store_data=True)(data, [LinearRegressionLearner(), MeanLearner()])
        table = res.get_augmented_data(['Linear Regression', 'Mean Learner'])

        self.assertEqual(len(table), len(data))
        self.assertEqual(len(table.domain.attributes), len(data.domain.attributes))
        self.assertEqual(len(table.domain.class_vars), len(data.domain.class_vars))
        # +2 for class, +1 for fold
        self.assertEqual(len(table.domain.metas), len(data.domain.metas) + 2 + 1)


class TestCrossValidationFeature(TestSampling):

    @staticmethod
    def add_meta_fold(data, f):
        fat = DiscreteVariable(name="fold", values=[str(a) for a in range(f)])
        domain = Domain(data.domain.attributes, data.domain.class_var, metas=[fat])
        ndata = data.transform(domain)
        vals = np.tile(range(f), len(data)//f + 1)[:len(data)]
        vals = vals.reshape((-1, 1))
        with ndata.unlocked(ndata.metas):
            ndata[:, fat] = vals
        return ndata

    def test_init(self):
        var = DiscreteVariable(name="fold", values="abc")
        res = CrossValidationFeature(feature=var)
        self.assertIs(res.feature, var)

    def test_call(self):
        t = self.random_table
        t = self.add_meta_fold(t, 3)
        res = CrossValidationFeature(feature=t.domain.metas[0])(t, [NaiveBayesLearner()])
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
        with t.unlocked(t.metas):
            t[0][fat] = float("nan")
        res = CrossValidationFeature(feature=fat)(t, [NaiveBayesLearner()])
        self.assertNotIn(0, res.row_indices)

    def test_bad_feature(self):
        feat = DiscreteVariable(name="fold", values="abc")
        domain = Domain([DiscreteVariable("x", values="ab")],
                        DiscreteVariable("y", values="cd"),
                        metas=[feat])
        t = Table.from_numpy(
            domain,
            np.zeros((10, 1)), np.ones((10, 1)), np.full((10, 1), np.nan))
        self.assertRaises(
            ValueError,
            CrossValidationFeature(feature=feat), t, [NaiveBayesLearner()])


class TestLeaveOneOut(TestSampling):
    def test_results(self):
        nrows = self.nrows
        t = self.random_table
        res = LeaveOneOut()(t, [NaiveBayesLearner()])
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
        res = LeaveOneOut()(t, [NaiveBayesLearner()])
        y = t.Y
        np.testing.assert_equal(res.actual, y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(res.predicted[0],
                                y[res.row_indices].reshape(nrows))
        np.testing.assert_equal(np.argmax(res.probabilities[0], axis=1),
                                y[res.row_indices].reshape(nrows))

    def test_store_data(self):
        t = self.random_table
        learners = [NaiveBayesLearner()]

        res = LeaveOneOut(store_data=False)(t, learners)
        self.assertIsNone(res.data)

        res = LeaveOneOut(store_data=True)(t, learners)
        self.assertIs(res.data, t)

    def test_store_models(self):
        t = self.random_table
        learners = [NaiveBayesLearner(), MajorityLearner()]

        res = LeaveOneOut()(t, learners)
        self.assertIsNone(res.models)

        res = LeaveOneOut(store_models=True)(t, learners)
        self.check_models(res, learners, self.nrows)

    def test_probs(self):
        data = Table('iris')[30:130]
        learners = [MajorityLearner(), MajorityLearner()]

        results = LeaveOneOut()(data, learners)

        self.assertEqual(results.predicted.shape, (2, len(data)))
        np.testing.assert_equal(results.predicted, np.ones((2, 100)))
        probs = results.probabilities
        self.assertTrue((probs[:, :, 0] < probs[:, :, 2]).all())
        self.assertTrue((probs[:, :, 2] < probs[:, :, 1]).all())

    @staticmethod
    def test_miss_majority():
        x = np.zeros((50, 3))
        y = x[:, -1]
        x[49] = 1
        data = Table.from_numpy(None, x, y)
        res = LeaveOneOut()(data, [MajorityLearner()])
        np.testing.assert_equal(res.predicted[0][:49], 0)

        with data.unlocked(data.X):
            x[49] = 0
        res = LeaveOneOut()(data, [MajorityLearner()])
        np.testing.assert_equal(res.predicted[0][:49], 0)

        with data.unlocked(data.X):
            x[25:] = 1
        data = Table.from_numpy(None, x, y)
        res = LeaveOneOut()(data, [MajorityLearner()])
        np.testing.assert_equal(res.predicted[0],
                                1 - data.Y[res.row_indices].flatten())

    def test_failed(self):
        self.run_test_failed(LeaveOneOut, 100)

    def test_callback(self):
        self.run_test_callback(LeaveOneOut, self._callback_values(2 * self.nrows))

    def test_preprocessor(self):
        self.run_test_preprocessor(LeaveOneOut, [149] * 150)


class TestTestOnTrainingData(TestSampling):
    @staticmethod
    def _callback_values(iterations):
        return np.hstack(((np.arange(iterations) + 1) / iterations, [1]))

    def test_results(self):
        nrows, _ = self.random_table.X.shape
        t = self.random_table
        res = TestOnTrainingData()(t, [NaiveBayesLearner()])
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
        res = TestOnTrainingData()(t, learners)
        self.assertIsNone(res.data)
        res = TestOnTrainingData(store_data=True)(t, learners)
        self.assertIs(res.data, t)

    def test_store_models(self):
        t = self.random_table
        learners = [NaiveBayesLearner(), MajorityLearner()]

        res = TestOnTrainingData()(t, learners)
        self.assertIsNone(res.models)

        res = TestOnTrainingData(store_models=True)(t, learners)
        self.check_models(res, learners, 1)

    def test_probs(self):
        data = self.iris[30:130]
        learners = [MajorityLearner(), MajorityLearner()]

        results = TestOnTrainingData()(data, learners)

        self.assertEqual(results.predicted.shape, (2, len(data)))
        np.testing.assert_equal(results.predicted, np.ones((2, 100)))
        probs = results.probabilities
        self.assertTrue((probs[:, :, 0] < probs[:, :, 2]).all())
        self.assertTrue((probs[:, :, 2] < probs[:, :, 1]).all())

    @staticmethod
    def test_miss_majority():
        x = np.zeros((50, 3))
        y = x[:, -1]
        x[49] = 1
        data = Table.from_numpy(None, x, y)
        res = TestOnTrainingData()(data, [MajorityLearner()])
        np.testing.assert_equal(res.predicted[0][:49], 0)

        with data.unlocked(data.X):
            x[49] = 0
        res = TestOnTrainingData()(data, [MajorityLearner()])
        np.testing.assert_equal(res.predicted[0][:49], 0)

        with data.unlocked(data.X):
            x[25:] = 1
        data = Table.from_numpy(None, x, y)
        res = TestOnTrainingData()(data, [MajorityLearner()])
        np.testing.assert_equal(res.predicted[0], res.predicted[0][0])

    def test_failed(self):
        self.run_test_failed(TestOnTrainingData, 2)

    def test_callback(self):
        self.run_test_callback(TestOnTrainingData, self._callback_values(2))

    def test_preprocessor(self):
        self.run_test_preprocessor(TestOnTrainingData, [150])


class TestTestOnTestData(TestSampling):
    @staticmethod
    def _callback_values(iterations):
        return np.hstack(((np.arange(iterations) + 1) / iterations, [1]))

    def test_results(self):
        nrows, _ = self.random_table.X.shape
        t = self.random_table
        res = TestOnTestData()(t, t, [NaiveBayesLearner()])
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
        results = TestOnTestData()(data, data, learners)

        self.assertEqual(results.predicted.shape, (2, len(data)))
        np.testing.assert_equal(results.predicted, np.ones((2, 100)))
        probs = results.probabilities
        self.assertTrue((probs[:, :, 0] < probs[:, :, 2]).all())
        self.assertTrue((probs[:, :, 2] < probs[:, :, 1]).all())

        train = self.iris[50:120]
        test = self.iris[:50]
        results = TestOnTestData()(train, test, learners)
        self.assertEqual(results.predicted.shape, (2, len(test)))
        np.testing.assert_equal(results.predicted, np.ones((2, 50)))
        probs = results.probabilities
        self.assertTrue((probs[:, :, 0] == 0).all())

    def test_store_data(self):
        data = self.random_table
        train = data[:int(self.nrows*.75)]
        test = data[int(self.nrows*.75):]
        learners = [MajorityLearner()]

        res = TestOnTestData()(train, test, learners)
        self.assertIsNone(res.data)

        res = TestOnTestData(store_data=True)(train, test, learners)
        self.assertIs(res.data, test)

    def test_store_models(self):
        data = self.random_table
        train = data[:int(self.nrows*.75)]
        test = data[int(self.nrows*.75):]
        learners = [NaiveBayesLearner(), MajorityLearner()]

        res = TestOnTestData()(train, test, learners)
        self.assertIsNone(res.models)

        res = TestOnTestData(store_models=True)(train, test, learners)
        self.check_models(res, learners, 1)

    @staticmethod
    def test_miss_majority():
        x = np.zeros((50, 3))
        y = x[:, -1]
        x[49] = 1
        data = Table.from_numpy(None, x, y)
        res = TestOnTrainingData()(data, [MajorityLearner()])
        np.testing.assert_equal(res.predicted[0][:49], 0)

        with data.unlocked(data.X):
            x[49] = 0
        res = TestOnTrainingData()(data, [MajorityLearner()])
        np.testing.assert_equal(res.predicted[0][:49], 0)

        with data.unlocked(data.X):
            x[25:] = 1
        y = x[:, -1]
        data = Table.from_numpy(None, x, y)
        res = TestOnTrainingData()(data, [MajorityLearner()])
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
        res = TestOnTestData()(data, data, [major, fails, major])
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
        TestOnTestData()(data, data, [MajorityLearner(), MajorityLearner()],
                         callback=record_progress)
        np.testing.assert_almost_equal(progress, self._callback_values(2))

    def test_preprocessor(self):
        def preprocessor(data):
            data_sizes.append(len(data))
            return data

        data_sizes = []
        data = random_data(50, 5)
        TestOnTestData()(data[:30], data[-20:],
                         [MajorityLearner(), MajorityLearner()],
                         preprocessor=preprocessor)
        self.assertEqual(data_sizes, [30])

    @patch("Orange.evaluation.testing.Validation.__new__")
    def test_train_data_argument(self, validation_new):
        data = Mock()
        test_data = Mock()
        TestOnTestData(data, test_data)
        args = validation_new.call_args[1]
        self.assertIs(args["data"], data)
        self.assertIs(args["test_data"], test_data)
        validation_new.reset_mock()

        TestOnTestData(test_data=test_data, train_data=data)
        args = validation_new.call_args[1]
        self.assertIs(args["data"], data)
        self.assertIs(args["test_data"], test_data)
        validation_new.reset_mock()

        self.assertRaises(
            ValueError,
            TestOnTestData, data, train_data=data, test_data=test_data)


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
    def test_init(self):
        res = ShuffleSplit(n_resamples=1, train_size=0.1, test_size=0.2,
                           stratified=False, random_state=42)
        self.assertEqual(res.n_resamples, 1)
        self.assertEqual(res.train_size, 0.1)
        self.assertEqual(res.test_size, 0.2)
        self.assertFalse(res.stratified)
        self.assertEqual(res.random_state, 42)

    def test_results(self):
        data = self.random_table
        train_size, n_resamples = 0.6, 10
        res = ShuffleSplit(
            train_size=train_size, test_size=1 - train_size,
            n_resamples=n_resamples)(data, [NaiveBayesLearner()])
        self.assertEqual(len(res.predicted[0]),
                         n_resamples * self.nrows * (1 - train_size))

    def test_stratified(self):
        # strata size
        n = 50
        res = ShuffleSplit(
            train_size=.5, test_size=.5, n_resamples=3, stratified=True,
            random_state=0)(self.iris, [NaiveBayesLearner()])

        for fold in res.folds:
            self.assertEqual(np.count_nonzero(res.row_indices[fold] < n), n // 2)
            self.assertEqual(np.count_nonzero(res.row_indices[fold] < 2 * n), n)

    def test_not_stratified(self):
        # strata size
        n = 50
        res = ShuffleSplit(
            train_size=.5, test_size=.5, n_resamples=3, stratified=False,
            random_state=0)(self.iris, [NaiveBayesLearner()])

        strata_samples = []
        for fold in res.folds:
            strata_samples += [
                np.count_nonzero(res.row_indices[fold] < n) == n // 2,
                np.count_nonzero(res.row_indices[fold] < 2 * n) == n]

        self.assertTrue(not all(strata_samples))


class TestResults(unittest.TestCase):
    def setUp(self):
        self.data = Table("iris")
        self.actual = np.zeros(100)
        self.row_indices = np.arange(100)
        self.folds = (range(50), range(10, 60)), (range(50, 100), range(50))
        self.learners = [MajorityLearner(), MajorityLearner()]
        self.models = np.array([[Mock(), Mock()]])
        self.predicted = np.zeros((2, 100))
        self.probabilities = np.zeros((2, 100, 3))
        self.failed = [False, True]

    def test_store_attributes(self):
        res = Results(
            self.data,
            row_indices=self.row_indices, folds=self.folds,
            score_by_folds=False, learners=self.learners, models=self.models,
            failed=self.failed, actual=self.actual, predicted=self.predicted,
            probabilities=self.probabilities, store_data=42, store_models=43)
        self.assertIs(res.data, self.data)
        self.assertEqual(res.nrows, 100)
        self.assertIs(res.row_indices, self.row_indices)
        self.assertIs(res.folds, self.folds)
        self.assertFalse(res.score_by_folds)
        self.assertIs(res.learners, self.learners)
        self.assertIs(res.models, self.models)
        self.assertIs(res.failed, self.failed)
        self.assertIs(res.actual, self.actual)
        self.assertIs(res.predicted, self.predicted)
        self.assertIs(res.probabilities, self.probabilities)

    def test_guess_sizes(self):
        res = Results(self.data, actual=self.actual)
        self.assertEqual(res.nrows, 100)
        self.assertIsNone(res.row_indices)
        self.assertIsNone(res.predicted)
        self.assertIsNone(res.probabilities)
        self.assertIsNone(res.models)
        self.assertIsNone(res.failed)

        res = Results(self.data, models=self.models)
        self.assertIsNone(res.nrows)
        self.assertIsNone(res.predicted)
        self.assertIsNone(res.probabilities)
        self.assertIs(res.models, self.models)
        self.assertEqual(res.failed, [False, False])

        res = Results(self.data, actual=self.actual, learners=self.learners)
        self.assertIs(res.data, self.data)
        self.assertIsNone(res.row_indices)
        self.assertEqual(res.nrows, 100)
        self.assertIsNone(res.folds, self.folds)
        self.assertEqual(res.predicted.shape, (2, 100))
        self.assertEqual(res.probabilities.shape, (2, 100, 3))
        self.assertEqual(res.failed, [False, False])

        res = Results(nrows=100, nmethods=2, nclasses=3)
        self.assertIsNone(res.data)
        self.assertIsNone(res.row_indices)
        self.assertIsNone(res.folds, self.folds)
        self.assertEqual(res.nrows, 100)
        self.assertEqual(res.predicted.shape, (2, 100))
        self.assertEqual(res.probabilities.shape, (2, 100, 3))
        self.assertEqual(res.failed, [False, False])

    def test_check_consistency_domain(self):
        self.assertRaises(
            ValueError,
            Results, self.data, domain=Domain([], []))

    def test_check_consistency_nrows(self):
        self.assertRaises(
            ValueError,
            Results, nrows=10, actual=self.actual)
        self.assertRaises(
            ValueError,
            Results, nrows=10, row_indices=self.row_indices)
        self.assertRaises(
            ValueError,
            Results, actual=self.actual, row_indices=self.row_indices[:5])
        self.assertRaises(
            ValueError,
            Results, nrows=10, predicted=self.predicted)
        self.assertRaises(
            ValueError,
            Results, nrows=10, probabilities=self.probabilities)

    def test_check_consistency_nclasses(self):
        self.assertRaises(
            ValueError,
            Results, self.data, nclasses=10)
        self.assertRaises(
            ValueError,
            Results, domain=self.data.domain, nclasses=10)
        self.assertRaises(
            ValueError,
            Results,
            nclasses=10,
            probabilities=self.probabilities, learners=self.learners, nrows=100)

        attributes = self.data.domain.attributes
        reg_domain = Domain(attributes[:3], attributes[3])
        self.assertRaises(
            ValueError,
            Results, nclasses=5, domain=self.data.domain)
        self.assertRaises(
            ValueError,
            Results, nclasses=5, probabilities=self.probabilities)
        self.assertRaises(
            ValueError,
            Results, nclasses=5, domain=reg_domain)
        self.assertRaises(
            ValueError,
            Results, domain=reg_domain, probabilities=self.probabilities)

    def test_check_consistency_nmethods(self):
        self.assertRaises(
            ValueError,
            Results, nmethods=10, learners=self.learners)
        self.assertRaises(
            ValueError,
            Results, nmethods=10, models=self.models)
        self.assertRaises(
            ValueError,
            Results, nmethods=10, failed=self.failed)
        self.assertRaises(
            ValueError,
            Results, nmethods=10, predicted=self.predicted)
        self.assertRaises(
            ValueError,
            Results, nmethods=10, probabilities=self.probabilities)
        self.assertRaises(
            ValueError,
            Results,
            probabilities=self.probabilities[:1], predicted=self.predicted)
