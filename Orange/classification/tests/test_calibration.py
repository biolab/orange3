import unittest
from unittest.mock import Mock, patch

import numpy as np

from Orange.base import Model
from Orange.classification import LogisticRegressionLearner
from Orange.classification.calibration import \
    ThresholdLearner, ThresholdClassifier, \
    CalibratedLearner, CalibratedClassifier
from Orange.data import Table


class TestThresholdClassifier(unittest.TestCase):
    def setUp(self):
        probs1 = np.array([0.3, 0.5, 0.2, 0.8, 0.9, 0]).reshape(-1, 1)
        self.probs = np.hstack((1 - probs1, probs1))
        base_model = Mock(return_value=self.probs)
        base_model.domain.class_var.is_discrete = True
        base_model.domain.class_var.values = ["a", "b"]
        self.model = ThresholdClassifier(base_model, 0.5)
        self.data = Mock()

    def test_threshold(self):
        vals = self.model(self.data)
        np.testing.assert_equal(vals, [0, 1, 0, 1, 1, 0])

        self.model.threshold = 0.8
        vals = self.model(self.data)
        np.testing.assert_equal(vals, [0, 0, 0, 1, 1, 0])

        self.model.threshold = 0
        vals = self.model(self.data)
        np.testing.assert_equal(vals, [1] * 6)

    def test_return_types(self):
        vals = self.model(self.data, ret=Model.Value)
        np.testing.assert_equal(vals, [0, 1, 0, 1, 1, 0])

        vals = self.model(self.data)
        np.testing.assert_equal(vals, [0, 1, 0, 1, 1, 0])

        probs = self.model(self.data, ret=Model.Probs)
        np.testing.assert_equal(probs, self.probs)

        vals, probs = self.model(self.data, ret=Model.ValueProbs)
        np.testing.assert_equal(vals, [0, 1, 0, 1, 1, 0])
        np.testing.assert_equal(probs, self.probs)

    def test_nans(self):
        self.probs[1, :] = np.nan
        vals, probs = self.model(self.data, ret=Model.ValueProbs)
        np.testing.assert_equal(vals, [0, np.nan, 0, 1, 1, 0])
        np.testing.assert_equal(probs, self.probs)

    def test_non_binary_base(self):
        base_model = Mock()
        base_model.domain.class_var.is_discrete = True
        base_model.domain.class_var.values = ["a"]
        self.assertRaises(ValueError, ThresholdClassifier, base_model, 0.5)

        base_model.domain.class_var.values = ["a", "b", "c"]
        self.assertRaises(ValueError, ThresholdClassifier, base_model, 0.5)

        base_model.domain.class_var = Mock()
        base_model.domain.class_var.is_discrete = False
        self.assertRaises(ValueError, ThresholdClassifier, base_model, 0.5)

    def test_np_data(self):
        """
        Test ThresholdModel with numpy data.
        When passing numpy data to model they should be already
        transformed to models domain since model do not know how to do it.
        """
        data = Table('heart_disease')
        base_learner = LogisticRegressionLearner()
        model = ThresholdLearner(base_learner)(data)
        res = model(model.data_to_model_domain(data).X)
        self.assertTupleEqual((len(data),), res.shape)


class TestThresholdLearner(unittest.TestCase):
    @patch("Orange.evaluation.performance_curves.Curves.from_results")
    @patch("Orange.classification.calibration.TestOnTrainingData")
    def test_fit_storage(self, test_on_training, curves_from_results):
        curves_from_results.return_value = curves = Mock()
        curves.probs = np.array([0.1, 0.15, 0.3, 0.45, 0.6, 0.8])
        curves.ca = lambda: np.array([0.1, 0.7, 0.4, 0.4, 0.3, 0.1])
        curves.f1 = lambda: np.array([0.1, 0.2, 0.4, 0.4, 0.3, 0.1])
        model = Mock()
        model.domain.class_var.is_discrete = True
        model.domain.class_var.values = ("a", "b")
        data = Table("heart_disease")
        learner = Mock()
        test_on_training.return_value = tot = Mock()
        res = Mock()
        res.models = np.array([[model]])
        tot.return_value = res

        thresh_learner = ThresholdLearner(
            base_learner=learner,
            threshold_criterion=ThresholdLearner.OptimizeCA)
        thresh_model = thresh_learner(data)
        self.assertEqual(thresh_model.threshold, 0.15)
        args, _ = tot.call_args  # pylint: disable=unpacking-non-sequence
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], data)
        self.assertIs(args[1][0], learner)

        _, kwargs = test_on_training.call_args
        self.assertEqual(len(args[1]), 1)
        self.assertEqual(kwargs, {"store_models": 1})

        thresh_learner = ThresholdLearner(
            base_learner=learner,
            threshold_criterion=ThresholdLearner.OptimizeF1)
        thresh_model = thresh_learner(data)
        self.assertEqual(thresh_model.threshold, 0.45)

    def test_non_binary_class(self):
        thresh_learner = ThresholdLearner(
            base_learner=Mock(),
            threshold_criterion=ThresholdLearner.OptimizeF1)

        data = Mock()
        data.domain.class_var.is_discrete = True
        data.domain.class_var.values = ["a"]
        self.assertRaises(ValueError, thresh_learner.fit_storage, data)

        data.domain.class_var.values = ["a", "b", "c"]
        self.assertRaises(ValueError, thresh_learner.fit_storage, data)

        data.domain.class_var = Mock()
        data.domain.class_var.is_discrete = False
        self.assertRaises(ValueError, thresh_learner.fit_storage, data)


class TestCalibratedClassifier(unittest.TestCase):
    def setUp(self):
        probs1 = np.array([0.3, 0.5, 0.2, 0.8, 0.9, 0]).reshape(-1, 1)
        self.probs = np.hstack((1 - probs1, probs1))
        base_model = Mock(return_value=self.probs)
        base_model.domain.class_var.is_discrete = True
        base_model.domain.class_var.values = ["a", "b"]
        self.model = CalibratedClassifier(base_model, None)
        self.data = Mock()

    def test_call(self):
        calprobs = np.arange(self.probs.size).reshape(self.probs.shape)
        calprobs = calprobs / np.sum(calprobs, axis=1)[:, None]
        calprobs[-1] = [0.7, 0.3]
        self.model.calibrated_probs = Mock(return_value=calprobs)

        probs = self.model(self.data, ret=Model.Probs)
        self.model.calibrated_probs.assert_called_with(self.probs)
        np.testing.assert_almost_equal(probs, calprobs)

        vals = self.model(self.data, ret=Model.Value)
        np.testing.assert_almost_equal(vals, [1, 1, 1, 1, 1, 0])

        vals, probs = self.model(self.data, ret=Model.ValueProbs)
        np.testing.assert_almost_equal(probs, calprobs)
        np.testing.assert_almost_equal(vals, [1, 1, 1, 1, 1, 0])

    def test_calibrated_probs(self):
        self.model.calibrators = None
        calprobs = self.model.calibrated_probs(self.probs)
        np.testing.assert_equal(calprobs, self.probs)
        self.assertIsNot(calprobs, self.probs)

        calibrator = Mock()
        calibrator.predict = lambda x: x**2
        self.model.calibrators = [calibrator] * 2
        calprobs = self.model.calibrated_probs(self.probs)
        expprobs = self.probs ** 2 / np.sum(self.probs ** 2, axis=1)[:, None]
        np.testing.assert_almost_equal(calprobs, expprobs)

        self.probs[1] = 0
        self.probs[2] = np.nan
        expprobs[1] = 0.5
        expprobs[2] = np.nan
        calprobs = self.model.calibrated_probs(self.probs)
        np.testing.assert_almost_equal(calprobs, expprobs)

    def test_np_data(self):
        """
        Test CalibratedClassifier with numpy data.
        When passing numpy data to model they should be already
        transformed to models domain since model do not know how to do it.
        """
        data = Table('heart_disease')
        base_learner = LogisticRegressionLearner()
        model = CalibratedLearner(base_learner)(data)
        res = model(model.data_to_model_domain(data).X)
        self.assertTupleEqual((len(data),), res.shape)


class TestCalibratedLearner(unittest.TestCase):
    @patch("Orange.classification.calibration._SigmoidCalibration.fit")
    @patch("Orange.classification.calibration.TestOnTrainingData")
    def test_fit_storage(self, test_on_training, sigmoid_fit):
        data = Table("heart_disease")
        learner = Mock()

        model = Mock()
        model.domain.class_var.is_discrete = True
        model.domain.class_var.values = ("a", "b")

        test_on_training.return_value = tot = Mock()
        res = Mock()
        res.models = np.array([[model]])
        res.probabilities = np.arange(20, dtype=float).reshape(1, 5, 4)
        tot.return_value = res

        sigmoid_fit.return_value = Mock()

        cal_learner = CalibratedLearner(
            base_learner=learner, calibration_method=CalibratedLearner.Sigmoid)
        cal_model = cal_learner(data)

        self.assertIs(cal_model.base_model, model)
        self.assertEqual(cal_model.calibrators, [sigmoid_fit.return_value] * 4)
        args, _ = tot.call_args  # pylint: disable=unpacking-non-sequence
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], data)
        self.assertIs(args[1][0], learner)
        self.assertEqual(len(args[1]), 1)

        _, kwargs = test_on_training.call_args
        self.assertEqual(kwargs, {"store_models": 1})

        for call, cls_probs in zip(sigmoid_fit.call_args_list,
                                   res.probabilities[0].T):
            np.testing.assert_equal(call[0][0], cls_probs)


if __name__ == "__main__":
    unittest.main()
