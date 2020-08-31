import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _SigmoidCalibration

from Orange.classification import Model, Learner
from Orange.evaluation import TestOnTrainingData
from Orange.evaluation.performance_curves import Curves

__all__ = ["ThresholdClassifier", "ThresholdLearner",
           "CalibratedLearner", "CalibratedClassifier"]


class ThresholdClassifier(Model):
    """
    A model that wraps a binary model and sets a different threshold.

    The target class is the class with index 1. A data instances is classified
    to class 1 it the probability of this class equals or exceeds the threshold

    Attributes:
        base_model (Orange.classification.Model): base mode
        threshold (float): decision threshold
    """
    def __init__(self, base_model, threshold):
        if not base_model.domain.class_var.is_discrete \
                or len(base_model.domain.class_var.values) != 2:
            raise ValueError("ThresholdClassifier requires a binary class")

        super().__init__(base_model.domain, base_model.original_domain)
        self.name = f"{base_model.name}, thresh={threshold:.2f}"
        self.base_model = base_model
        self.threshold = threshold

    def __call__(self, data, ret=Model.Value):
        probs = self.base_model(data, ret=Model.Probs)
        if ret == Model.Probs:
            return probs
        class_probs = probs[:, 1].ravel()
        with np.errstate(invalid="ignore"):  # we fix nanx below
            vals = (class_probs >= self.threshold).astype(float)
        vals[np.isnan(class_probs)] = np.nan
        if ret == Model.Value:
            return vals
        else:
            return vals, probs


class ThresholdLearner(Learner):
    """
    A learner that runs another learner and then finds the optimal threshold
    for CA or F1 on the training data.

    Attributes:
        base_leaner (Learner): base learner
        threshold_criterion (int):
            `ThresholdLearner.OptimizeCA` or `ThresholdLearner.OptimizeF1`
    """
    __returns__ = ThresholdClassifier

    OptimizeCA, OptimizeF1 = range(2)

    def __init__(self, base_learner, threshold_criterion=OptimizeCA):
        super().__init__()
        self.base_learner = base_learner
        self.threshold_criterion = threshold_criterion

    def fit_storage(self, data):
        """
        Induce a model using the provided `base_learner`, compute probabilities
        on training data and the find the optimal decision thresholds. In case
        of ties, select the threshold that is closest to 0.5.
        """
        if not data.domain.class_var.is_discrete \
                or len(data.domain.class_var.values) != 2:
            raise ValueError("ThresholdLearner requires a binary class")

        res = TestOnTrainingData(store_models=True)(data, [self.base_learner])
        model = res.models[0, 0]
        curves = Curves.from_results(res)
        curve = [curves.ca, curves.f1][self.threshold_criterion]()
        # In case of ties, we want the optimal threshold that is closest to 0.5
        best_threshs = curves.probs[curve == np.max(curve)]
        threshold = best_threshs[min(np.searchsorted(best_threshs, 0.5),
                                     len(best_threshs) - 1)]
        return ThresholdClassifier(model, threshold)


class CalibratedClassifier(Model):
    """
    A model that wraps another model and recalibrates probabilities

    Attributes:
        base_model (Mode): base mode
        calibrators (list of callable):
            list of functions that get a vector of probabilities and return
            calibrated probabilities
    """
    def __init__(self, base_model, calibrators):
        if not base_model.domain.class_var.is_discrete:
            raise ValueError("CalibratedClassifier requires a discrete target")

        super().__init__(base_model.domain, base_model.original_domain)
        self.base_model = base_model
        self.calibrators = calibrators
        self.name = f"{base_model.name}, calibrated"

    def __call__(self, data, ret=Model.Value):
        probs = self.base_model(data, Model.Probs)
        cal_probs = self.calibrated_probs(probs)
        if ret == Model.Probs:
            return cal_probs
        vals = np.argmax(cal_probs, axis=1)
        if ret == Model.Value:
            return vals
        else:
            return vals, cal_probs

    def calibrated_probs(self, probs):
        if self.calibrators:
            ps = np.hstack(
                tuple(
                    calibr.predict(cls_probs).reshape(-1, 1)
                    for calibr, cls_probs in zip(self.calibrators, probs.T)))
        else:
            ps = probs.copy()
        sums = np.sum(ps, axis=1)
        zero_sums = sums == 0
        with np.errstate(invalid="ignore"):  # handled below
            ps /= sums[:, None]
        if zero_sums.any():
            ps[zero_sums] = 1 / ps.shape[1]
        return ps


class CalibratedLearner(Learner):
    """
    Probability calibration for learning algorithms

    This learner that wraps another learner, so that after training, it predicts
    the probabilities on training data and calibrates them using sigmoid or
    isotonic calibration. It then returns a :obj:`CalibratedClassifier`.

    Attributes:
        base_learner (Learner): base learner
        calibration_method (int):
            `CalibratedLearner.Sigmoid` or `CalibratedLearner.Isotonic`
    """
    __returns__ = CalibratedClassifier

    Sigmoid, Isotonic = range(2)

    def __init__(self, base_learner, calibration_method=Sigmoid):
        super().__init__()
        self.base_learner = base_learner
        self.calibration_method = calibration_method

    def fit_storage(self, data):
        """
        Induce a model using the provided `base_learner`, compute probabilities
        on training data and use scipy's `_SigmoidCalibration` or
        `IsotonicRegression` to prepare calibrators.
        """
        res = TestOnTrainingData(store_models=True)(data, [self.base_learner])
        model = res.models[0, 0]
        probabilities = res.probabilities[0]
        return self.get_model(model, res.actual, probabilities)

    def get_model(self, model, ytrue, probabilities):
        if self.calibration_method == CalibratedLearner.Sigmoid:
            fitter = _SigmoidCalibration()
        else:
            fitter = IsotonicRegression(out_of_bounds='clip')
        probabilities[np.isinf(probabilities)] = 1
        calibrators = [fitter.fit(cls_probs, ytrue)
                       for cls_idx, cls_probs in enumerate(probabilities.T)]
        return CalibratedClassifier(model, calibrators)
