"""
Methods for scoring prediction results (CA, AUC, ...).

Examples
--------
>>> import Orange
>>> data = Orange.data.Table('iris')
>>> learner = Orange.classification.LogisticRegressionLearner(solver="liblinear")
>>> results = Orange.evaluation.TestOnTrainingData(data, [learner])

"""

import math

import numpy as np
import sklearn.metrics as skl_metrics
from sklearn.metrics import confusion_matrix

from Orange.data import DiscreteVariable, ContinuousVariable, Domain
from Orange.misc.wrapper_meta import WrapperMeta

__all__ = ["CA", "Precision", "Recall", "F1", "PrecisionRecallFSupport", "AUC",
           "MSE", "RMSE", "MAE", "MAPE", "R2", "LogLoss", "MatthewsCorrCoefficient"]


class ScoreMetaType(WrapperMeta):
    """
    Maintain a registry of non-abstract subclasses and assign the default
    value of `name`.

    The existing meta class Registry cannot be used since a meta class cannot
    have multiple inherited __new__ methods."""
    def __new__(mcs, name, bases, dict_, **kwargs):
        cls = WrapperMeta.__new__(mcs, name, bases, dict_)
        # Essentially `if cls is not Score`, except that Score may not exist yet
        if hasattr(cls, "registry"):
            if not kwargs.get("abstract"):
                # Don't use inherited names, look into dict_
                cls.name = dict_.get("name", name)
                cls.long_name = dict_.get("long_name", cls.name)
                cls.registry[name] = cls
        else:
            cls.registry = {}
        return cls

    def __init__(cls, *args, **_):
        WrapperMeta.__init__(cls, *args)


class Score(metaclass=ScoreMetaType):
    """
    ${sklpar}
    Parameters
    ----------
    results : Orange.evaluation.Results
        Stored predictions and actual data in model testing.
    """
    __wraps__ = None

    separate_folds = False
    is_scalar = True
    is_binary = False  #: If true, compute_score accepts `target` and `average`
    #: If the class doesn't explicitly contain `abstract=True`, it is not
    #: abstract; essentially, this attribute is not inherited
    abstract = True
    class_types = ()
    name = None
    long_name = None  #: A short user-readable name (e.g. a few words)

    default_visible = True
    priority = 100

    def __new__(cls, results=None, **kwargs):
        self = super().__new__(cls)
        if results is not None:
            self.__init__()
            return self(results, **kwargs)
        else:
            return self

    def __call__(self, results, **kwargs):
        if self.separate_folds and results.score_by_folds and results.folds:
            scores = self.scores_by_folds(results, **kwargs)
            return self.average(scores)

        return self.compute_score(results, **kwargs)

    def average(self, scores):
        if self.is_scalar:
            return np.mean(scores, axis=0)
        return NotImplementedError

    def scores_by_folds(self, results, **kwargs):
        nfolds = len(results.folds)
        nmodels = len(results.predicted)
        if self.is_scalar:
            scores = np.empty((nfolds, nmodels), dtype=np.float64)
        else:
            scores = [None] * nfolds
        for fold in range(nfolds):
            fold_results = results.get_fold(fold)
            scores[fold] = self.compute_score(fold_results, **kwargs)
        return scores

    def compute_score(self, results):
        wraps = type(self).__wraps__  # self.__wraps__ is invisible
        if wraps:
            return self.from_predicted(results, wraps)
        else:
            return NotImplementedError

    @staticmethod
    def from_predicted(results, score_function, **kwargs):
        return np.fromiter(
            (score_function(results.actual, predicted, **kwargs)
             for predicted in results.predicted),
            dtype=np.float64, count=len(results.predicted))

    @staticmethod
    def is_compatible(domain: Domain) -> bool:
        raise NotImplementedError


class ClassificationScore(Score, abstract=True):
    class_types = (DiscreteVariable, )

    @staticmethod
    def is_compatible(domain: Domain) -> bool:
        return domain.has_discrete_class


class RegressionScore(Score, abstract=True):
    class_types = (ContinuousVariable, )

    @staticmethod
    def is_compatible(domain: Domain) -> bool:
        return domain.has_continuous_class


# pylint: disable=invalid-name
class CA(ClassificationScore):
    __wraps__ = skl_metrics.accuracy_score
    name = "CA"
    long_name = "Classification accuracy"
    priority = 20


class PrecisionRecallFSupport(ClassificationScore):
    __wraps__ = skl_metrics.precision_recall_fscore_support
    is_scalar = False


class TargetScore(ClassificationScore):
    """
    Base class for scorers that need a target value (a "positive" class).

    Parameters
    ----------
    results : Orange.evaluation.Results
        Stored predictions and actual data in model testing.

    target : int, optional (default=None)
        Target class value.
        When None:
          - if averaging is specified, use all classes and average results
          - if average is 'binary' and class variable has exactly 2 values,
            use the value '1' as the positive class

    average: str, method for averaging (default='binary')
        Default requires a binary class or target to be set.
        Options: 'weighted', 'macro', 'micro', None

    """
    is_binary = True
    abstract = True
    __wraps__ = None  # Subclasses should set the scoring function

    def compute_score(self, results, target=None, average='binary'):
        if average == 'binary':
            if target is None:
                if len(results.domain.class_var.values) > 2:
                    raise ValueError(
                        "Multiclass data: specify target class or select "
                        "averaging ('weighted', 'macro', 'micro')")
                target = 1  # Default: use 1 as "positive" class
            average = None
        labels = None if target is None else [target]
        return self.from_predicted(
            results, type(self).__wraps__, labels=labels, average=average)


class Precision(TargetScore):
    __wraps__ = skl_metrics.precision_score
    name = "Prec"
    long_name = "Precision"
    priority = 40


class Recall(TargetScore):
    __wraps__ = skl_metrics.recall_score
    name = long_name = "Recall"
    priority = 50


class F1(TargetScore):
    __wraps__ = skl_metrics.f1_score
    name = long_name = "F1"
    priority = 30


class AUC(ClassificationScore):
    """
    ${sklpar}

    Parameters
    ----------
    results : Orange.evaluation.Results
        Stored predictions and actual data in model testing.

    target : int, optional (default=None)
        Value of class to report.
    """
    __wraps__ = skl_metrics.roc_auc_score
    separate_folds = True
    is_binary = True
    name = "AUC"
    long_name = "Area under ROC curve"
    priority = 10

    @staticmethod
    def calculate_weights(results):
        classes = np.unique(results.actual)
        class_cases = [sum(results.actual == class_)
                       for class_ in classes]
        N = results.actual.shape[0]
        weights = np.array([c * (N - c) for c in class_cases])
        wsum = np.sum(weights)
        if wsum == 0:
            raise ValueError("Class variable has less than two values")
        else:
            return weights / wsum

    @staticmethod
    def single_class_auc(results, target):
        y = np.array(results.actual == target, dtype=int)
        return np.fromiter(
            (skl_metrics.roc_auc_score(y, probabilities[:, int(target)])
             for probabilities in results.probabilities),
            dtype=np.float64, count=len(results.predicted))

    def multi_class_auc(self, results):
        classes = np.unique(results.actual)
        weights = self.calculate_weights(results)
        auc_array = np.array([self.single_class_auc(results, class_)
                              for class_ in classes])
        return np.sum(auc_array.T * weights, axis=1)

    def compute_score(self, results, target=None, average=None):
        domain = results.domain
        n_classes = len(domain.class_var.values)

        if n_classes < 2:
            raise ValueError("Class variable has less than two values")
        elif n_classes == 2:
            return self.single_class_auc(results, 1)
        else:
            if target is None:
                return self.multi_class_auc(results)
            else:
                return self.single_class_auc(results, target)


class LogLoss(ClassificationScore):
    """
    ${sklpar}

    Parameters
    ----------
    results : Orange.evaluation.Results
        Stored predictions and actual data in model testing.

    eps : float
        Log loss is undefined for p=0 or p=1, so probabilities are
        clipped to max(eps, min(1 - eps, p)).

    normalize : bool, optional (default=True)
        If true, return the mean loss per sample.
        Otherwise, return the sum of the per-sample losses.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Examples
    --------
    >>> Orange.evaluation.LogLoss(results)
    array([0.3...])

    """
    __wraps__ = skl_metrics.log_loss
    priority = 120
    name = "LogLoss"
    long_name = "Logistic loss"
    default_visible = False

    def compute_score(self, results, eps=1e-15, normalize=True,
                      sample_weight=None):
        return np.fromiter(
            (skl_metrics.log_loss(results.actual,
                                  probabilities,
                                  eps=eps,
                                  normalize=normalize,
                                  sample_weight=sample_weight)
             for probabilities in results.probabilities),
            dtype=np.float64, count=len(results.probabilities))


class Specificity(ClassificationScore):
    is_binary = True
    priority = 110
    name = "Spec"
    long_name = "Specificity"
    default_visible = False

    @staticmethod
    def calculate_weights(results):
        classes, counts = np.unique(results.actual, return_counts=True)
        n = np.array(results.actual).shape[0]
        return counts / n, classes

    @staticmethod
    def specificity(y_true, y_pred):
        tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)

    def single_class_specificity(self, results, target):
        y_true = (np.array(results.actual) == target).astype(int)
        return np.fromiter(
            (self.specificity(y_true,
                              np.array(predicted == target, dtype=int))
             for predicted in results.predicted),
            dtype=np.float64, count=len(results.predicted))

    def multi_class_specificity(self, results):
        weights, classes = self.calculate_weights(results)
        scores = np.array([self.single_class_specificity(results, class_)
                           for class_ in classes])
        return np.sum(scores.T * weights, axis=1)

    def compute_score(self, results, target=None, average="binary"):
        domain = results.domain
        n_classes = len(domain.class_var.values)

        if target is None:
            if average == "weighted":
                return self.multi_class_specificity(results)
            elif average == "binary":  # average is binary
                if n_classes != 2:
                    raise ValueError(
                        "Binary averaging needs two classes in data: "
                        "specify target class or use "
                        "weighted averaging.")
                return self.single_class_specificity(results, 1)
            else:
                raise ValueError(
                    "Wrong parameters: For averaging select one of the "
                    "following values: ('weighted', 'binary')")
        elif target is not None:
            return self.single_class_specificity(results, target)


class MatthewsCorrCoefficient(ClassificationScore):
    __wraps__ = skl_metrics.matthews_corrcoef
    name = "MCC"
    long_name = "Matthews correlation coefficient"


# Regression scores


class MSE(RegressionScore):
    __wraps__ = skl_metrics.mean_squared_error
    name = "MSE"
    long_name = "Mean square error"
    priority = 20


class RMSE(RegressionScore):
    name = "RMSE"
    long_name = "Root mean square error"

    def compute_score(self, results):
        return np.sqrt(MSE(results))
    priority = 30


class MAE(RegressionScore):
    __wraps__ = skl_metrics.mean_absolute_error
    name = "MAE"
    long_name = "Mean absolute error"
    priority = 40

class MAPE(RegressionScore):
    __wraps__ = skl_metrics.mean_absolute_percentage_error
    name = "MAPE"
    long_name = "Mean absolute percentage error"
    priority = 45

# pylint: disable=invalid-name
class R2(RegressionScore):
    __wraps__ = skl_metrics.r2_score
    name = "R2"
    long_name = "Coefficient of determination"
    priority = 50


class CVRMSE(RegressionScore):
    name = "CVRMSE"
    long_name = "Coefficient of variation of the RMSE"
    priority = 110
    default_visible = False

    def compute_score(self, results):
        mean = np.nanmean(results.actual)
        if mean < 1e-10:
            raise ValueError("Mean value is too small")
        return RMSE(results) / mean * 100
