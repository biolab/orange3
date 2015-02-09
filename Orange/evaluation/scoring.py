import numpy as np
import sklearn.metrics as skl_metrics
from Orange.data import DiscreteVariable

__all__ = ["CA", "Precision", "Recall", "F1", "PrecisionRecallFSupport", "AUC",
           "MSE", "RMSE", "MAE", "R2"]


class Score:
    separate_folds = False
    is_scalar = True

    def __new__(cls, results=None, **kwargs):
        self = super().__new__(cls)
        if results is not None:
            self.__init__()
            return self(results, **kwargs)
        else:
            return self

    def __call__(self, results, **kwargs):
        if not (self.separate_folds and results.folds):
            return self.compute_score(results, **kwargs)

        scores = self.scores_by_folds(results, **kwargs)
        return self.average(scores)

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
        return NotImplementedError

    @staticmethod
    def from_predicted(results, score_function):
        return np.fromiter(
            (score_function(results.actual, predicted)
             for predicted in results.predicted),
            dtype=np.float64, count=len(results.predicted))


## Classification scores

class CA(Score):
    def compute_score(self, results):
        return self.from_predicted(results, skl_metrics.accuracy_score)


class Precision(Score):
    def compute_score(self, results):
        return self.from_predicted(results, skl_metrics.precision_score)


class Recall(Score):
    def compute_score(self, results):
        return self.from_predicted(results, skl_metrics.recall_score)


class F1(Score):
    def compute_score(self, results):
        return self.from_predicted(results, skl_metrics.f1_score)


class PrecisionRecallFSupport(Score):
    is_scalar = False

    def compute_score(self, results):
        return self.from_predicted(
            results, skl_metrics.precision_recall_fscore_support)


class AUC(Score):
    separate_folds = True

    def multi_class_auc(self, results):
        number_of_classes = len(results.data.domain.class_var.values)
        N = results.actual.shape[0]

        class_cases = [sum(results.actual == class_)
                   for class_ in range(number_of_classes)]
        weights = [c * (N - c) for c in class_cases]
        weights_norm = [w / sum(weights) for w in weights]

        auc_array = np.array([np.mean(np.fromiter(
            (skl_metrics.roc_auc_score(results.actual == class_, predicted)
            for predicted in results.predicted == class_),
            dtype=np.float64, count=len(results.predicted)))
            for class_ in range(number_of_classes)])

        return np.array([np.sum(auc_array * weights_norm)])

    def compute_score(self, results):
        if len(results.data.domain.class_var.values) == 2:
            return self.from_predicted(results, skl_metrics.roc_auc_score)
        else:
            return self.multi_class_auc(results)


## Regression scores

class MSE(Score):
    def compute_score(self, results):
        return self.from_predicted(results, skl_metrics.mean_squared_error)


class RMSE(Score):
    def compute_score(self, results):
        return np.sqrt(MSE(results))


class MAE(Score):
    def compute_score(self, results):
        return self.from_predicted(results, skl_metrics.mean_absolute_error)


class R2(Score):
    def compute_score(self, results):
        return self.from_predicted(results, skl_metrics.r2_score)
