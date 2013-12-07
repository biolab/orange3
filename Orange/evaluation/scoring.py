import numpy as np
import sklearn.metrics
from Orange.data import DiscreteVariable


class Score:
    separate_folds = False
    is_scalar = True

    def __new__(cls, results=None, **kwargs):
        self = super().__new__(cls)
        if results is not None:
            self = self.__init__(**kwargs)
            return self(results)
        else:
            return self

    def __call__(self, results, **kwargs):
        if not (self.separate_folds and results.folds):
            return self.compute_score(results, **kwargs)

        scores = self.scores_by_folds(results, **kwargs)
        return self.average(scores)

    def average(self, scores):
        if self.is_scalar:
            return np.mean(scores)
        return NotImplementedError

    def scores_by_folds(self, results, **kwargs):
        nfolds = len(self.folds)
        if self.is_scalar:
            scores = np.empty((len(results), nfolds), dtype=np.float64)
        else:
            scores = [None] * nfolds
        for fold in range(nfolds):
            fold_results = results.get_results(fold)
            scores[fold] = self.compute_score(fold_results, **kwargs)
        return scores

    def compute_score(self, results):
        return NotImplementedError

    @staticmethod
    def from_predicted(results, score_function):
        return np.fromiter(
            (score_function(actual, predicted)
             for actual, predicted in zip(results.actual, results.predicted)),
            dtype=np.float64, count=len(results))

class CA(Score):
    def compute_score(self, results):
        return self.from_predicted(results, sklearn.metrics.accuracy_score)


class Precision(Score):
    def compute_score(self, results):
        return self.from_predicted(results, sklearn.metrics.precision_score)


class Recall(Score):
    def compute_score(self, results):
        return self.from_predicted(results, sklearn.metrics.recall_score)


class F1(Score):
    def compute_score(self, results):
        return self.from_predicted(results, sklearn.metrics.f1_score)


class PrecisionRecallFSupport(Score):
    is_scalar = False

    def compute_score(self, results):
        return self.from_predicted(
            results, sklearn.metrics.precision_recall_fscore_support)


class AUC(Score):
    separate_folds = True

    def compute_score(self, results, target=None):
        if not isinstance(results.domain.class_var, DiscreteVariable):
            raise ValueError("AUC.compute_score expects a domain with a "
                             "(single) discrete variable")
        n_classes = len(results.domain.class_var.values)
        if n_classes < 2:
            raise ValueError("Class variable has less than two values")
        if target is None:
            if n_classes > 2:
                raise ValueError("Class variable has more than two values and "
                                 "target class is not specified")
            else:
                target = 1
        return np.fromiter(
            (sklearn.metrics.roc_auc_score(actual, probabilities[target])
             for actual, probabilities in zip(results.actual,
                                              results.probabilities)),
            dtype=np.float64, count=len(results))

