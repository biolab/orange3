import numpy as np
import sklearn.metrics
from Orange.data import DiscreteVariable


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
            return np.mean(scores)
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
        domain = results.data.domain
        if not isinstance(domain.class_var, DiscreteVariable):
            raise ValueError("AUC.compute_score expects a domain with a "
                             "(single) discrete variable")
        n_classes = len(domain.class_var.values)
        if n_classes < 2:
            raise ValueError("Class variable has less than two values")
        if target is None:
            if n_classes > 2:
                raise ValueError("Class variable has more than two values and "
                                 "target class is not specified")
            else:
                target = 1

        y = np.array(results.actual == target, dtype=int)

        return np.fromiter(
            (sklearn.metrics.roc_auc_score(y, probabilities[:, target])
             for probabilities in results.probabilities),
            dtype=np.float64, count=len(results.predicted))
