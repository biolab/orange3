import numpy as np
import sklearn.metrics as skl_metrics
from Orange.data import DiscreteVariable

__all__ = ["CA", "Precision", "Recall", "F1", "PrecisionRecallFSupport", "AUC",
           "MSE", "RMSE", "MAE", "R2"]

def calculate_weights(results):
    number_of_classes = len(results.domain.class_var.values)
    class_cases = [sum(results.actual == class_)
               for class_ in range(number_of_classes)]
    N = results.actual.shape[0]
    weights = [c * (N - c) for c in class_cases]
    wsum = sum(weights)
    if wsum == 0:
        return None
    else:
        weights_norm = [w / wsum for w in weights]
        return weights_norm


def calculate_fold_weights(results):
    number_of_classes = len(results.domain.class_var.values)
    class_cases = [sum(results.actual == class_)
               for class_ in range(number_of_classes)]
    w = float(class_cases[0])
    for c in class_cases[1:]:
        w *= c
    return w


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

        return self.scores_by_folds(results, **kwargs)

    def average(self, scores):
        if self.is_scalar:
            return np.mean(scores, axis=0)
        return NotImplementedError

    def scores_by_folds(self, results, **kwargs):
        nfolds = len(results.folds)
        nmodels = len(results.predicted)
        if self.is_scalar:
            scores = np.empty((nfolds, nmodels), dtype=np.float64)
            fold_weights = np.empty((nfolds, nmodels), dtype=np.float64)
        else:
            scores = [None] * nfolds
            fold_weights = [None] * nfolds
        
        for fold in range(nfolds):
            fold_results = results.get_fold(fold)
            scores[fold] = self.compute_score(fold_results, **kwargs)
            fold_weights[fold] = calculate_fold_weights(fold_results)
        
        if any(np.isnan(scores)):
            fold_weights = np.array(fold_weights[~np.isnan(scores)]).reshape((-1,nmodels))
            scores = np.array(scores[~np.isnan(scores)]).reshape((-1,nmodels))
        
        fsum = np.sum(fold_weights)
        if fsum > 0:
            fold_weights /= fsum
        return np.array([np.sum(scores * fold_weights)])

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
        number_of_classes = len(results.domain.class_var.values)
        weights = calculate_weights(results)
        if weights == None:
            return np.array([np.nan])
        
        auc_array = np.zeros(shape=(number_of_classes))
        for class_ in range(number_of_classes):
            nclass = len(set(results.actual == class_))
            if nclass == 1:
                weights[class_] = 0.0
                wsum = sum(weights)
                if wsum > 0.0:
                    weights = [w/wsum for w in weights]
                else:
                    return np.array([np.nan])
            else:
                auc_array[class_] = np.mean(np.fromiter(
                (skl_metrics.roc_auc_score(results.actual == class_, predicted)
                for predicted in results.predicted == class_),
                dtype=np.float64, count=len(results.predicted)))

        return np.array([np.sum(auc_array * weights)])

    def compute_score(self, results, target=None):
        domain = results.domain
        if not isinstance(domain.class_var, DiscreteVariable):
            raise ValueError("AUC.compute_score expects a domain with a "
                             "(single) discrete variable")
        n_classes = len(domain.class_var.values)
        if n_classes < 2:
            raise ValueError("Class variable has less than two values")
        
        if len(domain.class_var.values) == 2:
            return self.from_predicted(results, skl_metrics.roc_auc_score)
        else:
            if target is None:
                return self.multi_class_auc(results)
            else:
                y = np.array(results.actual == target, dtype=int)
                return np.fromiter(
                    (sklearn.metrics.roc_auc_score(y, probabilities[:, target])
                     for probabilities in results.probabilities),
                    dtype=np.float64, count=len(results.predicted))


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
