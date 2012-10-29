import numpy as np
from sklearn import cross_validation
from Orange import data
from Orange import classification
from Orange.classification import Model

class CrossValidation:

    def __init__(self, tab, fit):
        assert(isinstance(tab, data.Table))
        assert(isinstance(fit, classification.Fitter))
        self.tab = tab
        self.learner = fit

    def KFold(self, k):
        n = len(self.tab)
        predictions = np.ndarray((n,1))

        kf = cross_validation.KFold(n, k)
        for train_index, test_index in kf:
            train = self.tab.new_from_table_rows(self.tab, train_index)
            test = self.tab.new_from_table_rows(self.tab, test_index)
            model = self.learner(train)
            fold_predictions = model(test.X, Model.ValueProbs)
            predictions[test_index] = fold_predictions

        return predictions

