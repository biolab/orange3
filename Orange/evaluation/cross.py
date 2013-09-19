import numpy as np
from sklearn import cross_validation
from Orange import data
from Orange import classification
from Orange.classification import Model
from Orange.data import domain as orange_domain


class CrossValidation:

    def __init__(self, tab, fit):
        assert(isinstance(tab, data.Table))
        assert(isinstance(fit, classification.Fitter))
        self.tab = tab
        self.learner = fit

    def KFold(self, k):
        n = len(self.tab)
        values = np.empty(n)
        get_probs = isinstance(self.tab.domain.class_var,
                               data.DiscreteVariable)
        if get_probs:
            probs = np.empty((n, len(self.tab.domain.class_var.values)))
        kf = cross_validation.KFold(n, k)
        for train_index, test_index in kf:
            train = self.tab.from_table_rows(self.tab, train_index)
            test = self.tab.from_table_rows(self.tab, test_index)
            model = self.learner(train)
            if get_probs:
                values[test_index], probs[test_index] = model(test.X,
                                                              Model.ValueProbs)
            else:
                values[test_index] = model(test.X)
        if get_probs:
            return values, probs
        else:
            return values
