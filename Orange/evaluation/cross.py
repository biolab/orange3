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
        n_class = len(self.tab.domain.class_var.values)
        values = np.empty(n)
        probs = np.empty((n,n_class))
        kf = cross_validation.KFold(n, k)
        for train_index, test_index in kf:
            train = self.tab.new_from_table_rows(self.tab, train_index)
            test = self.tab.new_from_table_rows(self.tab, test_index)
            model = self.learner(train)
            val, prob = model(test.X, Model.ValueProbs)
            values[test_index] = val
            probs[test_index] = prob

        return values, probs

