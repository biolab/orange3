# import numpy
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.preprocessing import Imputer
from numpy import isnan
import Orange.data
import Orange.classification

def replace_nan(X, imp_model):
        # Default scikit Imputer
        # Use Orange imputer when implemented
        if isnan(X).sum():
                X = imp_model.transform(X)
        return X

# TODO: implement sending a single decision tree
class RandomForestLearner(Orange.classification.SklFitter):
    def __init__(self, n_estimators=10, max_features="auto",
                 random_state=None, max_depth=3, max_leaf_nodes=5):
        self.params = vars()

    def fit(self, X, Y, W):
        self.imputer = Imputer()
        self.imputer.fit(X)
        X = replace_nan(X, self.imputer)
        rf_model = RandomForest(**self.params)
        rf_model.fit(X, Y.ravel())
        return RandomForestClassifier(rf_model, self.imputer)


class RandomForestClassifier(Orange.classification.SklModel):
    def __init__(self, clf, imp):
        self.clf = clf
        self.imputer = imp

    def predict(self, X):
        X = replace_nan(X, imp_model=self.imputer)
        value = self.clf.predict(X)
        prob = self.clf.predict_proba(X)
        return value, prob
