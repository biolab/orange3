import numbers
import sklearn.ensemble as skl_ensemble
import sklearn.preprocessing as skl_preprocessing
from numpy import isnan

import Orange.data
from Orange.classification import SklLearner, SklModel

__all__ = ["RandomForestLearner"]


def replace_nan(X, imp_model):
    # Default scikit Imputer
    # Use Orange imputer when implemented
    if isnan(X).sum():
        X = imp_model.transform(X)
    return X


class RandomForestLearner(SklLearner):
    __wraps__ = skl_ensemble.RandomForestClassifier
    def __init__(self, n_estimators=10, max_features="auto",
                 random_state=None, max_depth=3, max_leaf_nodes=5,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def fit(self, X, Y, W):
        self.imputer = skl_preprocessing.Imputer()
        self.imputer.fit(X)
        X = replace_nan(X, self.imputer)

        params = dict(self.params)
        max_features = params["max_features"]
        if isinstance(max_features, numbers.Integral) and \
                X.shape[1] < max_features:
            params["max_features"] = X.shape[1]

        rf_model = skl_ensemble.RandomForestClassifier(**params)
        rf_model.fit(X, Y.ravel())
        return RandomForestClassifier(rf_model, self.imputer)


class RandomForestClassifier(SklModel):
    def __init__(self, clf, imp):
        self.clf = clf
        self.imputer = imp

    def predict(self, X):
        X = replace_nan(X, imp_model=self.imputer)
        value = self.clf.predict(X)
        prob = self.clf.predict_proba(X)
        return value, prob
