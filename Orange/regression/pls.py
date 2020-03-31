import numpy as np
import pandas as pd

import sklearn.cross_decomposition as skl_pls
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from Orange.regression import SklLearner
from Orange.preprocess import Normalize
from Orange.data import Variable, ContinuousVariable
from Orange.preprocess.score import LearnerScorer
from Orange.base import Learner, Model, SklLearner, SklModel

__all__ = ["PLSRegressionLearner"]

# Add any preprocessing of data here
# Normalization is only needed if and when the data 
# is changing overall shape or the x axis is varying for every data row/instance
# Needs to be checked

pls_pps = SklLearner.preprocessors

class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = ContinuousVariable

    def score(self, data):
        model = self(data)
        return np.abs(model.coefficients), model.domain.attributes

# Simple PLS Regression:

class SklPLSRegressionLearner(SklLearner, _FeatureScorerMixin):
    __wraps__ = skl_pls.PLSRegression

    def __init__(self, preprocessors=None):
        super().__init__(preprocessors=preprocessors)

    def fit(self, X, Y, W=None):
        model = super().fit(X, Y, W)
        return PLSModel(model.skl_model)

class PLSRegressionLearner(SklPLSRegressionLearner):
    __wraps__ = skl_pls.PLSRegression
    preprocessors = pls_pps

    def __init__(self, n_components=2, scale=True, 
                max_iter=500,preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

# PLS MODEL

class PLSModel(SklModel):
    @property
    def intercept(self):
        return self.skl_model.intercept_

    @property
    def coefficients(self):
        return self.skl_model.coef_

    def predict(self, X):
        vals = self.skl_model.predict(X)
        if len(vals.shape) == 1:
            # Prevent IndexError for 1D array
            return vals
        elif vals.shape[1] == 1:
            return vals.ravel()
        else:
            return vals

    def __str__(self):
        return 'PLSModel {}'.format(self.skl_model)


if __name__ == '__main__':
    import Orange

    data = Orange.data.Table('housing')
    learners = [PLSRegressionLearner(n_components=2, max_iter=100)]
    res = Orange.evaluation.CrossValidation(data, learners)
    for l, ca in zip(learners, Orange.evaluation.RMSE(res)):
        print("learner: {}\nRMSE: {}\n".format(l, ca))
