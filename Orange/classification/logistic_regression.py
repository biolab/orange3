import numpy as np
import sklearn.linear_model as skl_linear_model

from Orange.classification import SklLearner, SklModel
from Orange.preprocess import Normalize
from Orange.preprocess.score import LearnerScorer
from Orange.data import Variable, DiscreteVariable

__all__ = ["LogisticRegressionLearner"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = DiscreteVariable

    def score(self, data):
        data = Normalize()(data)
        model = self(data)
        return np.abs(model.coefficients), model.domain.attributes


class LogisticRegressionClassifier(SklModel):
    @property
    def intercept(self):
        return self.skl_model.intercept_

    @property
    def coefficients(self):
        return self.skl_model.coef_


class LogisticRegressionLearner(SklLearner, _FeatureScorerMixin):
    __wraps__ = skl_linear_model.LogisticRegression
    __returns__ = LogisticRegressionClassifier
    preprocessors = SklLearner.preprocessors

    def __init__(self, penalty="l2", dual=False, tol=0.0001, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver="auto", max_iter=100,
                 multi_class="auto", verbose=0, n_jobs=1, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def _initialize_wrapped(self):
        params = self.params.copy()
        # The default scikit-learn solver `lbfgs` (v0.22) does not support the
        # l1 penalty.
        solver, penalty = params.pop("solver"), params.get("penalty")
        if solver == "auto":
            if penalty == "l1":
                solver = "liblinear"
            else:
                solver = "lbfgs"
        params["solver"] = solver

        return self.__wraps__(**params)

