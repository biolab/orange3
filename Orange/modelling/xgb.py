# pylint: disable=missing-docstring
from typing import Tuple

import numpy as np

from Orange.base import XGBBase
from Orange.classification import XGBClassifier, XGBRFClassifier
from Orange.data import Variable, Table
from Orange.modelling import SklFitter
from Orange.preprocess.score import LearnerScorer
from Orange.regression import XGBRegressor, XGBRFRegressor

__all__ = ["XGBLearner", "XGBRFLearner"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = Variable

    def score(self, data: Table) -> Tuple[np.ndarray, Tuple[Variable]]:
        model: XGBBase = self.get_learner(data)(data)
        return model.skl_model.feature_importances_, model.domain.attributes


class XGBLearner(SklFitter, _FeatureScorerMixin):
    name = "Extreme Gradient Boosting (xgboost)"
    __fits__ = {"classification": XGBClassifier,
                "regression": XGBRegressor}


class XGBRFLearner(SklFitter, _FeatureScorerMixin):
    name = "Extreme Gradient Boosting Random Forest (xgboost)"
    __fits__ = {"classification": XGBRFClassifier,
                "regression": XGBRFRegressor}
