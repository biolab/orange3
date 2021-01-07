from typing import Tuple

import numpy as np

import catboost

from Orange.base import CatGBBaseLearner
from Orange.regression import Learner
from Orange.data import Variable, ContinuousVariable, Table
from Orange.preprocess.score import LearnerScorer

__all__ = ["CatGBRegressor"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = ContinuousVariable

    def score(self, data: Table) -> Tuple[np.ndarray, Tuple[Variable]]:
        model: CatGBBaseLearner = self(data)
        return model.cat_model.feature_importances_, model.domain.attributes


class CatGBRegressor(CatGBBaseLearner, Learner, _FeatureScorerMixin):
    __wraps__ = catboost.CatBoostRegressor
