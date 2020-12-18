from typing import Tuple

import numpy as np

import catboost

from Orange.base import CatGBBaseLearner
from Orange.classification import Learner, SklModel
from Orange.data import Variable, DiscreteVariable, Table
from Orange.preprocess.score import LearnerScorer

__all__ = ["CatGBClassifier"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = DiscreteVariable

    def score(self, data: Table) -> Tuple[np.ndarray, Tuple[Variable]]:
        model: CatGBClassifier = self(data)
        return model.skl_model.feature_importances_, model.domain.attributes


class CatGBClsModel(SklModel):
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray]:
        value, probs = super().predict(X)
        return value.flatten(), probs


class CatGBClassifier(CatGBBaseLearner, Learner, _FeatureScorerMixin):
    __wraps__ = catboost.CatBoostClassifier
    __returns__ = CatGBClsModel
