from typing import Tuple

import numpy as np

import catboost

from Orange.base import CatGBBaseLearner, CatGBModel
from Orange.classification import Learner, Model
from Orange.data import Variable, DiscreteVariable, Table
from Orange.preprocess.score import LearnerScorer

__all__ = ["CatGBClassifier"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = DiscreteVariable

    def score(self, data: Table) -> Tuple[np.ndarray, Tuple[Variable]]:
        model: CatGBClassifier = self(data)
        return model.cat_model.feature_importances_, model.domain.attributes


class CatGBClsModel(CatGBModel, Model):
    pass


class CatGBClassifier(CatGBBaseLearner, Learner, _FeatureScorerMixin):
    __wraps__ = catboost.CatBoostClassifier
    __returns__ = CatGBClsModel
