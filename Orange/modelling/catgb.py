from typing import Tuple

import numpy as np

from Orange.base import CatGBBaseLearner
from Orange.classification import CatGBClassifier
from Orange.data import Variable, Table
from Orange.modelling import SklFitter
from Orange.preprocess.score import LearnerScorer
from Orange.regression import CatGBRegressor

__all__ = ["CatGBLearner"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = Variable

    def score(self, data: Table) -> Tuple[np.ndarray, Tuple[Variable]]:
        model: CatGBBaseLearner = self.get_learner(data)(data)
        return model.cat_model.feature_importances_, model.domain.attributes


class CatGBLearner(SklFitter, _FeatureScorerMixin):
    name = "Gradient Boosting (catboost)"
    __fits__ = {"classification": CatGBClassifier,
                "regression": CatGBRegressor}
