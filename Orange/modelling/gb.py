from typing import Tuple

import numpy as np

from Orange.base import SklLearner
from Orange.classification import GBClassifier
from Orange.data import Variable, Table
from Orange.modelling import SklFitter
from Orange.preprocess.score import LearnerScorer
from Orange.regression import GBRegressor

__all__ = ["GBLearner"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = Variable

    def score(self, data: Table) -> Tuple[np.ndarray, Tuple[Variable]]:
        model: SklLearner = self.get_learner(data)(data)
        return model.skl_model.feature_importances_, model.domain.attributes


class GBLearner(SklFitter, _FeatureScorerMixin):
    name = "Gradient Boosting (scikit-learn)"
    __fits__ = {"classification": GBClassifier,
                "regression": GBRegressor}
