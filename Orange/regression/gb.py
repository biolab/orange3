# pylint: disable=unused-argument,too-many-arguments
from typing import Tuple

import numpy as np
import sklearn.ensemble as skl_ensemble

from Orange.data import Variable, ContinuousVariable, Table
from Orange.preprocess.score import LearnerScorer
from Orange.regression import SklLearner, SklModel

__all__ = ["GBRegressor"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = ContinuousVariable

    def score(self, data: Table) -> Tuple[np.ndarray, Tuple[Variable]]:
        model: GBRegressor = self(data)
        return model.skl_model.feature_importances_, model.domain.attributes


class GBRegressor(SklLearner, _FeatureScorerMixin):
    __wraps__ = skl_ensemble.GradientBoostingRegressor
    __returns__ = SklModel

    def __init__(self,
                 loss="ls",
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 criterion="friedman_mse",
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_depth=3,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 init=None,
                 random_state=None,
                 max_features=None,
                 alpha=0.9,
                 verbose=0,
                 max_leaf_nodes=None,
                 warm_start=False,
                 presort="deprecated",
                 validation_fraction=0.1,
                 n_iter_no_change=None,
                 tol=1e-4,
                 ccp_alpha=0.0,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
