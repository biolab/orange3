# pylint: disable=too-many-arguments
from typing import Tuple

import numpy as np

import xgboost

from Orange.base import XGBBase
from Orange.classification import Learner
from Orange.data import Variable, DiscreteVariable, Table
from Orange.preprocess.score import LearnerScorer

__all__ = ["XGBClassifier", "XGBRFClassifier"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = DiscreteVariable

    def score(self, data: Table) -> Tuple[np.ndarray, Tuple[Variable]]:
        model: XGBBase = self(data)
        return model.skl_model.feature_importances_, model.domain.attributes


class XGBClassifier(XGBBase, Learner, _FeatureScorerMixin):
    __wraps__ = xgboost.XGBClassifier

    def __init__(self,
                 max_depth=6,
                 learning_rate=0.3,
                 n_estimators=100,
                 verbosity=None,
                 objective="binary:logistic",
                 booster="gbtree",
                 tree_method="exact",
                 n_jobs=0,
                 gamma=0,
                 min_child_weight=1,
                 max_delta_step=0,
                 subsample=1,
                 colsample_bytree=1,
                 colsample_bylevel=1,
                 colsample_bynode=1,
                 reg_alpha=0,
                 reg_lambda=1,
                 scale_pos_weight=None,
                 base_score=0.5,
                 random_state=0,
                 missing=np.nan,
                 num_parallel_tree=1,
                 monotone_constraints=(),
                 interaction_constraints="",
                 importance_type="gain",
                 gpu_id=-1,
                 validate_parameters=1,
                 preprocessors=None):
        super().__init__(max_depth=max_depth,
                         learning_rate=learning_rate,
                         n_estimators=n_estimators,
                         verbosity=verbosity,
                         objective=objective,
                         booster=booster,
                         tree_method=tree_method,
                         n_jobs=n_jobs,
                         gamma=gamma,
                         min_child_weight=min_child_weight,
                         max_delta_step=max_delta_step,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         colsample_bylevel=colsample_bylevel,
                         colsample_bynode=colsample_bynode,
                         reg_alpha=reg_alpha,
                         reg_lambda=reg_lambda,
                         scale_pos_weight=scale_pos_weight,
                         base_score=base_score,
                         random_state=random_state,
                         missing=missing,
                         num_parallel_tree=num_parallel_tree,
                         monotone_constraints=monotone_constraints,
                         interaction_constraints=interaction_constraints,
                         importance_type=importance_type,
                         gpu_id=gpu_id,
                         validate_parameters=validate_parameters,
                         use_label_encoder=False,
                         preprocessors=preprocessors)


class XGBRFClassifier(XGBBase, Learner, _FeatureScorerMixin):
    __wraps__ = xgboost.XGBRFClassifier

    def __init__(self,
                 max_depth=6,
                 learning_rate=1,
                 n_estimators=100,
                 verbosity=None,
                 objective="binary:logistic",
                 booster="gbtree",
                 tree_method="exact",
                 n_jobs=0,
                 gamma=0,
                 min_child_weight=1,
                 max_delta_step=0,
                 subsample=0.8,
                 colsample_bytree=1,
                 colsample_bylevel=1,
                 colsample_bynode=0.8,
                 reg_alpha=0,
                 reg_lambda=1e-5,
                 scale_pos_weight=None,
                 base_score=0.5,
                 random_state=0,
                 missing=np.nan,
                 num_parallel_tree=1,
                 monotone_constraints=(),
                 interaction_constraints="",
                 importance_type="gain",
                 gpu_id=-1,
                 validate_parameters=1,
                 preprocessors=None):
        super().__init__(max_depth=max_depth,
                         learning_rate=learning_rate,
                         n_estimators=n_estimators,
                         verbosity=verbosity,
                         objective=objective,
                         booster=booster,
                         tree_method=tree_method,
                         n_jobs=n_jobs,
                         gamma=gamma,
                         min_child_weight=min_child_weight,
                         max_delta_step=max_delta_step,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         colsample_bylevel=colsample_bylevel,
                         colsample_bynode=colsample_bynode,
                         reg_alpha=reg_alpha,
                         reg_lambda=reg_lambda,
                         scale_pos_weight=scale_pos_weight,
                         base_score=base_score,
                         random_state=random_state,
                         missing=missing,
                         num_parallel_tree=num_parallel_tree,
                         monotone_constraints=monotone_constraints,
                         interaction_constraints=interaction_constraints,
                         importance_type=importance_type,
                         gpu_id=gpu_id,
                         validate_parameters=validate_parameters,
                         use_label_encoder=False,
                         preprocessors=preprocessors)
