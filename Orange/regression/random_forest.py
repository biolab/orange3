import sklearn.ensemble as skl_ensemble

from Orange.classification import RandomForestLearner
from Orange.regression import SklLearner, SklModel
from Orange.regression.tree import TreeRegressionLearner
from Orange.data import Variable, ContinuousVariable
from Orange.preprocess.score import LearnerScorer

__all__ = ["RandomForestRegressionLearner"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = ContinuousVariable

    def score(self, data):
        model = self(data)
        return model.skl_model.feature_importances_


class RandomForestRegressor(SklModel):
    pass


class RandomForestRegressionLearner(SklLearner, _FeatureScorerMixin):
    __wraps__ = skl_ensemble.RandomForestRegressor
    __returns__ = RandomForestRegressor
    name = 'random forest regression'

    options = RandomForestLearner.ENSEMBLE_OPTIONS + \
              TreeRegressionLearner.options

    GUI = RandomForestLearner.GUI
