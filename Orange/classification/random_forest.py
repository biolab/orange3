import sklearn.ensemble as skl_ensemble

from Orange.classification import SklLearner, SklModel
from Orange.data import Variable, DiscreteVariable
from Orange.preprocess.score import LearnerScorer

__all__ = ["RandomForestLearner"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = DiscreteVariable

    def score(self, model):
        return model.skl_model.feature_importances_


class RandomForestClassifier(SklModel):
    pass


class RandomForestLearner(SklLearner, _FeatureScorerMixin):
    __wraps__ = skl_ensemble.RandomForestClassifier
    __returns__ = RandomForestClassifier
    name = 'random forest'

    def __init__(self, n_estimators=10, max_features="auto",
                 random_state=None, max_depth=None, max_leaf_nodes=None,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
