import sklearn.ensemble as skl_ensemble

from Orange.classification import SklLearner, SklModel
from Orange.classification.tree import TreeLearner
from Orange.data import Variable, DiscreteVariable
from Orange.preprocess.score import LearnerScorer
from Orange import options

__all__ = ["RandomForestLearner"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = DiscreteVariable

    def score(self, data):
        model = self(data)
        return model.skl_model.feature_importances_


class RandomForestClassifier(SklModel):
    pass


class RandomForestLearner(SklLearner, _FeatureScorerMixin):
    __wraps__ = skl_ensemble.RandomForestClassifier
    __returns__ = RandomForestClassifier
    name = 'random forest'

    ENSEMBLE_OPTIONS = [
        options.IntegerOption('n_estimators', default=10, range=(1, 10000),
                              verbose_name='Number of trees'),
        options.BoolOption('bootstrap', default=True),
    ]
    options = ENSEMBLE_OPTIONS + TreeLearner.options

    class GUI:
        main_scheme = ('n_estimators', 'bootstrap') + TreeLearner.GUI.main_scheme