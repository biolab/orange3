import sklearn.ensemble as skl_ensemble

from Orange.base import RandomForest
from Orange.classification import SklLearner, SklModel
from Orange.classification.tree import TreeClassifier
from Orange.data import Variable, DiscreteVariable
from Orange.preprocess.score import LearnerScorer

__all__ = ["RandomForestLearner"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = DiscreteVariable

    def score(self, data):
        model = self(data)
        return model.skl_model.feature_importances_


class RandomForestClassifier(SklModel, RandomForest):
    @property
    def trees(self):
        def wrap(tree, i):
            t = TreeClassifier(tree)
            t.domain = self.domain
            t.supports_multiclass = self.supports_multiclass
            t.name = "{} - tree {}".format(self.name, i)
            t.original_domain = self.original_domain
            return t

        return [wrap(tree, i)
                for i, tree in enumerate(self.skl_model.estimators_)]


class RandomForestLearner(SklLearner, _FeatureScorerMixin):
    __wraps__ = skl_ensemble.RandomForestClassifier
    __returns__ = RandomForestClassifier
    name = 'random forest'

    def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 class_weight=None,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
