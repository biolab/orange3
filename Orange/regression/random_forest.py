import sklearn.ensemble as skl_ensemble

from Orange.base import RandomForestModel
from Orange.data import Variable, ContinuousVariable
from Orange.preprocess.score import LearnerScorer
from Orange.regression import SklLearner, SklModel
from Orange.regression.tree import SklTreeRegressor

__all__ = ["RandomForestRegressionLearner"]


class _FeatureScorerMixin(LearnerScorer):
    feature_type = Variable
    class_type = ContinuousVariable

    def score(self, data):
        model = self(data)
        return model.skl_model.feature_importances_


class RandomForestRegressor(SklModel, RandomForestModel):
    @property
    def trees(self):
        def wrap(tree, i):
            t = SklTreeRegressor(tree)
            t.domain = self.domain
            t.supports_multiclass = self.supports_multiclass
            t.name = "{} - tree {}".format(self.name, i)
            t.original_domain = self.original_domain
            if hasattr(self, 'instances'):
                t.instances = self.instances
            return t

        return [wrap(tree, i)
                for i, tree in enumerate(self.skl_model.estimators_)]


class RandomForestRegressionLearner(SklLearner, _FeatureScorerMixin):
    __wraps__ = skl_ensemble.RandomForestRegressor
    __returns__ = RandomForestRegressor

    def __init__(self,
                 n_estimators=10,
                 criterion="mse",
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
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
