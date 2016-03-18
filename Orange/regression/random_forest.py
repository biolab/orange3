import sklearn.ensemble as skl_ensemble
from Orange.regression import SklLearner, SklModel
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
