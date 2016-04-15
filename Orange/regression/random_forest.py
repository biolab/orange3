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

"""
n_estimators = The number of trees in the forest.
criterion = The function to measure the quality of a split.
max_depth = The maximum depth of the tree.
min_samples_split = The minimum number of samples required to split an internal node.
min_samples_leaf = The minimum number of samples in newly created leaves.
min_weight_fraction_leaf = The minimum weighted fraction of the input samples required to be at a leaf node.
max_features = The number of features to consider when looking for the best split:
    If int, then consider max_features features at each split.
    If float, then max_features is a percentage and int(max_features * n_features) features are considered at each split.
    If “auto”, then max_features=sqrt(n_features).
    If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
    If “log2”, then max_features=log2(n_features).
    If None, then max_features=n_features.
max_leaf_nodes = Grow trees with max_leaf_nodes in best-first fashion.
bootstrap = Whether bootstrap samples are used when building trees.
oob_score = Whether to use out-of-bag samples to estimate the generalization error.
n_jobs = The number of jobs to run in parallel for both fit and predict.
          If -1, then the number of jobs is set to the number of cores.
random_state = If int, random_state is the seed used by the random number generator;
verbose = Controls the verbosity of the tree building process.

"""

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
