import sklearn.ensemble as skl_ensemble
from Orange.regression import SklLearner, SklModel

__all__ = ["RandomForestRegressionLearner"]


class RandomForestRegressor(SklModel):
    pass


class RandomForestRegressionLearner(SklLearner):
    __wraps__ = skl_ensemble.RandomForestRegressor
    __returns__ = RandomForestRegressor
    name = 'random forest regression'

    def __init__(self, n_estimators=10, max_features="auto",
                 random_state=None, max_depth=3, max_leaf_nodes=5,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
