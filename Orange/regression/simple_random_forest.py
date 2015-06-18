import numpy as np

from Orange.classification.simple_random_forest import SimpleRandomForestLearner as SRFL
from Orange.classification.simple_random_forest import SimpleRandomForestModel as SRFM

__all__ = ['SimpleRandomForestLearner']


class SimpleRandomForestLearner(SRFL):
    """
    A random forest regressor, optimized for speed. Trees in the forest
    are constructed with :obj:`SimpleTreeLearner` classification trees.


    Parameters
    ----------

    n_estimators : int, optional (default = 10)
        Number of trees in the forest.

    min_instances : int, optional (default = 2)
        Minimal number of data instances in leaves. When growing the three,
        new nodes are not introduced if they would result in leaves
        with fewer instances than min_instances. Instance count is weighed.

    max_depth : int, optional (default = 1024)
        Maximal depth of tree.

    max_majority : float, optional (default = 1.0)
        Maximal proportion of majority class. When this is
        exceeded, induction stops (only used for classification).

    skip_prob : string, optional (default = "sqrt")
        Data attribute will be skipped with probability ``skip_prob``.

        - if float, then skip attribute with this probability.
        - if "sqrt", then `skip_prob = 1 - sqrt(n_features) / n_features`
        - if "log2", then `skip_prob = 1 - log2(n_features) / n_features`

    seed : int, optional (default = 42)
        Random seed.
    """
    def fit_storage(self, data):
        return SimpleRandomForestModel(self, data)


class SimpleRandomForestModel(SRFM):

    def __init__(self, learner, data):
        self.estimators_ = []

        if data.domain.has_continuous_class:
            self.type = 'regression'
            self.cls_vals = 0
        else:
            assert(False)
        self.learn(learner, data)

    def predict_storage(self, data):
        if self.type == 'regression':
            p = np.zeros(data.X.shape[0])
            for tree in self.estimators_:
                p += tree(data)
            p /= len(self.estimators_)
            return p
        else:
            assert(False)
