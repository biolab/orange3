import numpy as np

import Orange

__all__ = ['SimpleRandomForestLearner']


class SimpleRandomForestLearner(Orange.classification.base.Learner):

    def __init__(self, n_estimators=10, min_instances=2, max_depth=1024,
                 max_majority=1.0, skip_prob='sqrt', seed=42):
        '''
        Parameters
        ----------

        n_estimators : Number of trees.

        min_instance : Minimal number of instances in leaves.  Instance count
        is weighed.

        max_depth : Maximal depth of tree.

        max_majority : Maximal proportion of majority class. When this is
        exceeded, induction stops (only used for classification).

        skip_prob : an attribute will be skipped with probability
        ``skip_prob``.

          - if float, then skip attribute with this probability.
          - if "sqrt", then `skip_prob = 1 - sqrt(n_features) / n_features`
          - if "log2", then `skip_prob = 1 - log2(n_features) / n_features`

        seed : Random seed.
        '''

        self.n_estimators = n_estimators
        self.skip_prob = skip_prob
        self.max_depth = max_depth
        self.min_instances = min_instances
        self.max_majority = max_majority
        self.seed = seed

    def fit_storage(self, data):
        return SimpleRandomForestModel(self, data)


class SimpleRandomForestModel(Orange.classification.base.Model):

    def __init__(self, learner, data):
        self.estimators_ = []

        if isinstance(data.domain.class_var, Orange.data.DiscreteVariable):
            self.type = 'classification'
            self.cls_vals = len(data.domain.class_var.values)
        elif isinstance(data.domain.class_var, Orange.data.ContinuousVariable):
            self.type = 'regression'
            self.cls_vals = 0
        else:
            assert(False)
        tree = Orange.classification.simple_tree.SimpleTreeLearner(
            learner.min_instances, learner.max_depth,
            learner.max_majority, learner.skip_prob, True)
        for i in range(learner.n_estimators):
            tree.seed = learner.seed + i
            self.estimators_.append(tree(data))

    def predict_storage(self, data):
        if self.type == 'classification':
            p = np.zeros((data.X.shape[0], self.cls_vals))
            for tree in self.estimators_:
                p += tree(data, tree.Probs)
            p /= len(self.estimators_)
            return p.argmax(axis=1), p
        elif self.type == 'regression':
            p = np.zeros(data.X.shape[0])
            for tree in self.estimators_:
                p += tree(data)
            p /= len(self.estimators_)
            return p
        else:
            assert(False)
