from Orange import classification
from sklearn import tree


class ClassificationTreeLearner(classification.SklFitter):

    def __init__(self, criterion="gini", splitter="best", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features=None,
                 random_state=None, max_leaf_nodes=None):
        self.params = vars()

    def fit(self, X, Y, W):
        clf = tree.DecisionTreeClassifier(**self.params)
        if W is None:
            return ClassificationTreeClassifier(clf.fit(X, Y))
        else:
            return ClassificationTreeClassifier(
                clf.fit(X, Y, sample_weight=W.reshape(-1)))


class ClassificationTreeClassifier(classification.SklModel):

    def __init__(self, clf):
        self.clf = clf

    def predict(self, X):
        value = self.clf.predict(X)
        return value
