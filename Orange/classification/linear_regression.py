from Orange import classification
from sklearn.linear_model import LinearRegression


class LinearRegressionLearner(classification.Fitter):
    def fit(self, X, Y, W):
        clf = LinearRegression()
        return LinearRegressionPredictor(clf.fit(X, Y.reshape(-1)))


class LinearRegressionPredictor(classification.Model):
    def __init__(self, clf):
        self.clf = clf

    def predict(self, X):
        value = self.clf.predict(X)
        return value
