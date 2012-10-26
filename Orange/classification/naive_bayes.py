from Orange import classification
from Orange.data import Table
from sklearn.naive_bayes import MultinomialNB

class BayesLearner(classification.Fitter):
    def __call__(self, data):
        assert isinstance(data, Table)
        clf = MultinomialNB()
        clf.fit(data.X, data.Y[:,0])
        return BayesClassifier(data.domain, clf)


class BayesClassifier(classification.Model):
    def __init__(self, domain, clf):
        super().__init__(domain)
        self.clf = clf

    def __call__(self, X):
        return self.predict(X)

    def predict(self, X):
        return self.clf.predict(X)
