from sklearn import linear_model

from Orange import classification


class LogisticRegressionLearner(classification.Fitter):
    def __init__(self, penalty="l2", dual=False, tol=0.0001, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.random_state = random_state

    def fit(self, X, Y, W):
        lr = linear_model.LogisticRegression(
            penalty=self.penalty,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            random_state=self.random_state
        )
        clsf = lr.fit(X, Y.ravel())

        return LogisticRegressionClassifier(clsf)


class LogisticRegressionClassifier(classification.Model):
    def __init__(self, clf):
        self.clf = clf

    def predict(self, X):
        value = self.clf.predict(X)
        prob = self.clf.predict_proba(X)
        return value, prob
