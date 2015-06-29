import sklearn.covariance as skl_covariance

from Orange.classification import SklLearner

__all__ = ["EllipticEnvelopeLearner"]


class EllipticEnvelopeLearner(SklLearner):
    __wraps__ = skl_covariance.EllipticEnvelope
    name = 'elliptic envelope'

    def __init__(self, store_precision=True, assume_centered=False,
                 support_fraction=None, contamination=0.1,
                 random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
        self.clf = None

    def fit(self, X, Y=None, W=None):
        self.clf = self.__wraps__(**self.params)
        if W is not None:
            return self.__returns__(self.clf.fit(X, W.reshape(-1)))
        return self.__returns__(self.clf.fit(X))

    def mahalanobis(self, observations):
        """Return Mahalanobis distances of the training set observations.

        Parameters
        ----------
        observations : ndarray (n_samples, n_features)
            Data on which `fit` is called.

        Returns
        -------
        distances : ndarray (n_samples,)
            Mahalanobis distances of the training set observations.
        """
        if self.clf is not None:
            return self.clf.mahalanobis(observations)
