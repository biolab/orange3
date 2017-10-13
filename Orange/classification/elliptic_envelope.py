import sklearn.covariance as skl_covariance

from Orange.base import SklLearner, SklModel
from Orange.data import Table
from Orange.preprocess import Continuize, RemoveNaNColumns, SklImpute

__all__ = ["EllipticEnvelopeLearner"]


class EllipticEnvelopeClassifier(SklModel):
    def mahalanobis(self, observations):
        """Computes squared Mahalanobis distances of given observations.

        Parameters
        ----------
        observations : ndarray (n_samples, n_features) or Orange Table

        Returns
        -------
        distances : ndarray (n_samples,)
            Squared Mahalanobis distances given observations.
        """
        if isinstance(observations, Table):
            observations = observations.X
        return self.skl_model.mahalanobis(observations)


class EllipticEnvelopeLearner(SklLearner):
    __wraps__ = skl_covariance.EllipticEnvelope
    __returns__ = EllipticEnvelopeClassifier
    preprocessors = [Continuize(), RemoveNaNColumns(), SklImpute()]

    def __init__(self, store_precision=True, assume_centered=False,
                 support_fraction=None, contamination=0.1,
                 random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()
