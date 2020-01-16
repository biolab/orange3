# pylint: disable=unused-argument
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from Orange.base import SklLearner, SklModel
from Orange.data import Table, Domain

__all__ = ["LocalOutlierFactorLearner", "IsolationForestLearner",
           "EllipticEnvelopeLearner"]


class _OutlierDetector(SklLearner):
    def __call__(self, data: Table):
        data = data.transform(Domain(data.domain.attributes))
        return super().__call__(data)


class LocalOutlierFactorLearner(_OutlierDetector):
    __wraps__ = LocalOutlierFactor
    name = "Local Outlier Factor"

    def __init__(self, n_neighbors=20, algorithm="auto", leaf_size=30,
                 metric="minkowski", p=2, metric_params=None,
                 contamination="auto", novelty=True, n_jobs=None,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class IsolationForestLearner(_OutlierDetector):
    __wraps__ = IsolationForest
    name = "Isolation Forest"

    def __init__(self, n_estimators=100, max_samples='auto',
                 contamination='auto', max_features=1.0, bootstrap=False,
                 n_jobs=None, behaviour='deprecated', random_state=None,
                 verbose=0, warm_start=False, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


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


class EllipticEnvelopeLearner(_OutlierDetector):
    __wraps__ = EllipticEnvelope
    __returns__ = EllipticEnvelopeClassifier
    name = "Covariance Estimator"

    def __init__(self, store_precision=True, assume_centered=False,
                 support_fraction=None, contamination=0.1,
                 random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def __call__(self, data: Table):
        data = data.transform(Domain(data.domain.attributes))
        return super().__call__(data)
