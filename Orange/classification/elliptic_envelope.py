import sklearn.covariance as skl_covariance

from Orange.base import SklLearner, SklModel
from Orange.data import Table
from Orange import options

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
    name = 'elliptic envelope'

    options = (
        options.BoolOption('store_precision', default=True),
        options.BoolOption('assume_centered', default=False),
        options.DisableableOption(
            'support_fraction',
            option=options.FloatOption(default=.5, range=(0., 1.)),
            disable_value=None, disable_label='auto'
        ),
        options.FloatOption('contamination', default=.1, range=(0., .5)),
        options.IntegerOption('random_state', default=0),
    )
