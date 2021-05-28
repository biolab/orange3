# pylint: disable=unused-argument
from typing import Callable

import numpy as np

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from Orange.base import SklLearner, SklModel
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.data.util import get_unique_names, SharedComputeValue
from Orange.preprocess import AdaptiveNormalize
from Orange.util import dummy_callback

__all__ = ["LocalOutlierFactorLearner", "IsolationForestLearner",
           "EllipticEnvelopeLearner", "OneClassSVMLearner"]


class _CachedTransform:
    # to be used with SharedComputeValue
    def __init__(self, model):
        self.model = model

    def __call__(self, data):
        return self.model.data_to_model_domain(data)


class _OutlierModel(SklModel):
    def __init__(self, skl_model):
        super().__init__(skl_model)
        self.outlier_var = None
        self.cached_transform = _CachedTransform(self)

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = self.skl_model.predict(X)
        pred[pred == -1] = 0
        return pred[:, None]

    def new_domain(self, data: Table) -> Domain:
        assert self.outlier_var is not None
        return Domain(data.domain.attributes, data.domain.class_vars,
               data.domain.metas + (self.outlier_var,))

    def __call__(self, data: Table, progress_callback: Callable = None) \
            -> Table:
        assert isinstance(data, Table)

        domain = self.new_domain(data)
        if progress_callback is None:
            progress_callback = dummy_callback
        progress_callback(0, "Predicting...")
        new_table = data.transform(domain)
        progress_callback(1)
        return new_table


class _OutlierLearner(SklLearner):
    __returns__ = _OutlierModel
    supports_multiclass = True

    def _fit_model(self, data: Table) -> _OutlierModel:
        domain = data.domain
        model = super()._fit_model(data.transform(Domain(domain.attributes)))

        transformer = _Transformer(model)
        names = [v.name for v in domain.variables + domain.metas]
        variable = DiscreteVariable(
            get_unique_names(names, "Outlier"),
            values=("Yes", "No"),
            compute_value=transformer
        )

        model.outlier_var = variable
        return model


class _Transformer(SharedComputeValue):
    def __init__(self, model: _OutlierModel):
        super().__init__(model.cached_transform)
        self._model = model

    def compute(self, data: Table, shared_data: Table) -> np.ndarray:
        return self._model.predict(shared_data.X)[:, 0]


class OneClassSVMLearner(_OutlierLearner):
    name = "One class SVM"
    __wraps__ = OneClassSVM
    preprocessors = SklLearner.preprocessors + [AdaptiveNormalize()]

    def __init__(self, kernel='rbf', degree=3, gamma="auto", coef0=0.0,
                 tol=0.001, nu=0.5, shrinking=True, cache_size=200,
                 max_iter=-1, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class LocalOutlierFactorLearner(_OutlierLearner):
    __wraps__ = LocalOutlierFactor
    name = "Local Outlier Factor"

    def __init__(self, n_neighbors=20, algorithm="auto", leaf_size=30,
                 metric="minkowski", p=2, metric_params=None,
                 contamination="auto", novelty=True, n_jobs=None,
                 preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class IsolationForestLearner(_OutlierLearner):
    __wraps__ = IsolationForest
    name = "Isolation Forest"

    def __init__(self, n_estimators=100, max_samples='auto',
                 contamination='auto', max_features=1.0, bootstrap=False,
                 n_jobs=None, behaviour='deprecated', random_state=None,
                 verbose=0, warm_start=False, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()


class EllipticEnvelopeClassifier(_OutlierModel):
    def __init__(self, skl_model):
        super().__init__(skl_model)
        self.mahal_var = None

    def mahalanobis(self, observations: np.ndarray) -> np.ndarray:
        """Computes squared Mahalanobis distances of given observations.

        Parameters
        ----------
        observations : ndarray (n_samples, n_features)

        Returns
        -------
        distances : ndarray (n_samples, 1)
            Squared Mahalanobis distances given observations.
        """
        return self.skl_model.mahalanobis(observations)[:, None]

    def new_domain(self, data: Table) -> Domain:
        assert self.mahal_var is not None
        domain = super().new_domain(data)
        return Domain(domain.attributes, domain.class_vars,
                        domain.metas + (self.mahal_var,))


class _TransformerMahalanobis(_Transformer):
    def compute(self, data: Table, shared_data: Table) -> np.ndarray:
        return self._model.mahalanobis(shared_data.X)[:, 0]


class EllipticEnvelopeLearner(_OutlierLearner):
    __wraps__ = EllipticEnvelope
    __returns__ = EllipticEnvelopeClassifier
    name = "Covariance Estimator"

    def __init__(self, store_precision=True, assume_centered=False,
                 support_fraction=None, contamination=0.1,
                 random_state=None, preprocessors=None):
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def _fit_model(self, data: Table) -> EllipticEnvelopeClassifier:
        domain = data.domain
        model = super()._fit_model(data.transform(Domain(domain.attributes)))

        transformer = _TransformerMahalanobis(model)
        names = [v.name for v in domain.variables + domain.metas]
        variable = ContinuousVariable(
            get_unique_names(names, "Mahalanobis"),
            compute_value=transformer
        )

        model.mahal_var = variable
        return model
