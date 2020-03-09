# pylint: disable=unused-argument
from typing import Callable

import numpy as np

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from Orange.base import SklLearner, SklModel
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable, \
    Variable
from Orange.data.table import DomainTransformationError
from Orange.data.util import get_unique_names
from Orange.preprocess import AdaptiveNormalize
from Orange.statistics.util import all_nan
from Orange.util import wrap_callback, dummy_callback

__all__ = ["LocalOutlierFactorLearner", "IsolationForestLearner",
           "EllipticEnvelopeLearner", "OneClassSVMLearner"]


class _OutlierModel(SklModel):
    def __init__(self, skl_model):
        super().__init__(skl_model)
        self._cached_data = None
        self.outlier_var = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = self.skl_model.predict(X)
        pred[pred == -1] = 0
        return pred[:, None]

    def __call__(self, data: Table, progress_callback: Callable = None) \
            -> Table:
        assert isinstance(data, Table)
        assert self.outlier_var is not None

        domain = Domain(data.domain.attributes, data.domain.class_vars,
                        data.domain.metas + (self.outlier_var,))
        if progress_callback is None:
            progress_callback = dummy_callback
        progress_callback(0, "Preprocessing...")
        self._cached_data = self.data_to_model_domain(
            data, wrap_callback(progress_callback, end=0.1))
        progress_callback(0.1, "Predicting...")
        metas = np.hstack((data.metas, self.predict(self._cached_data.X)))
        progress_callback(1)
        return Table.from_numpy(domain, data.X, data.Y, metas)

    def data_to_model_domain(self, data: Table, progress_callback: Callable) \
            -> Table:
        if data.domain == self.domain:
            return data

        progress_callback(0)
        if self.original_domain.attributes != data.domain.attributes \
                and data.X.size \
                and not all_nan(data.X):
            progress_callback(0.5)
            new_data = data.transform(self.original_domain)
            if all_nan(new_data.X):
                raise DomainTransformationError(
                    "domain transformation produced no defined values")
            progress_callback(0.75)
            data = new_data.transform(self.domain)
            progress_callback(1)
            return data

        progress_callback(0.5)
        data = data.transform(self.domain)
        progress_callback(1)
        return data


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

        transformer.variable = variable
        model.outlier_var = variable
        return model


class _Transformer:
    def __init__(self, model: _OutlierModel):
        self._model = model
        self._variable = None

    @property
    def variable(self) -> Variable:
        return self._variable

    @variable.setter
    def variable(self, var: Variable):
        self._variable = var

    def __call__(self, data: Table) -> np.ndarray:
        assert isinstance(self._variable, Variable)
        return self._model(data).get_column_view(self._variable)[0]


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

    def __call__(self, data: Table, progress_callback: Callable = None) \
            -> Table:
        pred = super().__call__(data, progress_callback)
        domain = Domain(pred.domain.attributes, pred.domain.class_vars,
                        pred.domain.metas + (self.mahal_var,))
        metas = np.hstack((pred.metas, self.mahalanobis(self._cached_data.X)))
        return Table.from_numpy(domain, pred.X, pred.Y, metas)


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

        transformer = _Transformer(model)
        names = [v.name for v in domain.variables + domain.metas]
        variable = ContinuousVariable(
            get_unique_names(names, "Mahalanobis"),
            compute_value=transformer
        )

        transformer.variable = variable
        model.mahal_var = variable
        return model
