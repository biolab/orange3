import numpy as np
import scipy.sparse

from Orange.data import Table, Instance
from Orange.data.table import DomainTransformationError
from Orange.misc.wrapper_meta import WrapperMeta
from Orange.preprocess import Continuize, SklImpute


class ClusteringModel:

    def __init__(self, projector):
        self.projector = projector
        self.domain = None
        self.original_domain = None
        self.labels = projector.labels_

    def __call__(self, data):
        def fix_dim(x):
            return x[0] if one_d else x

        one_d = False
        if isinstance(data, np.ndarray):
            one_d = data.ndim == 1
            prediction = self.predict(np.atleast_2d(data))
        elif isinstance(data, scipy.sparse.csr_matrix) or \
                isinstance(data, scipy.sparse.csc_matrix):
            prediction = self.predict(data)
        elif isinstance(data, (Table, Instance)):
            if isinstance(data, Instance):
                data = Table.from_list(data.domain, [data])
                one_d = True
            if data.domain != self.domain:
                if self.original_domain.attributes != data.domain.attributes \
                        and data.X.size \
                        and not np.isnan(data.X).all():
                    data = data.transform(self.original_domain)
                    if np.isnan(data.X).all():
                        raise DomainTransformationError(
                            "domain transformation produced no defined values")
                data = data.transform(self.domain)
            prediction = self.predict(data.X)
        elif isinstance(data, (list, tuple)):
            if not isinstance(data[0], (list, tuple)):
                data = [data]
                one_d = True
            data = Table.from_list(self.original_domain, data)
            data = data.transform(self.domain)
            prediction = self.predict(data.X)
        else:
            raise TypeError("Unrecognized argument (instance of '{}')"
                            .format(type(data).__name__))

        return fix_dim(prediction)

    def predict(self, X):
        raise NotImplementedError(
            "This clustering algorithm does not support predicting.")


class Clustering(metaclass=WrapperMeta):
    """
    ${skldoc}
    Additional Orange parameters

    preprocessors : list, optional (default = [Continuize(), SklImpute()])
        An ordered list of preprocessors applied to data before
        training or testing.
    """
    __wraps__ = None
    __returns__ = ClusteringModel
    preprocessors = [Continuize(), SklImpute()]

    def __init__(self, preprocessors, parameters):
        self.preprocessors = preprocessors if preprocessors is not None else self.preprocessors
        self.params = {k: v for k, v in parameters.items()
                       if k not in ["self", "preprocessors", "__class__"]}

    def __call__(self, data):
        return self.get_model(data).labels

    def get_model(self, data):
        orig_domain = data.domain
        data = self.preprocess(data)
        model = self.fit_storage(data)
        model.domain = data.domain
        model.original_domain = orig_domain
        return model

    def fit_storage(self, data):
        # only data Table
        return self.fit(data.X)

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        return self.__returns__(self.__wraps__(**self.params).fit(X))

    def preprocess(self, data):
        for pp in self.preprocessors:
            data = pp(data)
        return data
