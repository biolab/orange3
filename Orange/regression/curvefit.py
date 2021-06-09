from typing import Iterable, Callable, List, Optional

import numpy as np
from scipy.optimize import curve_fit

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.data.filter import HasClass
from Orange.preprocess import RemoveNaNColumns, Impute
from Orange.regression import Learner, Model

__all__ = ["CurveFitLearner"]


class CurveFitModel(Model):
    def __init__(self, domain: Domain, original_domain: Domain,
                 parameters_names: List[str],
                 parameters: np.ndarray, function: Callable):
        super().__init__(domain, original_domain)
        self.__parameters_names = parameters_names
        self.__parameters = parameters
        self.__function = function

    @property
    def coefficients(self) -> Table:
        return Table(Domain([ContinuousVariable("coef")],
                            metas=[StringVariable("name")]),
                     self.__parameters[:, None],
                     metas=np.array(self.__parameters_names)[:, None])

    def predict(self, X: np.ndarray) -> np.ndarray:
        predicted = self.__function(X, *self.__parameters)
        if isinstance(predicted, float):
            # handle constant function; i.e. len(self.domain.attributes) == 0
            return np.full(len(X), predicted)
        return predicted.flatten()


class CurveFitLearner(Learner):
    preprocessors = [HasClass(), RemoveNaNColumns(), Impute()]
    __returns__ = CurveFitModel
    name = "Curve Fit"

    def __init__(self, function: Callable, parameters_names: List[str],
                 feature_names: List[str], p0: Optional[Iterable] = None,
                 bounds: Iterable = (-np.inf, np.inf), preprocessors=None):
        super().__init__(preprocessors)

        if not callable(function):
            raise TypeError("Function is not callable.")

        self.__function = function
        self.__parameters_names = parameters_names
        self.__feature_names = feature_names
        self.__p0 = p0
        self.__bounds = bounds

    def fit_storage(self, data: Table) -> CurveFitModel:
        domain = data.domain
        attributes = []
        for attr in domain.attributes:
            if attr.name in self.__feature_names:
                if not attr.is_continuous:
                    raise ValueError("Numeric feature expected.")
                attributes.append(attr)

        new_domain = Domain(attributes, domain.class_vars, domain.metas)
        transformed = data.transform(new_domain)
        params, _ = curve_fit(self.__function, transformed.X, transformed.Y,
                              p0=self.__p0, bounds=self.__bounds)
        return CurveFitModel(new_domain, domain,
                             self.__parameters_names, params, self.__function)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    housing = Table("housing")
    xdata = housing.X
    ydata = housing.Y

    func = lambda x, a, b, c: a * np.exp(-b * x[:, 0]) + c
    pred = CurveFitLearner(func, ["a", "b", "c"], ["LSTAT"])(housing)(housing)

    plt.plot(xdata[:, 12], ydata, "o")
    indices = np.argsort(xdata[:, 12])
    plt.plot(xdata[indices, 12], pred[indices])
    plt.show()
