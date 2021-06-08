from typing import Iterable, Callable, List, Dict, Optional

import numpy as np
from scipy.optimize import curve_fit

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.data.filter import HasClass
from Orange.preprocess import RemoveNaNColumns, Impute
from Orange.regression import Learner, Model

__all__ = ["CurveFitLearner"]


class CurveFitModel(Model):
    def __init__(self, domain: Domain, parameters_names: List[str],
                 parameters: np.ndarray, function: Callable):
        super().__init__(domain)
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
        return self.__function(X, *self.__parameters).flatten()


class CurveFitLearner(Learner):
    preprocessors = [HasClass(), RemoveNaNColumns(), Impute()]
    __returns__ = CurveFitModel
    name = "Curve Fit"

    def __init__(self, function: Callable, parameters_names: List[str],
                 p0=None, bounds=(-np.inf, np.inf), preprocessors=None):
        if not callable(function):
            raise TypeError("Function is not callable.")

        super().__init__(preprocessors)
        self.__function = function
        self.__parameters_names = parameters_names
        self.__p0: Optional[Iterable] = p0
        self.__bounds: Iterable = bounds

    def fit_storage(self, data: Table) -> CurveFitModel:
        domain = data.domain
        for attr in domain.attributes:
            if not attr.is_continuous:
                raise ValueError("Numeric feature expected.")

        params, _ = curve_fit(self.__function, data.X, data.Y,
                              p0=self.__p0, bounds=self.__bounds)
        return CurveFitModel(domain, self.__parameters_names,
                             params, self.__function)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    housing = Table("housing")
    xdata = housing.X
    ydata = housing.Y

    func = lambda x, a, b, c: a * np.exp(-b * x[:, 12]) + c
    pred = CurveFitLearner(func, ["a", "b", "c"])(housing)(housing)

    plt.plot(xdata[:, 12], ydata, "o")
    indices = np.argsort(xdata[:, 12])
    plt.plot(xdata[indices, 12], pred[indices])
    plt.show()
