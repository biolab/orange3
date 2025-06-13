from typing import Optional

import numpy as np

from Orange.data import Variable, DiscreteVariable, Domain, Table
from Orange.classification import LogisticRegressionLearner
from Orange.regression import LinearRegressionLearner
from Orange.modelling import Model, Learner

__all__ = ["ColumnLearner", "ColumnModel"]


def _check_column_combinations(
        class_var: Variable,
        column: Variable,
        fit_regression: bool):
    if class_var.is_continuous:
        if not column.is_continuous:
            raise ValueError(
                "Regression can only be used with numeric variables")
        return

    assert isinstance(class_var, DiscreteVariable)  # remove type warnings
    if column.is_continuous:
        if len(class_var.values) != 2:
            raise ValueError(
                "Numeric columns can only be used with binary class variables")
    else:
        assert isinstance(column, DiscreteVariable)
        if not valid_value_sets(class_var, column):
            raise ValueError(
                "Column contains values that are not in class variable")
    if fit_regression and not column.is_continuous:
        raise ValueError(
            "Intercept and coefficient are only allowed for continuous "
            "variables")


def valid_prob_range(values: np.ndarray):
    return np.nanmin(values) >= 0 and np.nanmax(values) <= 1


def valid_value_sets(class_var: DiscreteVariable,
                     column_var: DiscreteVariable):
    return set(column_var.values) <= set(class_var.values)


class ColumnLearner(Learner):
    def __init__(self,
                 class_var: Variable,
                 column: Variable,
                 fit_regression: bool = False):
        super().__init__()
        _check_column_combinations(class_var, column, fit_regression)
        self.class_var = class_var
        self.column = column
        self.fit_regression = fit_regression
        self.name = f"column '{column.name}'"

    def __fit_coefficients(self, data: Table):
        # Use learners from Orange rather than directly calling
        # scikit-learn, so that we make sure we use the same parameters
        # and get the same result as we would if we used the widgets.
        data1 = data.transform(Domain([self.column], self.class_var))
        if self.class_var.is_discrete:
            model = LogisticRegressionLearner()(data1)
            return model.intercept[0], model.coefficients[0][0]
        else:
            model = LinearRegressionLearner()(data1)
            return model.intercept, model.coefficients[0]

    def fit_storage(self, data: Table):
        if data.domain.class_var != self.class_var:
            raise ValueError("Class variable does not match the data")
        if not self.fit_regression:
            return ColumnModel(self.class_var, self.column)

        intercept, coefficient = self.__fit_coefficients(data)
        return ColumnModel(self.class_var, self.column, intercept, coefficient)


class ColumnModel(Model):
    def __init__(self,
                 class_var: Variable,
                 column: Variable,
                 intercept: Optional[float] = None,
                 coefficient: Optional[float] = None):
        super().__init__(Domain([column], class_var))

        _check_column_combinations(class_var, column, intercept is not None)
        if (intercept is not None) is not (coefficient is not None):
            raise ValueError(
                "Intercept and coefficient must both be provided or absent")

        self.class_var = class_var
        self.column = column
        self.intercept = intercept
        self.coefficient = coefficient
        if (column.is_discrete and
                class_var.values[:len(column.values)] != column.values):
            self.value_mapping = np.array([class_var.to_val(x)
                                          for x in column.values])
        else:
            self.value_mapping = None

        pars = f" ({intercept}, {coefficient})" if intercept is not None else ""
        self.name = f"column '{column.name}'{pars}"

    def predict_storage(self, data: Table):
        vals = data.get_column(self.column)
        if self.class_var.is_discrete:
            return self._predict_discrete(vals)
        else:
            return self._predict_continuous(vals)

    def _predict_discrete(self, vals):
        assert isinstance(self.class_var, DiscreteVariable)
        nclasses = len(self.class_var.values)
        proba = np.full((len(vals), nclasses), np.nan)
        rows = np.isfinite(vals)
        if self.column.is_discrete:
            mapped = vals[rows].astype(int)
            if self.value_mapping is not None:
                mapped = self.value_mapping[mapped]
                vals = vals.copy()
                vals[rows] = mapped
            proba[rows] = 0
            proba[rows, mapped] = 1
        else:
            if self.coefficient is None:
                if not valid_prob_range(vals):
                    raise ValueError("Column values must be in [0, 1] range "
                                     "unless logistic function is applied")
                proba[rows, 1] = vals[rows]
            else:
                proba[rows, 1] = (
                    1 /
                    (1 + np.exp(-self.intercept - self.coefficient * vals[rows])
                     ))

            proba[rows, 0] = 1 - proba[rows, 1]
            vals = (proba[:, 1] > 0.5).astype(float)
            vals[~rows] = np.nan
        return vals, proba

    def _predict_continuous(self, vals):
        if self.coefficient is None:
            return vals
        else:
            return vals * self.coefficient + self.intercept

    def __str__(self):
        pars = f" ({self.intercept}, {self.coefficient})" \
            if self.intercept is not None else ""
        return f'ColumnModel {self.column.name}{pars}'
