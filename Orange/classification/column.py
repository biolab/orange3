from typing import Optional

import numpy as np

from Orange.data import Variable, DiscreteVariable, Domain
from Orange.classification import Model, Learner


__all__ = ["ColumnClassifier"]


class ColumnLearner(Learner):
    def __init__(self,
                 class_var: DiscreteVariable,
                 column: Variable,
                 offset: Optional[float] = None,
                 k: Optional[float] = None):
        super().__init__()
        self.class_var = class_var
        self.column = column
        self.offset = offset
        self.k = k
        self.name = f"column '{column.name}'"

    def fit_storage(self, _):
        return ColumnClassifier(
            self.class_var, self.column, self.offset, self.k)


class ColumnClassifier(Model):
    def __init__(self,
                 class_var: DiscreteVariable,
                 column: Variable,
                 offset: Optional[float] = None,
                 k: Optional[float] = None):
        super().__init__(Domain([column], class_var))
        assert class_var.is_discrete
        if column.is_continuous:
            assert len(class_var.values) == 2
            self.value_mapping = np.array([0, 1])
        else:
            assert column.is_discrete
            assert offset is None and k is None
            if not self.check_value_sets(class_var, column):
                raise ValueError(
                    "Column contains values that are not in class variable")
            self.value_mapping = np.array(
                [class_var.to_val(x) for x in column.values])
        self.class_var = class_var
        self.column = column
        self.offset = offset
        self.k = k
        self.name = f"column '{column.name}'"

    @staticmethod
    def check_prob_range(values: np.ndarray):
        return np.nanmin(values) >= 0 and np.nanmax(values) <= 1

    @staticmethod
    def check_value_sets(class_var: DiscreteVariable,
                         column_var: DiscreteVariable):
        return set(column_var.values) <= set(class_var.values)

    def predict_storage(self, data):
        vals = data.get_column(self.column)
        rows = np.isfinite(vals)
        if self.column.is_discrete:
            proba = np.zeros((len(data), len(self.class_var.values)))
            vals = self.value_mapping[vals[rows].astype(int)]
            proba[rows, vals] = 1
        else:
            proba = np.full((len(data), len(self.class_var.values)), 0.5)
            if self.k is None:
                if not self.check_prob_range(vals):
                    raise ValueError("Column values must be in [0, 1] range "
                                     "unless logistic function is applied")
                proba[rows, 1] = vals[rows]
                proba[rows, 0] = 1 - vals[rows]
                vals = vals > 0.5
            else:
                proba[rows, 1] = (
                    1 / (1 + np.exp(-self.k * (vals[rows] - self.offset))))
                proba[rows, 0] = 1 - proba[:, 1]
                vals = vals > self.offset
        return vals, proba

    def __str__(self):
        return f'ColumnClassifier {self.column.name}'
