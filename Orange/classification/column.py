from typing import Optional

import numpy as np

from Orange.data import Variable, DiscreteVariable, Domain
from Orange.classification import Model


__all__ = ["ColumnClassifier"]


class ColumnClassifier(Model):
    def __init__(self, class_var: DiscreteVariable, column: Variable,
                 k: Optional[float] = None):
        assert class_var.is_discrete
        assert column.is_continuous and len(class_var.values) == 2 or \
            column.is_discrete and len(class_var.values) == len(column.values)
        super().__init__(Domain([column], class_var))
        self.column = column
        self.k = k
        self.name = column.name

    def predict_storage(self, data):
        vals = data.get_column(self.column)
        if self.column.is_discrete:
            proba = np.zeros((len(data), len(self.column.values)))
            rows = np.isfinite(vals)
            proba[rows, vals[rows].astype(int)] = 1
        else:
            proba = np.zeros((len(data), 2))
            if self.k is None:
                if np.nanmin(vals) < 0 or np.nanmax(vals) > 1:
                    raise ValueError("Column values must be in [0, 1] range "
                                     "unless logistic function is applied")
                proba[:, 1] = vals
                proba[:, 0] = 1 - vals
                vals = vals > 0.5
            else:
                proba[:, 1] = 1 / (1 + np.exp(-self.k * vals))
                proba[:, 0] = 1 - proba[:, 1]
                vals = vals > 0
        return vals, proba

    def __str__(self):
        return f'ColumnClassifier {self.column.name}'
