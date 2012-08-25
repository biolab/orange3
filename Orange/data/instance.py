from numbers import Real
from ..data.value import Value, Unknown
import numpy as np
from math import isnan

class Instance:
    def __init__(self, domain, values=None):
        # First handle copy constructor
        if values is None and isinstance(domain, Instance):
            values = domain
            domain = domain.domain

        if values is not None:
            self.domain = None
            domain.convert(values, self)
        else:
            self.domain = domain
            self._values = []
        self._metas = [Unknown] * len(self.domain.metas)
        self.weight = 1

    def __iter__(self):
        return (Value(var, value)
                for var, value in zip(self.domain.variables, self._values))

    def attributes(self):
        return (Value(var, value)
                for var, value in zip(self.domain.attributes, self._values))

    def variables(self):
        return self.__iter__()

    def __getitem__(self, key):
        if not isinstance(key, int):
            key = self.domain.index(key)
        if key >= 0:
            value = self._values[key]
        else:
            value = self._metas[-1-key]
        return Value(self.domain[key], value)

    def __str__(self):
        res = "["
        res += ", ".join(var.str_val(value) for var, value in
                         zip(self.domain.attributes, self._values[:5]))
        n_attrs = len(self.domain.attributes)
        if n_attrs > 5:
            res += ", ..."
        if self.domain.class_vars:
            res += " | " + ", ".join(var.str_val(value) for var, value in
                zip(self.domain.class_vars, self._values[n_attrs:n_attrs+5]))
        res += "]"
        if self.domain.metas:
            res += " {"
            res += ", ".join(var.str_val(value)
                for var, value in zip(self.domain.metas, self._metas[:5]))
            if len(self._metas) > 5:
                res += ", ..."
            res += "}"
        return res

    __repr__ = __str__


    def get_class(self):
        if self.domain.class_var:
            return Value(self.domain.class_var, self._values[-1])
        elif not self.domain.class_vars:
            raise TypeError("Domain has no class variable")
        else:
            raise TypeError("Domain has multiple class variables")

    def get_classes(self):
        return (Value(var, value) for var, value in
            zip(self.domain.class_vars, self._values[:len(self.domain.attributes)]))

    def set_weight(self, weight):
        self.weight = weight

    def get_weight(self):
        return self.weight

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            key = self.domain.index(key)
        if key >= 0:
            if not isinstance(value, float):
                raise TypeError("Expected primitive value, got '%s'" %
                                type(value).__name__)
            self._values[key] = value
        else:
            self._metas[-1-key] = value

    def __eq__(self, other):
        # TODO: rewrite to Cython
        if not isinstance(other, Instance):
            other = Instance(self.domain, other)
        for v1, v2 in zip(self._values, other._values):
            if not (isnan(v1) or isnan(v2) or v1 == v2):
                return False
        for m1, m2 in zip(self._metas, other._metas):
            if not (isnan(m1) or isnan(m2) or m1 == m2):
                return False
        return True



class RowInstance(Instance):
    def __init__(self, table, row_index):
        super().__init__(table.domain)
        if table._X is not None:
            self._x = table._X[row_index]
            self._values = list(self._x)
        else:
            self._x = None
            self._values = []
        if table._Y is not None:
            self._y = table._Y[row_index]
            self._values += list(self._y)
        else:
            self._y = None
        self.row_index = row_index
        self._metas = table._metas is not None and table._metas[row_index]
        self.table = table

    def _check_single_class(self):
        if not self.domain.class_vars:
            raise TypeError("Domain has no class variable")
        elif len(self.domain.class_vars) > 1:
            raise TypeError("Domain has multiple class variables")

    def get_class(self):
        self._check_single_class()
        if self.table.domain.class_var:
            return Value(self.table.domain.class_var, self._y[0])

    def set_class(self, value):
        self._check_single_class()
        if not isinstance(value, Real):
            self._y[0] = self.table.domain.class_var.to_val(value)
        else:
            self._y[0] = value

    def get_classes(self):
        return (Value(var, value) for var, value in
            zip(self.table.domain.class_vars, self._y))

    def set_weight(self, weight):
        if self.table._W is None:
            self.table.set_weights()
        self.table._W[self.row_index] = weight

    def get_weight(self):
        if not self.table._W:
            raise ValueError("Instances in the referenced table have no weights")
        return self.table._W[self.row_index]

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            key = self.domain.index(key)
        if isinstance(value, str):
            var = self.domain[key]
            value = var.to_val(value)
        if key >= 0:
            if not isinstance(value, Real):
                raise TypeError("Expected primitive value, got '%s'" %
                                type(value).__name__)
            if key < len(self._x):
                self._x[key] = value
            else:
                self._y[key - len(self._x)] = value
        else:
            self._metas[-1-key] = value
