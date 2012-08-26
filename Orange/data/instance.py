from ..data.value import Value, Unknown
from .domain import Domain
from math import isnan

class Instance:
    def __init__(self, domain, values=None):
        # First handle copy constructor
        if values is None:
            if isinstance(domain, Instance):
                self.domain = domain.domain
                self._values = list(domain._values)
                self._metas = list(domain._metas)
                self.weight = domain.weight
                return
            elif isinstance(domain, Domain):
                self.domain = domain
                self._values = [Unknown] * len(domain.variables)
                self._metas =  [Unknown] * len(domain.metas)
                self.weight = 1
                return
        else:
            self.domain = domain
            attributes, classes, metas = domain.convert_as_list(values)
            self._values = attributes + classes
            self._metas = metas
            self.weight = 1
            return
        raise TypeError("Expected an instance of Domain, not '%s'",
            domain.__class__.name)


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
