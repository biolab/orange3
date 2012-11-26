from ..data.value import Value, Unknown
from math import isnan
import zlib
import numpy as np


class Instance:
    def __init__(self, domain, values=None):
        self.domain = domain

        if values is None:
            if isinstance(domain, Instance):
                values = domain
                self.domain = domain = domain.domain
            else:
                self._values = np.repeat(Unknown, len(domain.variables))
                self._x = self._values[:len(domain.attributes)]
                self._y = self._values[len(domain.attributes):]
                self._metas = np.array(
                    [Unknown if var.is_primitive else None for var in
                     domain.metas],
                    dtype=object)
                self.weight = 1
                return

        if isinstance(values, Instance) and values.domain == domain:
            self._values = np.array(values._values)
            self._metas = np.array(values._metas)
            self.weight = values.weight
        else:
            self._values, self._metas = domain.convert(values)
            self.weight = 1
        self._x = self._values[:len(domain.attributes)]
        self._y = self._values[len(domain.attributes):]

    def variables(self):
        return iter(self._values)

    __iter__ = variables

    def attributes(self):
        return iter(self._values[:len(self.domain.attributes)])

    def classes(self):
        return iter(self._y)

    def __getitem__(self, key):
        if not isinstance(key, int):
            key = self.domain.index(key)
        if key >= 0:
            value = self._values[key]
        else:
            value = self._metas[-1 - key]
        return Value(self.domain[key], value)

    #TODO Should we return an instance of `object` if we have a meta attribute
    #     that is not Discrete or Continuous? E.g. when we have strings, we'd
    #     like to be able to use startswith, lower etc...
    #     Or should we even return Continuous as floats and use Value only
    #     for discrete attributes?!
    #     Same in Table.__getitem__

    def __str__(self):
        res = "["
        res += ", ".join(
            var.str_val(value) for var, value in zip(self.domain.attributes,
                                                     self._x[:5]))
        n_attrs = len(self.domain.attributes)
        if n_attrs > 5:
            res += ", ..."
        if self.domain.class_vars:
            res += " | " + ", ".join(
                var.str_val(value) for var, value in zip(self.domain.class_vars,
                                                         self._y[:5]))
        res += "]"
        if self.domain.metas:
            res += " {"
            res += ", ".join(
                var.str_val(value) for var, value in zip(self.domain.metas,
                                                         self._metas[:5]))
            if len(self._metas) > 5:
                res += ", ..."
            res += "}"
        return res

    __repr__ = __str__

    def _check_single_class(self):
        if not self.domain.class_vars:
            raise TypeError("Domain has no class variable")
        elif len(self.domain.class_vars) > 1:
            raise TypeError("Domain has multiple class variables")

    def get_class(self):
        self._check_single_class()
        return Value(self.domain.class_var, self._y[0])

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
            self._metas[-1 - key] = value

    def __eq__(self, other):
        # TODO: rewrite to Cython
        if not isinstance(other, Instance):
            other = Instance(self.domain, other)
        for v1, v2 in zip(self._values, other._values):
            if not (isnan(v1) or isnan(v2) or v1 == v2):
                return False
        for m1, m2 in zip(self._metas, other._metas):
            if not (m1 == m2 or isnan(m1) or m1 is None or isnan(m2) or m2 is None):
                return False
        return True

    def checksum(self):
        return zlib.adler32(self._metas, zlib.adler32(self._values))
