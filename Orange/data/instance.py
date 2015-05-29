from itertools import chain
from numbers import Real, Integral
from ..data.value import Value, Unknown
from math import isnan
import numpy as np


class Instance:
    def __init__(self, domain, data=None):
        """
        Construct a new data instance.

        :param domain: domain that describes the instance's variables
        :type domain: Orange.data.Domain
        :param data: instance's values
        :type data: Orange.data.Instance or a sequence of values
        """
        if data is None and isinstance(domain, Instance):
            data = domain
            domain = data.domain

        self._domain = domain
        if data is None:
            self._x = np.repeat(Unknown, len(domain.attributes))
            self._y = np.repeat(Unknown, len(domain.class_vars))
            self._metas = np.array([var.Unknown for var in domain.metas],
                                   dtype=object)
            self._weight = 1
        elif isinstance(data, Instance) and data.domain == domain:
            self._x = np.array(data._x)
            self._y = np.array(data._y)
            self._metas = np.array(data._metas)
            self._weight = data._weight
        else:
            self._x, self._y, self._metas = domain.convert(data)
            self._weight = 1

    @property
    def domain(self):
        """The domain describing the instance's values."""
        return self._domain

    @property
    def x(self):
        """
        Instance's attributes as a 1-dimensional numpy array whose length
        equals `len(self.domain.attributes)`.
        """
        return self._x

    @property
    def y(self):
        """
        Instance's classes as a 1-dimensional numpy array whose length
        equals `len(self.domain.attributes)`.
        """
        return self._y

    @property
    def metas(self):
        """
        Instance's meta attributes as a 1-dimensional numpy array whose length
        equals `len(self.domain.attributes)`.
        """
        return self._metas

    @property
    def list(self):
        """
        All instance's values, including attributes, classes and meta
        attributes, as a list whose length equals `len(self.domain.attributes)
        + len(self.domain.class_vars) + len(self.domain.metas)`.
        """
        n_self, n_metas = len(self), len(self._metas)
        return [self[i].value if i < n_self else self[n_self - i - 1].value
                for i in range(n_self + n_metas)]

    @property
    def weight(self):
        """The weight of the data instance. Default is 1."""
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = weight

    def __setitem__(self, key, value):
        if not isinstance(key, Integral):
            key = self._domain.index(key)
        value = self._domain[key].to_val(value)
        if key >= 0 and not isinstance(value, (int, float)):
            raise TypeError("Expected primitive value, got '%s'" %
                            type(value).__name__)

        if 0 <= key < len(self._domain.attributes):
            self._x[key] = value
        elif len(self._domain.attributes) <= key:
            self._y[key - len(self.domain.attributes)] = value
        else:
            self._metas[-1 - key] = value

    def __getitem__(self, key):
        if not isinstance(key, Integral):
            key = self._domain.index(key)
        if 0 <= key < len(self._domain.attributes):
            value = self._x[key]
        elif key >= len(self._domain.attributes):
            value = self._y[key - len(self.domain.attributes)]
        else:
            value = self._metas[-1 - key]
        return Value(self._domain[key], value)

    #TODO Should we return an instance of `object` if we have a meta attribute
    #     that is not Discrete or Continuous? E.g. when we have strings, we'd
    #     like to be able to use startswith, lower etc...
    #     Or should we even return Continuous as floats and use Value only
    #     for discrete attributes?!
    #     Same in Table.__getitem__

    @staticmethod
    def str_values(data, variables, limit=True):
        if limit:
            s = ", ".join(var.str_val(val)
                          for var, val in zip(variables, data[:5]))
            if len(data) > 5:
                s += ", ..."
            return s
        else:
            return ", ".join(var.str_val(val)
                             for var, val in zip(variables, data))

    def _str(self, limit):
        s = "[" + self.str_values(self._x, self._domain.attributes, limit)
        if self._domain.class_vars:
            s += " | " + \
                 self.str_values(self._y, self._domain.class_vars, limit)
        s += "]"
        if self._domain.metas:
            s += " {" + \
                 self.str_values(self._metas, self._domain.metas, limit) + \
                 "}"
        return s

    def __str__(self):
        return self._str(False)

    def __repr__(self):
        return self._str(True)

    def __eq__(self, other):
        if not isinstance(other, Instance):
            other = Instance(self._domain, other)

        def same(x1, x2):
            nan1 = np.isnan(x1)
            nan2 = np.isnan(x2)
            return np.array_equal(nan1, nan2) and \
                   np.array_equal(x1[~nan1], x2[~nan2])

        return same(self._x, other._x) and same(self._y, other._y) \
               and all(m1 == m2 or
                       type(m1) == type(m2) == float and isnan(m1) and isnan(m2)
                       for m1, m2 in zip(self._metas, other._metas))

    def __iter__(self):
        return chain(iter(self._x), iter(self._y))

    def values(self):
        return (Value(var, val)
                for var, val in zip(self.domain.variables, self))

    def __len__(self):
        return len(self._x) + len(self._y)

    def attributes(self):
        """Return iterator over the instance's attributes"""
        return iter(self._x)

    def classes(self):
        """Return iterator over the instance's class attributes"""
        return iter(self._y)

    # A helper function for get_class and set_class
    def _check_single_class(self):
        if not self._domain.class_vars:
            raise TypeError("Domain has no class variable")
        elif len(self._domain.class_vars) > 1:
            raise TypeError("Domain has multiple class variables")

    def get_class(self):
        """
        Return the class value as an instance of :obj:`Orange.data.Value`.
        Throws an exception if there are multiple classes.
        """
        self._check_single_class()
        return Value(self._domain.class_var, self._y[0])

    def get_classes(self):
        """
        Return the class value as a list of instances of
        :obj:`Orange.data.Value`.
        """
        return (Value(var, value)
                for var, value in zip(self._domain.class_vars, self._y))

    def set_class(self, value):
        """
        Set the instance's class. Throws an exception if there are multiple
        classes.
        """
        self._check_single_class()
        if not isinstance(value, Real):
            self._y[0] = self._domain.class_var.to_val(value)
        else:
            self._y[0] = value
