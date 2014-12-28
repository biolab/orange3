from numbers import Real
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
        self.sparse_x = self.sparse_y = self.sparse_metas = None
        if data is None:
            self._values = np.repeat(Unknown, len(domain.variables))
            self._metas = np.array([var.Unknown for var in domain.metas],
                                   dtype=object)
            self._weight = 1
        elif isinstance(data, Instance) and data.domain == domain:
            self._values = np.array(data._values)
            self._metas = np.array(data._metas)
            self._weight = data._weight
        else:
            self._values, self._metas = domain.convert(data)
            self._weight = 1
        self._x = self._values[:len(domain.attributes)]
        self._y = self._values[len(domain.attributes):]

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
    def weight(self):
        """The weight of the data instance. Default is 1."""
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = weight

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            key = self._domain.index(key)
        value = self._domain[key].to_val(value)
        if key >= 0:
            if not isinstance(value, (int, float)):
                raise TypeError("Expected primitive value, got '%s'" %
                                type(value).__name__)
            self._values[key] = value
        else:
            self._metas[-1 - key] = value

    def __getitem__(self, key):
        if not isinstance(key, int):
            key = self._domain.index(key)
        if key >= 0:
            value = self._values[key]
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
    def str_values(data, variables):
        s = ", ".join(var.str_val(val)
            for var, val in zip(variables, data[:5]))
        if len(data) > 5:
            s += ", ..."
        return s

    def __str__(self):
        s = "[" + self.str_values(self._x, self._domain.attributes)
        if self._domain.class_vars:
            s += " | " + self.str_values(self._y, self._domain.class_vars)
        s += "]"
        if self._domain.metas:
            s += " {" + self.str_values(self._metas, self._domain.metas) + "}"
        return s

    __repr__ = __str__

    def __eq__(self, other):
        if not isinstance(other, Instance):
            other = Instance(self._domain, other)
        nan1 = np.isnan(self._values)
        nan2 = np.isnan(other._values)
        return np.array_equal(nan1, nan2) and \
            np.array_equal(self._values[~nan1], other._values[~nan2]) \
            and all(m1 == m2 or
                    type(m1) == type(m2) == float and isnan(m1) and isnan(m2)
                    for m1, m2 in zip(self._metas, other._metas))

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def attributes(self):
        """Return iterator over the instance's attributes"""
        return iter(self._values[:len(self._domain.attributes)])

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
        self._values[len(self._domain.attributes)] = self._y[0]
