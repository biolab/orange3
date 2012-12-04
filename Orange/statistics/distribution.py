import random
import zlib
import math
from numbers import Real
import numpy as np
from Orange import data

class Discrete(np.ndarray):
    def __new__(cls, variable, dat=None):
        if isinstance(dat, data.Storage):
            return cls.from_data(variable, dat)

        if variable is not None:
            variable = cls._get_variable(variable, dat)
            n = len(variable.values)
        else:
            n = len(dat)

        self = super().__new__(cls, n)
        self.variable = variable
        if dat is None:
            self[:] = self.unknowns = 0
        else:
            self[:] = dat
            self.unknowns = getattr(dat, "unknowns", 0)
        return self


    @staticmethod
    def _get_variable(variable, dat):
        if isinstance(variable, data.Variable):
            datvar = getattr(dat, "variable", None)
            if datvar is not None and datvar is not variable:
                raise ValueError("variable does not match the variable "
                                 "in the data")
        else:
            if hasattr(dat, "domain"):
                variable = dat.domain[variable]
            if hasattr(dat, "variable"):
                variable = dat.variable
        if not isinstance(variable, data.DiscreteVariable):
            if isinstance(variable, data.Variable):
                raise ValueError(
                    "expected discrete variable not %s" % variable)
            else:
                raise ValueError("expected DiscreteVariable, not '%s'" %
                                 type(variable).__name__)
        return variable


    @classmethod
    def from_data(cls, variable, data):
        variable = cls._get_variable(variable, data)
        try:
            dist, unknowns = data._compute_distributions([variable])[0]
            self = super().__new__(cls, len(dist))
            self[:] = dist
            self.unknowns = unknowns
        except NotImplementedError:
            self = np.zeros(len(variable.values))
            self.unknowns = 0
            if data.has_weights():
                for val, w in zip(data[:, variable], data.W):
                    if not math.isnan(val):
                        self[val] += w
                    else:
                        self.unknowns += w
            else:
                for inst in data:
                    val = inst[variable]
                    if val == val:
                        self[val] += 1
                    else:
                        self.unknowns += 1
        self.variable = variable
        return self


    def __eq__(self, other):
        return np.array_equal(self, other) and (
            not hasattr(other, "unknowns") or self.unknowns == other.unknowns)


    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.variable.to_val(index)
        return super().__getitem__(index)


    def __setitem__(self, index, value):
        if isinstance(index, str):
            index = self.variable.to_val(index)
        super().__setitem__(index, value)


    def __hash__(self):
        return zlib.adler32(self) ^ hash(self.unknowns)


    def __add__(self, other):
        s = super().__add__(other)
        s.unknowns = self.unknowns + getattr(other, "unknowns", 0)
        return s


    def __iadd__(self, other):
        super().__iadd__(other)
        self.unknowns += getattr(other, "unknowns", 0)
        return self


    def __sub__(self, other):
        s = super().__sub__(other)
        s.unknowns = self.unknowns - getattr(other, "unknowns", 0)
        return s


    def __isub__(self, other):
        super().__isub__(other)
        self.unknowns -= getattr(other, "unknowns", 0)
        return self


    def __mul__(self, other):
        s = super().__mul__(other)
        if isinstance(other, Real):
            s.unknowns = self.unknowns / other
        return s


    def __imul__(self, other):
        super().__imul__(other)
        if isinstance(other, Real):
            self.unknowns *= other
        return self


    def __div__(self, other):
        s = super().__mul__(other)
        if isinstance(other, Real):
            s.unknowns = self.unknowns / other
        return s


    def __idiv__(self, other):
        super().__imul__(other)
        if isinstance(other, Real):
            self.unknowns /= other
        return self


    def normalize(self):
        t = np.sum(self)
        self[:] /= t
        self.unknowns /= t


    def modus(self):
        val = np.argmax(self)
        return data.Value(self.variable,
                          val) if self.variable is not None else val


    def random(self):
        v = random.random() * np.sum(self)
        s = i = 0
        for i, e in enumerate(self):
            s += e
            if s > v:
                break
        return data.Value(self.variable, i) if self.variable is not None else i


class Continuous(np.ndarray):
    def __new__(cls, variable, dat=None):
        if dat is None:
            raise ValueError(
                "continuous distribution cannot be constructed without data")
        if isinstance(dat, data.Storage):
            return cls.from_data(variable, dat)
        if isinstance(dat, int):
            self = super().__new__(cls, dat)
            self[:] = self.unknowns = 0
        else:
            self = super().__new__(cls, dat)
            self.unknowns = getattr(dat, "unknowns", 0)
        self.variable = variable
        return self


    @staticmethod
    def _get_variable(variable, dat):
        if isinstance(variable, data.Variable):
            datvar = getattr(dat, "variable", None)
            if datvar is not None and datvar is not variable:
                raise ValueError("variable does not match the variable "
                                 "in the data")
        else:
            if hasattr(dat, "domain"):
                variable = dat.domain[variable]
            if hasattr(dat, "variable"):
                variable = dat.variable
        if not isinstance(variable, data.ContinuousVariable):
            if isinstance(variable, data.Variable):
                raise ValueError(
                    "expected discrete variable not %s" % variable)
            else:
                raise ValueError("expected DiscreteVariable, not '%s'" %
                                 type(variable).__name__)
        return variable


    @classmethod
    def from_data(cls, variable, data):
        variable = cls._get_variable(variable, data)
        try:
            dist, unknowns = data._compute_distributions([variable])[0]
        except NotImplementedError:
            col = data[:, variable]
            dtype = col.dtype
            if data.has_weights():
                if not "float" in dtype.name and "float" in col.dtype.name:
                    dtype = col.dtype.name
                dist = np.empty((2, len(col)), dtype=dtype)
                dist[0, :] = col
                dist[1, :] = data.W
            else:
                dist = np.ones((2, len(col)), dtype=dtype)
                dist[0, :] = col
            dist.sort(axis=0)
            dist = np.array(_orange.valuecount(dist))
            unknowns = len(col) - dist.shape[1]

        self = super().__new__(cls, len(dist))
        self[:] = dist
        self.unknowns = unknowns
        self.variable = variable
        return self


    def __eq__(self, other):
        return np.array_equal(self, other) and (
            not hasattr(other, "unknowns") or self.unknowns == other.unknowns)


    def __getitem__(self, index):
        i = np.searchsorted(self[0, :], index)
        if super().__getitem__((0, i)) != index:
            raise KeyError("distribution does not have value %.3f" % index)
        return super().__getitem__((1, i))


    def __setitem__(self, index, value):
        i = np.searchsorted(self[0, :], index)
        if super().__setitem__((0, i)) != index:
            raise KeyError("distribution does not have value %.3f" % index)
        super().__setitem__((1, i), value)


    def __hash__(self):
        return zlib.adler32(self) ^ hash(self.unknowns)


    def __add__(self, other):
        raise TypeError("continuous distribution does not implement addition")


    def __iadd__(self, other):
        raise TypeError("continuous distribution does not implement addition")


    def __sub__(self, other):
        raise TypeError(
            "continuous distribution does not implement subtraction")


    def __isub__(self, other):
        raise TypeError(
            "continuous distribution does not implement subtraction")


    def __mul__(self, other):
        if isinstance(other, Real):
            dist = self.__class__(self)
            super().__mul__(dist[1, :], other)
            dist.unknowns *= other
            return dist
        else:
            raise TypeError("cannot multiply continuous distributions")


    def __imul__(self, other):
        if isinstance(other, Real):
            super().__mul__(self[1, :], other)
            self.unknowns *= other
            return self
        else:
            raise TypeError("cannot multiply continuous distributions")


    def __div__(self, other):
        if isinstance(other, Real):
            dist = self.__class__(self)
            super().__div__(dist[1, :], other)
            dist.unknowns /= other
            return dist
        else:
            raise TypeError("cannot divide continuous distributions")


    def __idiv__(self, other):
        if isinstance(other, Real):
            super().__div__(self[1, :], other)
            self.unknowns /= other
            return self
        else:
            raise TypeError("cannot divide continuous distributions")


    def normalize(self):
        t = np.sum(self)
        self[1, :] /= t
        self.unknowns /= t


    def modus(self):
        val = np.argmax(self[1, :])
        return self[0, val]


    def random(self):
        v = random.random() * np.sum(self[1, :])
        s = 0
        for prob, x in enumerate(self.T):
            s += prob
            if s > v:
                return x


def class_distribution(data):
    nattrs = len(data.domain.attributes)
    if data.domain.class_var:
        return Discrete(nattrs, data)
    return [Discrete(nattrs + i, data)
            for i in range(len(data.domain.class_vars))]
