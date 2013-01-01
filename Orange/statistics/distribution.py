import random
import zlib
import math
from numbers import Real
import numpy as np
from Orange import data


#TODO: Handle sparse data -- at least by get_distributions

def _get_variable(variable, dat, expected_type=None, expected_name=""):
    if isinstance(variable, data.Variable):
        datvar = getattr(dat, "variable", None)
        if datvar is not None and datvar is not variable:
            raise ValueError("variable does not match the variable"
                             "in the data")
    else:
        if hasattr(dat, "domain"):
            variable = dat.domain[variable]
        if hasattr(dat, "variable"):
            variable = dat.variable
    if expected_type is not None and not isinstance(variable, expected_type):
        if isinstance(variable, data.Variable):
            raise ValueError(
                "expected %s variable not %s" % (expected_name, variable))
        else:
            raise ValueError("expected expected, not '%s'" %
                             (expected_type.__name, type(variable).__name__))
    return variable


class Discrete(np.ndarray):
    def __new__(cls, variable, dat=None, unknowns=None):
        if isinstance(dat, data.Storage):
            if unknowns is not None:
                raise TypeError(
                    "incompatible arguments (data storage and 'unknowns'")
            return cls.from_data(variable, dat)

        if variable is not None:
            variable = _get_variable(variable, dat)
            n = len(variable.values)
        else:
            n = len(dat)

        self = super().__new__(cls, n)
        self.variable = variable
        if dat is None:
            self[:] = 0
            self.unknowns = unknowns or 0
        else:
            self[:] = dat
            self.unknowns = (unknowns if unknowns is not None
                             else getattr(dat, "unknowns", 0))
        return self


    @classmethod
    def from_data(cls, variable, data):
        variable = _get_variable(variable, data)
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
        if t > 1e-6:
            self[:] /= t
            self.unknowns /= t
        elif self.shape[0]:
            self[:] = 1 / self.shape[0]


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
    def __new__(cls, variable, dat, unknowns=None):
        if isinstance(dat, data.Storage):
            if unknowns is not None:
                raise TypeError(
                    "incompatible arguments (data storage and 'unknowns'")
            return cls.from_data(variable, dat)
        if isinstance(dat, int):
            self = super().__new__(cls, (2, dat))
            self[:] = 0
            self.unknowns = unknowns or 0
        else:
            if not isinstance(dat, np.ndarray):
                dat = np.asarray(dat)
            self = super().__new__(cls, dat.shape)
            self[:] = dat
            self.unknowns = (unknowns if unknowns is not None
                             else getattr(dat, "unknowns", 0))
        self.variable = variable
        return self


    @classmethod
    def from_data(cls, variable, data):
        variable = _get_variable(variable, data)
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

        self = super().__new__(cls, dist.shape)
        self[:] = dist
        self.unknowns = unknowns
        self.variable = variable
        return self


    def __eq__(self, other):
        return np.array_equal(self, other) and (
            not hasattr(other, "unknowns") or self.unknowns == other.unknowns)


    def __hash__(self):
        return zlib.adler32(self) ^ hash(self.unknowns)


    def normalize(self):
        t = np.sum(self[1, :])
        if t > 1e-6:
            self[1, :] /= t
            self.unknowns /= t
        elif self.shape[1]:
            self[1, :] = 1 / self.shape[1]


    def modus(self):
        val = np.argmax(self[1, :])
        return self[0, val]


    def random(self):
        v = random.random() * np.sum(self[1, :])
        s = 0
        for x, prob in self.T:
            s += prob
            if s > v:
                return x


def class_distribution(data):
    if data.domain.class_var:
        return get_distribution(data.domain.class_var, data)
    elif data.domain.class_vars:
        return [get_distribution(cls, data) for cls in data.domain.class_vars]
    else:
        raise ValueError("domain has no class attribute")


def get_distribution(variable, dat, unknowns=None):
    variable = _get_variable(variable, dat)
    if isinstance(variable, data.DiscreteVariable):
        return Discrete(variable, dat, unknowns)
    elif isinstance(variable, data.ContinuousVariable):
        return Continuous(variable, dat, unknowns)
    else:
        raise TypeError("cannot compute distribution of '%s'" %
                        type(variable).__name__)


def get_distributions(dat, skipDiscrete=False, skipContinuous=False):
    vars = dat.domain.variables
    if skipDiscrete:
        if skipContinuous:
            return []
        columns = [i for i, var in enumerate(vars)
                   if isinstance(var, data.ContinuousVariable)]
    elif skipContinuous:
        columns = [i for i, var in enumerate(vars)
                   if isinstance(var, data.DiscreteVariable)]
    else:
        columns = None
    try:
        dist_unks = dat._compute_distributions(columns)
        if columns is None:
            columns = np.arange(len(vars))
        distributions = []
        for col, (dist, unks) in zip(columns, dist_unks):
            distributions.append(get_distribution(vars[col], dist, unks))
    except NotImplementedError:
        if columns is None:
            columns = np.arange(len(vars))
        distributions = [get_distribution(i, dat) for i in columns]
    return distributions
