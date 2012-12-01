import random
import zlib
from numbers import Real
import numpy as np
from Orange.data import DiscreteVariable, Variable, Storage, Value

class Discrete(np.ndarray):
    def __new__(cls, variable, data=None):
        if variable is not None:
            if not isinstance(variable, Variable):
                if hasattr(data, "domain"):
                    variable = data.domain[variable]
                if hasattr(data, "variable"):
                    variable = data.variable
            Discrete._check_var_type(variable)
            n = len(variable.values)
        else:
            if isinstance(data, Storage):
                raise ValueError("variable is not specified")
            n = len(data)

        self = super().__new__(cls, n)
        self.variable = variable
        if isinstance(data, Storage):
            self.compute(variable, data)
        elif data is None:
            self[:] = self.unknowns = 0
        else:
            self[:] = data
            self.unknowns = getattr(data, "unknowns", 0)
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


    @staticmethod
    def _check_var_type(var):
        if not isinstance(var, DiscreteVariable):
            if isinstance(var, Variable):
                raise ValueError("expected discrete variable not %s" % var)
            else:
                raise ValueError("expected DiscreteVariable, not '%s'" %
                    type(var).__name__)


    def compute(self, variable, data):
        var = data.domain[variable]
        Discrete._check_var_type(var)
        self.variable = variable
        try:
            self[:], self.unknowns = data._compute_distributions([variable])[0]
            return
        except NotImplementedError:
            pass

        self[:] = self.unknowns = 0
        if data.has_weights():
            for val, w in zip(data[:, variable], data.W):
                if val == val:
                    self[val] += w
                else:
                    self.unknowns += w
        else:
            for inst in data:
                val = inst[var]
                if val == val:
                    self[val] += 1
                else:
                    self.unknowns += 1


    def normalize(self):
        self /= np.sum(self)


    def modus(self):
        val = np.argmax(self)
        return Value(self.variable, val) if self.variable is not None else val


    def random(self):
        v = random.random() * np.sum(self)
        s = 0
        for i, e in enumerate(self):
            s += e
            if s > v:
                break
        return Value(self.variable, i) if self.variable is not None else i

def class_distribution(data):
    nattrs = len(data.domain.attributes)
    if data.domain.class_var:
        return Discrete(nattrs, data)
    return [Discrete(nattrs+i, data)
            for i in range(len(data.domain.class_vars))]
