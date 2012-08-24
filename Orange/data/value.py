from numbers import Real
from math import isnan
Unknown = float("nan")

class Value(float):
    __slots__ = "variable", "value"
    def __new__(cls, variable, value=Unknown):
        if isinstance(value, Real):
            self = super().__new__(cls, value)
            self.value = None
        else:
            self = super().__new__(cls, -1)
            self.value = value
        self.variable = variable
        return self

    def __init__(self, _, __=Unknown):
        pass

    def __repr__(self):
        return "Value('%s', %s)" % (self.variable.name,
                                    self.variable.repr_val(self))

    def __str__(self):
        return self.variable.str_val(self)

    def __eq__(self, other):
        if isinstance(self, Real) and isnan(self):
            return isinstance(other, Real) and isnan(other) or \
                   other in self.variable.unknown_str
        if isinstance(other, str):
            return self.variable.str_val(self) == other
        if self.value:
            if isinstance(other, Value) and other.value:
                other = other.value
            return self.value == other
        return super().__eq__(other)
"""
Remove when implemented (or when decided to not reimplement)

from orange import \
              Distribution, \
                   ContDistribution, \
                   DiscDistribution, \
                   GaussianDistribution, \
"""