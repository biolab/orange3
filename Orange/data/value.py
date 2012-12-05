from numbers import Real
from math import isnan

Unknown = float("nan")


class Value(float):
    __slots__ = "variable", "value"

    def __new__(cls, variable, value=Unknown):
        if not isinstance(value, str):
            try:
                self = super().__new__(cls, value)
                self.variable = variable
                self.value = None
                return self
            except:
                pass
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
            return (isinstance(other, Real) and isnan(other)
                    or other in self.variable.unknown_str)
        if isinstance(other, str):
            return self.variable.str_val(self) == other
        if self.value:
            if isinstance(other, Value) and other.value:
                other = other.value
            return self.value == other
        return super().__eq__(other)

    def __contains__(self, other):
        if (self.value is not None
                and isinstance(self.value, str)
                and isinstance(other, str)):
            return other in self.value
        raise TypeError("invalid operation on Value()")

    def __hash__(self):
        if self.value is None:
            return super().__hash__(self)
        else:
            return super().__hash__(self) ^ hash(self.value)

"""
Remove when implemented (or when decided to not reimplement)

from orange import \
              Distribution, \
                   ContDistribution, \
                   DiscDistribution, \
                   GaussianDistribution, \
"""
