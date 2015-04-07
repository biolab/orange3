from numbers import Real
from math import isnan

#: A constant representing unknown value (NaN). Use this for storing unknowns,
#: but not for checking for unknowns.
Unknown = float("nan")


class Value(float):
    """
    The class representing a value. The class is not used to store values but
    only to return them in contexts in which we want the value to be accompanied
    with the descriptor, for instance to print the symbolic value of discrete
    variables.

    The class is derived from `float`, with an additional attribute `variable`
    which holds the descriptor of type :obj:`Orange.data.Variable`. If the
    value continuous or discrete, it is stored as a float. Other types of
    values, like strings, are stored in the attribute `value`.

    The class overloads the methods for printing out the value:
    `variable.repr_val` and `variable.str_val` are used to get a suitable
    representation of the value.

    Equivalence operator is overloaded as follows:

    - unknown values are equal; if one value is unknown and the other is not,
      they are different;

    - if the value is compared with the string, the value is converted to a
      string using `variable.str_val` and the two strings are compared

    - if the value is stored in attribute `value`, it is compared with the
      given other value

    - otherwise, the inherited comparison operator for `float` is called.

    Finally, value defines a hash, so values can be put in sets and appear as
    keys in dictionaries.

    .. attribute:: variable (:obj:`Orange.data.Variable`)

        Descriptor; used for printing out and for comparing with strings

    .. attribute:: value

        Value; the value can be of arbitrary type and is used only for variables
        that are neither discrete nor continuous. If `value` is `None`, the
        derived `float` value is used.
    """
    __slots__ = "variable", "_value"

    def __new__(cls, variable, value=Unknown):
        """
        Construct a new instance of Value with the given descriptor and value.
        If the argument `value` can be converted to float, it is stored as
        `float` and the attribute `value` is set to `None`. Otherwise, the
        inherited float is set to `Unknown` and the value is held by the
        attribute `value`.

        :param variable: descriptor
        :type variable: Orange.data.Variable
        :param value: value
        """
        if not isinstance(value, str):
            try:
                self = super().__new__(cls, value)
                self.variable = variable
                return self
            except:
                pass
        self = super().__new__(cls, -1)
        self._value = value
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
        if isinstance(other, Value):
            return self.value == other.value
        return super().__eq__(other)

    def __contains__(self, other):
        if (self.value is not None
                and isinstance(self.value, str)
                and isinstance(other, str)):
            return other in self.value
        raise TypeError("invalid operation on Value()")

    def __hash__(self):
        if self.value is None:
            return super().__hash__()
        else:
            return super().__hash__() ^ hash(self.value)

    @property
    def value(self):
        from . import DiscreteVariable, StringVariable
        if isinstance(self.variable, DiscreteVariable):
            return self.variable.values[int(self)]
        if isinstance(self.variable, StringVariable):
            return self._value
        return float(self)

    def __getnewargs__(self):
        return self.variable, float(self)

    def __getstate__(self):
        return dict(value=getattr(self, '_value', None))

    def __setstate__(self, state):
        self._value = state.get('value', None)
