import re
import warnings
from collections.abc import Iterable

from datetime import datetime, timedelta, timezone
from numbers import Number, Real, Integral
from math import isnan, floor
from pickle import PickleError

import numpy as np
import scipy.sparse as sp

from Orange.data import _variable
from Orange.util import Registry, Reprable, OrangeDeprecationWarning


__all__ = ["Unknown", "MISSING_VALUES", "make_variable", "is_discrete_values",
           "Value", "Variable", "ContinuousVariable", "DiscreteVariable",
           "StringVariable", "TimeVariable"]


# For storing unknowns
Unknown = ValueUnknown = float("nan")
# For checking for unknowns
MISSING_VALUES = {np.nan, "?", "nan", ".", "", "NA", "~", None}

DISCRETE_MAX_VALUES = 3  # == 2 + nan
MAX_NUM_OF_DECIMALS = 5
# the variable with more than 100 different values should not be StringVariable
DISCRETE_MAX_ALLOWED_VALUES = 100


def make_variable(cls, compute_value, *args):
    if compute_value is not None:
        return cls(*args, compute_value=compute_value)
    else:
        # For compatibility with old pickles: remove the second arg if it's
        # bool `compute_value` (args[3]) can't be bool, so this should be safe
        if len(args) > 2 and isinstance(args[2], bool):
            args = args[:2] + args[3:]
        return cls(*args)


def is_discrete_values(values):
    """
    Return set of uniques if `values` is an iterable of discrete values
    else False if non-discrete, or None if indeterminate.

    Note
    ----
    Assumes consistent type of items of `values`.
    """
    if len(values) == 0:
        return None
    # If the first few values are, or can be converted to, floats,
    # the type is numeric
    try:
        isinstance(next(iter(values)), Number) or \
        [v not in MISSING_VALUES and float(v)
         for _, v in zip(range(min(3, len(values))), values)]
    except ValueError:
        is_numeric = False
        max_values = int(round(len(values)**.7))
    else:
        is_numeric = True
        max_values = DISCRETE_MAX_VALUES

    # If more than max values => not discrete
    unique = set()
    for i in values:
        unique.add(i)
        if (len(unique) > max_values or
                len(unique) > DISCRETE_MAX_ALLOWED_VALUES):
            return False

    # Strip NaN from unique
    unique = {i for i in unique
              if (not i in MISSING_VALUES and
                  not (isinstance(i, Number) and np.isnan(i)))}

    # All NaNs => indeterminate
    if not unique:
        return None

    # Strings with |values| < max_unique
    if not is_numeric:
        return unique

    # Handle numbers
    try:
        unique_float = set(map(float, unique))
    except ValueError:
        # Converting all the values to floats resulted in an error.
        # Since the values have enough unique values, they are probably
        # string values and discrete.
        return unique

    # If only values are {0, 1} or {1, 2} (or a subset of those sets) => discrete
    return (not (unique_float - {0, 1}) or
            not (unique_float - {1, 2})) and unique


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
        if variable.is_primitive():
            self = super().__new__(cls, value)
            self.variable = variable
            self._value = None
        else:
            isunknown = value == variable.Unknown
            self = super().__new__(
                cls, np.nan if isunknown else np.finfo(float).min)
            self.variable = variable
            self._value = value
        return self

    def __init__(self, _, __=Unknown):
        # __new__ does the job, pylint: disable=super-init-not-called
        pass

    def __repr__(self):
        return "Value('%s', %s)" % (self.variable.name,
                                    self.variable.repr_val(self))

    def __str__(self):
        return self.variable.str_val(self)

    def __eq__(self, other):
        if isinstance(self, Real) and isnan(self):
            if isinstance(other, Real):
                return isnan(other)
            else:
                return other in self.variable.unknown_str
        if isinstance(other, str):
            return self.variable.str_val(self) == other
        if isinstance(other, Value):
            return self.value == other.value
        return super().__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if self.variable.is_primitive():
            if isinstance(other, str):
                return super().__lt__(self.variable.to_val(other))
            else:
                return super().__lt__(other)
        else:
            if isinstance(other, str):
                return self.value < other
            else:
                return self.value < other.value

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __contains__(self, other):
        if (self._value is not None
                and isinstance(self._value, str)
                and isinstance(other, str)):
            return other in self._value
        raise TypeError("invalid operation on Value()")

    def __hash__(self):
        if self.variable.is_discrete:
            # It is not possible to hash the id and the domain value to the
            # same number as required by __eq__.
            # hash(1)
            # == hash(Value(DiscreteVariable("var", ["red", "green", "blue"]), 1))
            # == hash("green")
            # User should hash directly ids or domain values instead.
            raise TypeError("unhashable type - cannot hash values of discrete variables!")
        if self._value is None:
            return super().__hash__()
        else:
            return hash(self._value)

    @property
    def value(self):
        if self.variable.is_discrete:
            return Unknown if isnan(self) else self.variable.values[int(self)]
        if self.variable.is_string:
            return self._value
        return float(self)

    def __getnewargs__(self):
        return self.variable, float(self)

    def __getstate__(self):
        return dict(value=getattr(self, '_value', None))

    def __setstate__(self, state):
        # defined in __new__, pylint: disable=attribute-defined-outside-init
        self._value = state.get('value', None)


class VariableMeta(Registry):
    pass


class _predicatedescriptor(property):
    """
    A property that behaves as a class method if accessed via a class
    >>> class A:
    ...     foo = False
    ...     @_predicatedescriptor
    ...     def is_foo(self):
    ...         return self.foo
    ...
    >>> a = A()
    >>> a.is_foo
    False
    >>> A.is_foo(a)
    False
    """
    def __get__(self, instance, objtype=None):
        if instance is None:
            return self.fget
        else:
            return super().__get__(instance, objtype)


class Variable(Reprable, metaclass=VariableMeta):
    """
    The base class for variable descriptors contains the variable's
    name and some basic properties.

    .. attribute:: name

        The name of the variable.

    .. attribute:: unknown_str

        A set of values that represent unknowns in conversion from textual
        formats. Default is `{"?", ".", "", "NA", "~", None}`.

    .. attribute:: compute_value

        A function for computing the variable's value when converting from
        another domain which does not contain this variable. The function will
        be called with a data set (`Orange.data.Table`) and has to return
        an array of computed values for all its instances. The base class
        defines a static method `compute_value`, which returns `Unknown`.
        Non-primitive variables must redefine it to return `None`.

    .. attribute:: sparse

        A flag about sparsity of the variable. When set, the variable suggests
        it should be stored in a sparse matrix.

    .. attribute:: source_variable

        An optional descriptor of the source variable - if any - from which
        this variable is derived and computed via :obj:`compute_value`.

    .. attribute:: attributes

        A dictionary with user-defined attributes of the variable
    """
    Unknown = ValueUnknown

    def __init__(self, name="", compute_value=None, *, sparse=False):
        """
        Construct a variable descriptor.
        """
        if not name:
            warnings.warn("Variable must have a name", OrangeDeprecationWarning,
                          stacklevel=3)
        self._name = name
        self._compute_value = compute_value
        self.unknown_str = MISSING_VALUES
        self.source_variable = None
        self.sparse = sparse
        self.attributes = {}

    @property
    def name(self):
        return self._name

    def make_proxy(self):
        """
        Copy the variable and set the master to `self.master` or to `self`.

        :return: copy of self
        :rtype: Variable
        """
        var = self.__class__(self.name)
        var.__dict__.update(self.__dict__)
        var.attributes = dict(self.attributes)
        return var

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        var1 = self._get_identical_source(self)
        var2 = self._get_identical_source(other)
        # pylint: disable=protected-access
        return (
            self.name == other.name
            and var1.name == var2.name
            and var1._compute_value == var2._compute_value
        )

    def __hash__(self):
        var = self._get_identical_source(self)
        return hash((self.name, var.name, type(self), var._compute_value))

    @staticmethod
    def _get_identical_source(var):
        # pylint: disable=protected-access,import-outside-toplevel
        from Orange.preprocess.transformation import Identity
        while isinstance(var._compute_value, Identity):
            var = var._compute_value.variable
        return var

    @classmethod
    def make(cls, name, *args, **kwargs):
        """
        Return an existing continuous variable with the given name, or
        construct and return a new one.
        """
        return cls(name, *args, **kwargs)

    @classmethod
    def _clear_cache(cls):
        warnings.warn(
            "_clear_cache is no longer needed and thus deprecated")

    @staticmethod
    def _clear_all_caches():
        warnings.warn(
            "_clear_all_caches is no longer needed and thus deprecated")

    @classmethod
    def is_primitive(cls, var=None):
        """
        `True` if the variable's values are stored as floats.
        Non-primitive variables can appear in the data only as meta attributes.
        """
        to_check = cls if var is None else type(var)
        return issubclass(to_check, (DiscreteVariable, ContinuousVariable))

    @_predicatedescriptor
    def is_discrete(self):
        return isinstance(self, DiscreteVariable)

    @_predicatedescriptor
    def is_continuous(self):
        return isinstance(self, ContinuousVariable)

    @_predicatedescriptor
    def is_string(self):
        return isinstance(self, StringVariable)

    @_predicatedescriptor
    def is_time(self):
        return isinstance(self, TimeVariable)

    @staticmethod
    def repr_val(val):
        """
        Return a textual representation of variable's value `val`. Argument
        `val` must be a float (for primitive variables) or an arbitrary
        Python object (for non-primitives).

        Derived classes must overload the function.
        """
        raise RuntimeError("variable descriptors must overload repr_val()")

    str_val = repr_val

    def to_val(self, s):
        """
        Convert the given argument to a value of the variable. The
        argument can be a string, a number or `None`. For primitive variables,
        the base class provides a method that returns
        :obj:`~Orange.data.Unknown` if `s` is found in
        :obj:`~Orange.data.Variable.unknown_str`, and raises an exception
        otherwise. For non-primitive variables it returns the argument itself.

        Derived classes of primitive variables must overload the function.

        :param s: value, represented as a number, string or `None`
        :type s: str, float or None
        :rtype: float or object
        """
        if not self.is_primitive():
            return s
        if s in self.unknown_str:
            return Unknown
        raise RuntimeError(
            "primitive variable descriptors must overload to_val()")

    def val_from_str_add(self, s):
        """
        Convert the given string to a value of the variable. The method
        is similar to :obj:`to_val` except that it only accepts strings and
        that it adds new values to the variable's domain where applicable.

        The base class method calls `to_val`.

        :param s: symbolic representation of the value
        :type s: str
        :rtype: float or object
        """
        return self.to_val(s)

    def __str__(self):
        return self.name

    @property
    def compute_value(self):
        return self._compute_value

    def __reduce__(self):
        if not self.name:
            raise PickleError("Variables without names cannot be pickled")

        # Use make to unpickle variables.
        return make_variable, (self.__class__, self._compute_value, self.name), self.__dict__

    _CopyComputeValue = object()

    def copy(self, compute_value=_CopyComputeValue, *, name=None, **kwargs):
        if compute_value is self._CopyComputeValue:
            compute_value = self.compute_value
        var = type(self)(name=name or self.name,
                         compute_value=compute_value,
                         sparse=self.sparse, **kwargs)
        var.attributes = dict(self.attributes)
        return var

    def renamed(self, new_name):
        # prevent cyclic import, pylint: disable=import-outside-toplevel
        from Orange.preprocess.transformation import Identity
        return self.copy(name=new_name, compute_value=Identity(variable=self))

del _predicatedescriptor


class ContinuousVariable(Variable):
    """
    Descriptor for continuous variables.

    .. attribute:: number_of_decimals

        The number of decimals when the value is printed out (default: 3).

    .. attribute:: adjust_decimals

        A flag regulating whether the `number_of_decimals` is being adjusted
        by :obj:`to_val`.

    The value of `number_of_decimals` is set to 3 and `adjust_decimals`
    is set to 2. When :obj:`val_from_str_add` is called for the first
    time with a string as an argument, `number_of_decimals` is set to the
    number of decimals in the string and `adjust_decimals` is set to 1.
    In the subsequent calls of `to_val`, the nubmer of decimals is
    increased if the string argument has a larger number of decimals.

    If the `number_of_decimals` is set manually, `adjust_decimals` is
    set to 0 to prevent changes by `to_val`.
    """

    TYPE_HEADERS = ('continuous', 'c', 'numeric', 'n')

    def __init__(self, name="", number_of_decimals=None, compute_value=None, *, sparse=False):
        """
        Construct a new continuous variable. The number of decimals is set to
        three, but adjusted at the first call of :obj:`to_val`.
        """
        super().__init__(name, compute_value, sparse=sparse)
        self._max_round_diff = 0
        self.number_of_decimals = number_of_decimals

    @property
    def number_of_decimals(self):
        return self._number_of_decimals

    @property
    def format_str(self):
        return self._format_str

    @format_str.setter
    def format_str(self, value):
        self._format_str = value

    # noinspection PyAttributeOutsideInit
    @number_of_decimals.setter
    def number_of_decimals(self, x):
        if x is None:
            self._number_of_decimals = 3
            self.adjust_decimals = 2
            self._format_str = "%g"
            return

        self._number_of_decimals = x
        self._max_round_diff = 10 ** (-x - 6)
        self.adjust_decimals = 0
        if self._number_of_decimals <= MAX_NUM_OF_DECIMALS:
            self._format_str = "%.{}f".format(self.number_of_decimals)
        else:
            self._format_str = "%g"

    def to_val(self, s):
        """
        Convert a value, given as an instance of an arbitrary type, to a float.
        """
        if s in self.unknown_str:
            return Unknown
        return float(s)

    def val_from_str_add(self, s):
        """
        Convert a value from a string and adjust the number of decimals if
        `adjust_decimals` is non-zero.
        """
        return _variable.val_from_str_add_cont(self, s)

    def repr_val(self, val):
        """
        Return the value as a string with the prescribed number of decimals.
        """
        # Table value can't be inf, but repr_val can be used to print any float
        if not np.isfinite(val):
            return "?"
        if self.format_str != "%g" \
                and abs(round(val, self._number_of_decimals) - val) \
                > self._max_round_diff:
            return f"{val:.{self._number_of_decimals + 2}f}"
        return self._format_str % val

    str_val = repr_val

    def copy(self, compute_value=Variable._CopyComputeValue,
             *, name=None, **kwargs):
        # pylint understand not that `var` is `DiscreteVariable`:
        # pylint: disable=protected-access
        number_of_decimals = kwargs.pop("number_of_decimals", None)
        var = super().copy(compute_value=compute_value, name=name, **kwargs)
        if number_of_decimals is not None:
            var.number_of_decimals = number_of_decimals
        else:
            var._number_of_decimals = self._number_of_decimals
            var._max_round_diff = self._max_round_diff
            var.adjust_decimals = self.adjust_decimals
            var.format_str = self._format_str
        return var


TupleList = tuple # backward compatibility (for pickled table)


class DiscreteVariable(Variable):
    """
    Descriptor for symbolic, discrete variables. Values of discrete variables
    are stored as floats; the numbers corresponds to indices in the list of
    values.

    .. attribute:: values

        A list of variable's values.
    """

    TYPE_HEADERS = ('discrete', 'd', 'categorical')

    presorted_values = []

    def __init__(
            self, name="", values=(), compute_value=None, *, sparse=False
    ):
        """ Construct a discrete variable descriptor with the given values. """
        values = tuple(values)  # some people (including me) pass a generator
        if not all(isinstance(value, str) for value in values):
            raise TypeError("values of DiscreteVariables must be strings")

        super().__init__(name, compute_value, sparse=sparse)
        self._values = values
        self._value_index = {value: i for i, value in enumerate(values)}

    @property
    def values(self):
        return self._values

    def get_mapping_from(self, other):
        return np.array(
            [self._value_index.get(value, np.nan) for value in other.values],
            dtype=float)

    def get_mapper_from(self, other):
        mapping = self.get_mapping_from(other)
        if not mapping.size:
            # Nans in data are temporarily replaced with 0, mapped and changed
            # back to nans. This would fail is mapping[0] is out of range.
            mapping = np.array([np.nan])

        def mapper(value, col_idx=None):

            # In-place mapping
            if col_idx is not None:
                if sp.issparse(value) and mapping[0] != 0:
                    raise ValueError(
                        "In-place mapping of sparse matrices must map 0 to 0")

                # CSR requires mapping of non-contiguous area
                if sp.isspmatrix_csr(value):
                    col = value.indices == col_idx
                    nans = np.isnan(value.data) * col
                    value.data[nans] = 0
                    value.data[col] = mapping[value.data[col].astype(int)]
                    value.data[nans] = np.nan
                    return None

                # Dense and CSC map a contiguous area
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    col = value[:, col_idx]
                elif sp.isspmatrix_csc(value):
                    col = value.data[value.indptr[col_idx]
                                     :value.indptr[col_idx + 1]]
                else:
                    raise ValueError(
                        "In-place column mapping requires a 2d array or"
                        "a csc or csr matrix.")

                nans = np.isnan(col)
                col[nans] = 0
                col[:] = mapping[col.astype(int)]
                col[nans] = np.nan
                return None

            # Mapping into a copy
            if isinstance(value, (int, float)):
                return value if np.isnan(value) else mapping[int(value)]
            if isinstance(value, str):
                return mapping[other.values.index(value)]
            if isinstance(value, np.ndarray):
                if not (value.ndim == 1
                        or value.ndim != 2 and min(value.shape) != 1):
                    raise ValueError(
                        f"Column mapping can't map {value.ndim}-d objects")

                if value.dtype == object:
                    value = value.astype(float)  # this happens with metas
                try:
                    nans = np.isnan(value)
                except TypeError:  # suppose it's already an integer type
                    return mapping[value]
                value = value.astype(int)
                value[nans] = 0
                value = mapping[value]
                value[nans] = np.nan
                return value
            if sp.issparse(value):
                if min(value.shape) != 1:
                    raise ValueError("Column mapping can't map "
                                     f"{value.ndim}-dimensional objects")
                if mapping[0] != 0 and not np.isnan(mapping[0]):
                    return mapper(np.array(value.todense()).flatten())
                value = value.copy()
                value.data = mapper(value.data)
                return value
            if isinstance(value, Iterable):
                return type(value)(val if np.isnan(val) else mapping[int(val)]
                                   for val in value)
            raise ValueError(
                f"invalid type for value(s): {type(value).__name__}")

        return mapper

    def to_val(self, s):
        """
        Convert the given argument to a value of the variable (`float`).
        If the argument is numeric, its value is returned without checking
        whether it is integer and within bounds. `Unknown` is returned if the
        argument is one of the representations for unknown values. Otherwise,
        the argument must be a string and the method returns its index in
        :obj:`values`.

        :param s: values, represented as a number, string or `None`
        :rtype: float
        """
        if s is None:
            return ValueUnknown

        if isinstance(s, Integral):
            return s
        if isinstance(s, Real):
            return s if isnan(s) else floor(s + 0.25)
        if s in self.unknown_str:
            return ValueUnknown
        if not isinstance(s, str):
            raise TypeError('Cannot convert {} to value of "{}"'.format(
                type(s).__name__, self.name))
        if s not in self._value_index:
            raise ValueError(f"Value {s} does not exist")
        return self._value_index[s]

    def add_value(self, s):
        """ Add a value `s` to the list of values.
        """
        if not isinstance(s, str):
            raise TypeError("values of DiscreteVariables must be strings")
        if s in self._value_index:
            return
        self._value_index[s] = len(self.values)
        self._values += (s, )

    def val_from_str_add(self, s):
        """
        Similar to :obj:`to_val`, except that it accepts only strings and that
        it adds the value to the list if it does not exist yet.

        :param s: symbolic representation of the value
        :type s: str
        :rtype: float
        """
        s = str(s) if s is not None else s
        if s in self.unknown_str:
            return ValueUnknown
        val = self._value_index.get(s)
        if val is None:
            self.add_value(s)
            val = len(self.values) - 1
        return val

    def repr_val(self, val):
        """
        Return a textual representation of the value (`self.values[int(val)]`)
        or "?" if the value is unknown.

        :param val: value
        :type val: float (should be whole number)
        :rtype: str
        """
        if isnan(val):
            return "?"
        return '{}'.format(self.values[int(val)])

    str_val = repr_val

    def __reduce__(self):
        if not self.name:
            raise PickleError("Variables without names cannot be pickled")
        __dict__ = dict(self.__dict__)
        __dict__.pop("_values")
        return (
            make_variable,
            (self.__class__, self._compute_value, self.name, self.values),
            __dict__
        )

    def copy(self, compute_value=Variable._CopyComputeValue,
             *, name=None, values=None, **_):
        # pylint: disable=arguments-differ
        if values is not None and len(values) != len(self.values):
            raise ValueError(
                "number of values must match the number of original values")
        return super().copy(compute_value=compute_value, name=name,
                            values=values or self.values)


class StringVariable(Variable):
    """
    Descriptor for string variables. String variables can only appear as
    meta attributes.
    """
    Unknown = ""
    TYPE_HEADERS = ('string', 's', 'text')

    def to_val(self, s):
        """
        Return the value as a string. If it is already a string, the same
        object is returned.
        """
        if s is None:
            return ""
        if isinstance(s, str):
            return s
        return str(s)

    val_from_str_add = to_val

    @staticmethod
    def str_val(val):
        """Return a string representation of the value."""
        if isinstance(val, str) and val == "":
            return "?"
        if isinstance(val, Value):
            if not val.value:
                return "?"
            val = val.value
        return str(val)

    def repr_val(self, val):
        """Return a string representation of the value."""
        return '"{}"'.format(self.str_val(val))


class TimeVariable(ContinuousVariable):
    """
    TimeVariable is a continuous variable with Unix epoch
    (1970-01-01 00:00:00+0000) as the origin (0.0). Later dates are positive
    real numbers (equivalent to Unix timestamp, with microseconds in the
    fraction part), and the dates before it map to the negative real numbers.

    Unfortunately due to limitation of Python datetime, only dates
    with year >= 1 (A.D.) are supported.

    If time is specified without a date, Unix epoch is assumed.

    If time is specified wihout an UTC offset, localtime is assumed.
    """
    _all_vars = {}
    TYPE_HEADERS = ('time', 't')
    UNIX_EPOCH = datetime(1970, 1, 1)
    _ISO_FORMATS = (
        # have_date, have_time, format_str
        # in order of decreased probability
        (1, 1, '%Y-%m-%d %H:%M:%S%z'),
        (1, 1, '%Y-%m-%d %H:%M:%S'),
        (1, 1, '%Y-%m-%d %H:%M'),
        (1, 1, '%Y-%m-%dT%H:%M:%S%z'),
        (1, 1, '%Y-%m-%dT%H:%M:%S'),

        (1, 0, '%Y-%m-%d'),

        (1, 1, '%Y-%m-%d %H:%M:%S.%f'),
        (1, 1, '%Y-%m-%dT%H:%M:%S.%f'),
        (1, 1, '%Y-%m-%d %H:%M:%S.%f%z'),
        (1, 1, '%Y-%m-%dT%H:%M:%S.%f%z'),

        (1, 1, '%Y%m%dT%H%M%S%z'),
        (1, 1, '%Y%m%d%H%M%S%z'),

        (0, 1, '%H:%M:%S.%f'),
        (0, 1, '%H:%M:%S'),
        (0, 1, '%H:%M'),

        # These parse as continuous features (plain numbers)
        (1, 1, '%Y%m%dT%H%M%S'),
        (1, 1, '%Y%m%d%H%M%S'),
        (1, 0, '%Y%m%d'),
        (1, 0, '%Y%j'),
        (1, 0, '%Y'),
        (0, 1, '%H%M%S.%f'),

        # BUG: In Python as in C, %j doesn't necessitate 0-padding,
        # so these two lines must be in this order
        (1, 0, '%Y-%m'),
        (1, 0, '%Y-%j'),
    )
    # Order in which `_ISO_FORMATS` are tried. Must never change order of
    # last 2 items. Only modified via assignment in `parse`.
    __ISO_FORMATS_PROBE_SEQ = list(range(len(_ISO_FORMATS)))
    # The regex that matches all above formats
    REGEX = (r'^('
             r'\d{1,4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?([+-]\d{4})?)?)?|'
             r'\d{1,4}\d{2}\d{2}(T?\d{2}\d{2}\d{2}([+-]\d{4})?)?|'
             r'\d{2}:\d{2}(:\d{2}(\.\d+)?)?|'
             r'\d{2}\d{2}\d{2}\.\d+|'
             r'\d{1,4}(-?\d{2,3})?'
             r')$')

    class InvalidDateTimeFormatError(ValueError):
        def __init__(self, date_string):
            super().__init__(
                "Invalid datetime format '{}'. "
                "Only ISO 8601 supported.".format(date_string))

    _matches_iso_format = re.compile(REGEX).match

    # UTC offset and associated timezone. If parsed datetime values provide an
    # offset, it is used for display. If not all values have the same offset,
    # +0000 (=UTC) timezone is used and utc_offset is set to False.
    _utc_offset = None  # deprecated - remove in 3.32
    _timezone = None

    def __init__(self, *args, have_date=0, have_time=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.have_date = have_date
        self.have_time = have_time

    # deprecated - remove in 3.32 - from here
    @property
    def utc_offset(self):
        warnings.warn(
            "utc_offset is deprecated and will be removed in Orange 3.32",
            OrangeDeprecationWarning
        )
        return self._utc_offset

    @utc_offset.setter
    def utc_offset(self, val):
        warnings.warn(
            "utc_offset is deprecated and will be removed in Orange 3.32ß",
            OrangeDeprecationWarning
        )
        self._utc_offset = val
    # remove to here

    @property
    def timezone(self):
        if self._timezone is None or self._timezone == "different timezones":
            return timezone.utc
        else:
            return self._timezone

    @timezone.setter
    def timezone(self, tz):
        """
        Set timezone value:
        - if self._timezone is None set it to  new timezone
        - if current timezone is different that new indicate that TimeVariable
          have two date-times with different timezones
        - if timezones are same keep it
        """
        if self._timezone is None:
            self._timezone = tz
        elif tz != self.timezone:
            self._timezone = "different timezones"

    def copy(self, compute_value=Variable._CopyComputeValue, *, name=None, **_):
        return super().copy(compute_value=compute_value, name=name,
                            have_date=self.have_date, have_time=self.have_time)

    @staticmethod
    def _tzre_sub(s, _subtz=re.compile(r'([+-])(\d\d):(\d\d)$').sub):
        # Replace +ZZ:ZZ with ISO-compatible +ZZZZ, or strip +0000
        return s[:-6] if s.endswith(('+00:00', '-00:00')) else _subtz(r'\1\2\3', s)

    def repr_val(self, val):
        if isnan(val):
            return '?'
        if not self.have_date and not self.have_time:
            # The time is relative, unitless. The value is absolute.
            return str(val.value) if isinstance(val, Value) else str(val)

        # If you know how to simplify this, be my guest
        seconds = int(val)
        microseconds = int(round((val - seconds) * 1e6))
        if val < 0:
            if microseconds:
                seconds, microseconds = seconds - 1, int(1e6) + microseconds
            date = datetime.fromtimestamp(0, tz=self.timezone) + timedelta(seconds=seconds)
        else:
            date = datetime.fromtimestamp(seconds, tz=self.timezone)
        date = str(date.replace(microsecond=microseconds))

        if self.have_date and not self.have_time:
            date = date.split()[0]
        elif not self.have_date and self.have_time:
            date = date.split()[1]
        date = self._tzre_sub(date)
        return date

    str_val = repr_val

    def parse(self, datestr):
        """
        Return `datestr`, a datetime provided in one of ISO 8601 formats,
        parsed as a real number. Value 0 marks the Unix epoch, positive values
        are the dates after it, negative before.

        If date is unspecified, epoch date is assumed.

        If time is unspecified, 00:00:00.0 is assumed.

        If timezone is unspecified, local time is assumed.
        """
        if datestr in MISSING_VALUES:
            return Unknown
        datestr = datestr.strip().rstrip('Z')
        datestr = self._tzre_sub(datestr)

        if not self._matches_iso_format(datestr):
            try:
                # If it is a number, assume it is a unix timestamp
                value = float(datestr)
                self.have_date = self.have_time = 1
                return value
            except ValueError:
                raise self.InvalidDateTimeFormatError(datestr)

        try_order = self.__ISO_FORMATS_PROBE_SEQ
        for i, (have_date, have_time, fmt) in enumerate(
                map(self._ISO_FORMATS.__getitem__, try_order)):
            try:
                dt = datetime.strptime(datestr, fmt)
            except ValueError:
                continue
            else:
                # Pop this most-recently-used format index to front,
                # excluding last 2
                if 0 < i < len(try_order) - 2:
                    try_order = try_order.copy()
                    try_order[i], try_order[0] = try_order[0], try_order[i]
                    TimeVariable.__ISO_FORMATS_PROBE_SEQ = try_order
                self.have_date |= have_date
                self.have_time |= have_time
                if not have_date:
                    dt = dt.replace(self.UNIX_EPOCH.year,
                                    self.UNIX_EPOCH.month,
                                    self.UNIX_EPOCH.day)
                break
        else:
            raise self.InvalidDateTimeFormatError(datestr)

        offset = dt.utcoffset()
        self.timezone = timezone(offset) if offset is not None else None
        # deprecated - remove in 3.32 - from here
        if self._utc_offset is not False:
            if offset and self._utc_offset is None:
                self._utc_offset = offset
            elif self._utc_offset != offset:
                self._utc_offset = False
        # remove to here

        # Convert time to UTC timezone. In dates without timezone,
        # localtime is assumed. See also:
        # https://docs.python.org/3.4/library/datetime.html#datetime.datetime.timestamp
        if dt.tzinfo:
            dt -= dt.utcoffset()
        dt = dt.replace(tzinfo=timezone.utc)

        # Unix epoch is the origin, older dates are negative
        try:
            return dt.timestamp()
        except OverflowError:
            return -(self.UNIX_EPOCH - dt).total_seconds()

    def parse_exact_iso(self, datestr):
        """
        This function is a meta function to `parse` function. It checks
        whether the date is of the iso format - it does not accept float-like
        date.
        """
        if not self._matches_iso_format(datestr):
            raise self.InvalidDateTimeFormatError(datestr)
        return self.parse(datestr)

    def to_val(self, s):
        """
        Convert a value, given as an instance of an arbitrary type, to a float.
        """
        if isinstance(s, str):
            return self.parse(s)
        else:
            return super().to_val(s)
