import re
from numbers import Number, Real, Integral
from math import isnan, floor, sqrt
import numpy as np
import pandas as pd
from pickle import PickleError
import copy
import dateutil
import pytz

import collections
from datetime import datetime, timedelta, timezone

from . import _variable

from Orange.util import Registry, color_to_hex, hex_to_color


# For storing unknowns
Unknown = ValueUnknown = float("nan")


def make_variable(cls, compute_value, *args):
    if compute_value is not None:
        return cls(*args, compute_value=compute_value)
    return cls.make(*args)


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
        if self._value is None:
            return super().__hash__()
        else:
            return hash((super().__hash__(), self._value))

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
        self._value = state.get('value', None)


class VariableMeta(Registry):
    def __new__(cls, name, bases, attrs):
        obj = super().__new__(cls, name, bases, attrs)
        if not hasattr(obj, '_all_vars') or obj._all_vars is Variable._all_vars:
            obj._all_vars = {}
        return obj


class Variable(str, metaclass=VariableMeta):
    """
    The base class for variable descriptors contains the variable's
    name and some basic properties. This extends str so it plays nicely with
    pandas' column values.

    .. attribute:: name

        The name of the variable.

    .. attribute:: unknown_str

        A set of values that represent unknowns in conversion from textual
        formats. Default is `{"?", ".", "", "NA", "~", None}`.

    .. attribute:: compute_value

        A function for computing the variable's value when converting from
        another domain which does not contain this variable. The base class
        defines a static method `compute_value`, which returns `Unknown`.
        Non-primitive variables must redefine it to return `None`.

    .. attribute:: source_variable

        An optional descriptor of the source variable - if any - from which
        this variable is derived and computed via :obj:`compute_value`.

    .. attribute:: attributes

        A dictionary with user-defined attributes of the variable

    .. attribute:: master

        The variable that this variable is a copy of. If a copy is made from a
        copy, the copy has a reference to the original master. If the variable
        is not a copy, it is its own master.
    """
    Unknown = ValueUnknown
    MISSING_VALUES = {np.nan, "?", "nan", ".", "", "NA", "~", None}

    def __new__(cls, name="", *args, **kwargs):
        # compatibility with str
        return super().__new__(cls, name)

    def __init__(self, name="", compute_value=None):
        """
        Construct a variable descriptor.
        """
        super().__init__()
        self.name = name
        self._compute_value = compute_value
        self.unknown_str = Variable.MISSING_VALUES
        self.source_variable = None
        self.attributes = {}
        self.master = self
        if name and compute_value is None:
            if isinstance(self._all_vars, collections.defaultdict):
                self._all_vars[name].append(self)
            else:
                self._all_vars[name] = self
        self._colors = None

    def make_proxy(self):
        """
        Copy the variable and set the master to `self.master` or to `self`.

        :return: copy of self
        :rtype: Variable
        """
        var = self.__class__()
        var.__dict__.update(self.__dict__)
        var.master = self.master
        return var

    def __eq__(self, other):
        """
        If comparing two variables, compare masters if at least one master
        is set (otherwise compare names).  When comparing strings, compare names,
        otherwise, they are not equal.
        """
        if isinstance(other, Variable):
            return self.master is other.master and self.master is not None
        else:
            return self.name == other

    def __ne__(self, other):
        """Variable extends str, so we have to set this to use our implementation of __eq__."""
        return not self.__eq__(other)

    def __hash__(self):
        return super().__hash__()

    @classmethod
    def make(cls, name):
        """
        Return an existing continuous variable with the given name, or
        construct and return a new one.
        """
        if not name:
            raise ValueError("Variables without names cannot be stored or made")
        return cls._all_vars.get(name) or cls(name)

    @classmethod
    def _clear_cache(cls):
        """
        Clear the list of variables for reuse by :obj:`make`.
        """
        cls._all_vars.clear()

    @staticmethod
    def _clear_all_caches():
        """
        Clears list of stored variables for all subclasses
        """
        for cls in Variable.registry.values():
            cls._clear_cache()

    @classmethod
    def is_primitive(cls):
        """
        `True` if the variable's values are stored as floats.
        Non-primitive variables can appear in the data only as meta attributes.
        """
        return issubclass(cls, (DiscreteVariable, ContinuousVariable))

    @property
    def is_discrete(self):
        return isinstance(self, DiscreteVariable)

    @property
    def is_continuous(self):
        return isinstance(self, ContinuousVariable)

    @property
    def is_string(self):
        return isinstance(self, StringVariable)

    def repr_val(self, val):
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

    def __repr__(self):
        """
        Return a representation of the variable, like,
        `'DiscreteVariable("gender")'`. Derived classes may overload this
        method to provide a more informative representation.
        """
        return "{}('{}')".format(self.__class__.__name__, self.name)

    @property
    def compute_value(self):
        return self._compute_value

    def __reduce__(self):
        if not self.name:
            raise PickleError("Variables without names cannot be pickled")

        return make_variable, (self.__class__, self._compute_value, self.name), self.__dict__

    def copy(self, compute_value):
        var = Variable(self.name, compute_value)
        var.attributes = dict(self.attributes)
        return var


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

    TYPE_HEADERS = ('continuous', 'c')

    def __init__(self, name="", number_of_decimals=None, compute_value=None):
        """
        Construct a new continuous variable. The number of decimals is set to
        three, but adjusted at the first call of :obj:`to_val`.
        """
        super().__init__(name, compute_value)
        if number_of_decimals is None:
            self.number_of_decimals = 3
            self.adjust_decimals = 2
        else:
            self.number_of_decimals = number_of_decimals

    @property
    def number_of_decimals(self):
        return self._number_of_decimals

    @property
    def colors(self):
        if self._colors is None:
            if "colors" in self.attributes:
                col1, col2, black = self.attributes["colors"]
                self._colors = (hex_to_color(col1), hex_to_color(col2), black)
            else:
                self._colors = ((0, 0, 255), (255, 255, 0), False)
        return self._colors

    @colors.setter
    def colors(self, value):
        col1, col2, black = self._colors = value
        self.attributes["colors"] = \
            [color_to_hex(col1), color_to_hex(col2), black]

    # noinspection PyAttributeOutsideInit
    @number_of_decimals.setter
    def number_of_decimals(self, x):
        self._number_of_decimals = x
        self.adjust_decimals = 0
        self._out_format = "%.{}f".format(self.number_of_decimals)

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
        if isnan(val):
            return "?"
        return self._out_format % val

    str_val = repr_val

    def copy(self, compute_value=None):
        var = ContinuousVariable(self.name, self.number_of_decimals, compute_value)
        var.attributes = dict(self.attributes)
        return var


class DiscreteVariable(Variable):
    """
    Descriptor for symbolic, discrete variables. Values of discrete variables
    are stored as floats; the numbers corresponds to indices in the list of
    values.

    .. attribute:: values

        A list of variable's values.

    .. attribute:: ordered

        Some algorithms (and, in particular, visualizations) may
        sometime reorder the values of the variable, e.g. alphabetically.
        This flag hints that the given order of values is "natural"
        (e.g. "small", "middle", "large") and should not be changed.

    .. attribute:: base_value

        The index of the base value, or -1 if there is none. The base value is
        used in some methods like, for instance, when creating dummy variables
        for regression.
    """

    TYPE_HEADERS = ('discrete', 'd')

    _all_vars = collections.defaultdict(list)

    def __init__(self, name="", values=(), ordered=False, base_value=-1, compute_value=None):
        """ Construct a discrete variable descriptor with the given values. """
        super().__init__(name, compute_value)
        self.ordered = ordered
        self.values = list(values)
        self.base_value = base_value

    @property
    def colors(self):
        if self._colors is None:
            if "colors" in self.attributes:
                self._colors = np.array(
                    [hex_to_color(col) for col in self.attributes["colors"]],
                    dtype=np.uint8)
            else:
                from Orange.widgets.utils.colorpalette import \
                    ColorPaletteGenerator
                self._colors = ColorPaletteGenerator.palette(self)
            self._colors.flags.writeable = False
        return self._colors

    @colors.setter
    def colors(self, value):
        self._colors = value
        self._colors.flags.writeable = False
        self.attributes["colors"] = [color_to_hex(col) for col in value]

    def set_color(self, i, color):
        self.colors = self.colors
        self._colors.flags.writeable = True
        self._colors[i, :] = color
        self._colors.flags.writeable = False
        self.attributes["colors"][i] = color_to_hex(color)

    def __repr__(self):
        """
        Give a string representation of the variable, for instance,
        `"DiscreteVariable('Gender', values=['male', 'female'])"`.
        """
        args = "values=[{}]".format(
            ", ".join([repr(x) for x in self.values[:5]] +
                      ["..."] * (len(self.values) > 5)))
        if self.ordered:
            args += ", ordered=True"
        if self.base_value >= 0:
            args += ", base_value={}".format(self.base_value)
        return "{}('{}', {})".format(self.__class__.__name__, self.name, args)

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

        if isinstance(s, (Integral, Real)):
            if isnan(s):
                return s
            elif s in self.values:
                return self.values.index(s)
            else:
                # find the nearest instance
                np.argmin([np.abs(v - s) if isinstance(v, Number) else np.inf for v in self.values])
        if s in self.unknown_str:
            return ValueUnknown
        if not isinstance(s, str):
            raise TypeError('Cannot convert {} to value of "{}"'.format(
                type(s).__name__, self.name))
        return self.values.index(s)

    def add_value(self, s):
        """ Add a value `s` to the list of values.
        """
        self.values.append(s)

    def val_from_str_add(self, s):
        """
        Similar to :obj:`to_val`, except that it accepts only strings and that
        it adds the value to the list if it does not exist yet.

        :param s: symbolic representation of the value
        :type s: str
        :rtype: float
        """
        s = str(s) if s is not None else s
        try:
            return ValueUnknown if s in self.unknown_str \
                else self.values.index(s)
        except ValueError:
            self.add_value(s)
            return len(self.values) - 1

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
        return make_variable, (self.__class__, self._compute_value, self.name,
                               self.values, self.ordered, self.base_value), \
            self.__dict__

    @classmethod
    def make(cls, name, values=(), ordered=False, base_value=-1):
        """
        Return a variable with the given name and other properties. The method
        first looks for a compatible existing variable: the existing
        variable must have the same name and both variables must have either
        ordered or unordered values. If values are ordered, the order must be
        compatible: all common values must have the same order. If values are
        unordered, the existing variable must have at least one common value
        with the new one, except when any of the two lists of values is empty.

        If a compatible variable is find, it is returned, with missing values
        appended to the end of the list. If there is no explicit order, the
        values are ordered using sorted. Otherwise, it
        constructs and returns a new variable descriptor.

        :param name: the name of the variable
        :type name: str
        :param values: symbolic values for the variable
        :type values: list
        :param ordered: tells whether the order of values is fixed
        :type ordered: bool
        :param base_value: the index of the base value, or -1 if there is none
        :type base_value: int
        :returns: an existing compatible variable or `None`
        """
        if not name:
            raise ValueError("Variables without names cannot be stored or made")
        var = cls._find_compatible(
            name, values, ordered, base_value)
        if var:
            return var
        if not ordered:
            base_value_rep = base_value != -1 and values[base_value]
            try:
                values = sorted(values, key=float)
            except ValueError:
                values = sorted(values)
            if base_value != -1:
                base_value = values.index(base_value_rep)
        return cls(name, values, ordered, base_value)

    @classmethod
    def _find_compatible(cls, name, values=(), ordered=False, base_value=-1):
        """
        Return a compatible existing value, or `None` if there is None.
        See :obj:`make` for details; this function differs by returning `None`
        instead of constructing a new descriptor. (Method :obj:`make` calls
        this function.)

        :param name: the name of the variable
        :type name: str
        :param values: symbolic values for the variable
        :type values: list
        :param ordered: tells whether the order of values is fixed
        :type ordered: bool
        :param base_value: the index of the base value, or -1 if there is none
        :type base_value: int
        :returns: an existing compatible variable or `None`
        """
        base_rep = base_value != -1 and values[base_value]
        existing = cls._all_vars.get(name)
        if existing is None:
            return None
        if not ordered:
            try:
                values = sorted(values, key=float)
            except ValueError:
                values = sorted(values)
        for var in existing:
            if (var.ordered != ordered or
                    var.base_value != -1
                    and var.values[var.base_value] != base_rep):
                continue
            if not values:
                break  # we have the variable - any existing values are OK
            if not set(var.values) & set(values):
                continue  # empty intersection of values; not compatible
            if ordered:
                i = 0
                for val in var.values:
                    if values[i] == val:
                        i += 1
                        if i == len(values):
                            break  # we have all the values
                else:  # we have some remaining values: check them, add them
                    if set(values[i:]) & set(var.values):
                        continue  # next var in existing
                    for val in values[i:]:
                        var.add_value(val)
                break  # we have the variable
            else:  # not ordered
                vv = set(var.values)
                for val in values:
                    if val not in vv:
                        var.add_value(val)
                break  # we have the variable
        else:
            return None
        if base_value != -1 and var.base_value == -1:
            var.base_value = var.values.index(base_rep)
        return var

    def copy(self, compute_value=None):
        var = DiscreteVariable(self.name, self.values, self.ordered,
                               self.base_value, compute_value)
        var.attributes = dict(self.attributes)
        return var

    @classmethod
    def generate_unique_values(cls, column):
        """
        Generate a sorted set of unique values from a pandas column,
        taking into account values we consider missing (Variable.MISSING_VALUES).
        """
        # comparing np.nan doesn't always work, use the appropriate mechanism
        raw = column[~column.isnull()].unique()
        return sorted([v for v in raw if v not in Variable.MISSING_VALUES])


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
        if s is None or (isinstance(s, Number) and np.isnan(s)):
            return ""
        if isinstance(s, str):
            return s
        return str(s)

    val_from_str_add = to_val

    @staticmethod
    def str_val(val):
        """Return a string representation of the value."""
        if val is "":
            return "?"
        if isinstance(val, Value):
            if val.value is "":
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
    TYPE_HEADERS = ('time', 't')

    # The regex that matches most ISO formats
    REGEX = (r'^('
             '\d{1,4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?([+-]\d{4})?)?)?|'
             '\d{1,4}\d{2}\d{2}(T?\d{2}\d{2}\d{2}([+-]\d{4})?)?|'
             '\d{2}:\d{2}(:\d{2}(\.\d+)?)?|'
             '\d{2}\d{2}\d{2}\.\d+|'
             '\d{1,4}(-?\d{2,3})?'
             ')$')
    _matches_iso_format = re.compile(REGEX).match

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # None if no timezone, pytz object if any timezone
        self.timezone = None

    @staticmethod
    def column_looks_like_time(column):
        """
        Determine whether a column looks like it should be a TimeVariable.
        """
        # all values must be strings (otherwise integers under 10e5 would be years)
        # and be able to be parsed with python's datetime module
        return all(isinstance(val, str) and TimeVariable._matches_iso_format(val)
                   for val in column)

    @classmethod
    def _detect_timezone(cls, date_string):
        # detect a timezone from the first date, but only if we don't have one yet
        tzinfo = dateutil.parser.parse(date_string).tzinfo
        if tzinfo is None:
            return None
        else:
            offset = tzinfo.utcoffset(0)
            # for the common case where there is no offset, use UTC explicitly
            if offset == timedelta(0):
                return pytz.utc
            else:
                now = datetime.now(pytz.utc)
                appropriate_timezones = [tz for tz in pytz.all_timezones
                                         if now.astimezone(pytz.timezone(tz)).utcoffset() == offset]
                return appropriate_timezones[0] if appropriate_timezones else None

    def column_to_datetime(self, column):
        """
        Convert a column to a pandas datetime column.
        Takes note of the source timezone to display it correctly later.
        """
        for val in column:
            self.timezone = TimeVariable._detect_timezone(val) if not np.issubdtype(column.dtype, np.number) else None
            # if any value doesn't have a timezone, permanently strip display timezones for the column
            if self.timezone is None:
                break

        # if the columns are integers (timestamps), use different logic
        # than when we are dealing with strings
        if 'format' not in self.attributes and np.issubdtype(column.dtype, np.number):
            # timestamps are seconds
            kwargs = {'unit': 's'}
        else:
            # allow the variable to specify a format (overrides integers)
            kwargs = {'format': self.attributes.get('format')}

        # utc=True: make timezone aware
        # .values: return a DatetimeIndex so we can actually localize to UTC
        return pd.to_datetime(column.values, errors='raise', exact=True, utc=True,
                              infer_datetime_format=True, **kwargs)

    def repr_val(self, val):
        return str(val.tz_convert(self.timezone or None))

    str_val = repr_val

    def to_val(self, s):
        # unix float seconds
        return s.timestamp()
