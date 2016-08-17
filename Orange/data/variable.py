import re
from numbers import Number, Real, Integral
from math import isnan, floor, sqrt
from pickle import PickleError
import copy
import dateutil
import pytz
import collections
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from Orange.util import Registry, color_to_hex, hex_to_color


# For storing unknowns
Unknown = ValueUnknown = float("nan")


def make_variable(cls, compute_value, *args):
    if compute_value is not None:
        return cls(*args, compute_value=compute_value)
    return cls.make(*args)


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

    Attributes
    ----------
    name : str
        The name of the variable.
    compute_value : Callable
        A function for computing the variable's value when converting from
        another domain which does not contain this variable. The base class
        defines a static method `compute_value`, which returns `Unknown`.
        Non-primitive variables must redefine it to return `None`.
    source_variable : Variable
        An optional descriptor of the source variable - if any - from which
        this variable is derived and computed via :obj:`compute_value`.
    attributes : dict
        A dictionary with user-defined attributes of the variable
    master : Variable
        The variable that this variable is a copy of. If a copy is made from a
        copy, the copy has a reference to the original master. If the variable
        is not a copy, it is its own master.
    """
    Unknown = ValueUnknown
    MISSING_VALUES = {np.nan, "?", "nan", ".", "", "NA", "~"}

    def __new__(cls, name="", *args, **kwargs):
        # compatibility with str
        return super().__new__(cls, name)

    def __init__(self, name="", compute_value=None):
        """Construct a variable descriptor."""
        super().__init__()
        self.name = name
        self._compute_value = compute_value
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
        """Copy the variable and set the master to `self.master` or to `self`.

        Returns
        -------
        Variable
            A copy of self.
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
        """Make a new variable with respect to the cache.

        Parameters
        ----------
        name : str
            The name of the variable.

        Returns
        -------
        Variable
            An existing continuous variable with the given name, or
            construct and return a new one.
        """
        if not name:
            raise ValueError("Variables without names cannot be stored or made")
        return cls._all_vars.get(name) or cls(name)

    @classmethod
    def _clear_cache(cls):
        """Clear the list of variables for reuse by :obj:`make`."""
        cls._all_vars.clear()

    @staticmethod
    def _clear_all_caches():
        """Clears list of stored variables for all subclasses"""
        for cls in Variable.registry.values():
            cls._clear_cache()

    @classmethod
    def is_primitive(cls):
        """Determine whether the variable is primitive.

        Non-primitive variables can appear in the data only as meta attributes.

        Returns
        -------
        bool
            True if if the variable's values are stored as floats.
        """
        return issubclass(cls, (DiscreteVariable, ContinuousVariable))

    @property
    def is_discrete(self):
        """Determine whether the variable is discrete."""
        return isinstance(self, DiscreteVariable)

    @property
    def is_continuous(self):
        """Determine whether the variable is continuous."""
        return isinstance(self, ContinuousVariable)

    @property
    def is_string(self):
        """Determine whether the variable has string values."""
        return isinstance(self, StringVariable)

    def repr_val(self, val):
        """Return a textual representation val, determined by the variable.

        Parameters
        ----------
        val
            The value for which to generate a textual representation.

        Returns
        -------
        str
            The textual representation of the value.

        Notes
        -----
        Derived classes must overload the function.
        """
        raise RuntimeError("variable descriptors must overload repr_val()")

    str_val = repr_val

    def to_val(self, s):
        """Convert the given argument to the (numeric) value of the variable.

        For continuous variables, output a float representation of the data.
        For discrete variables, return the indices of its variable.values.
        For string variables, return the string.

        Must support converting either single values or a complete column (pd.Series).
        The column operation should be faster than iterating and transforming
        one value at a time.

        Parameters
        ----------
        s : pd.Series or Number or str
            The value(s) to generate a numeric representation for.

        Returns
        -------
        pd.Series or Number or str
            The (numeric) representation of the given values.
        """
        if not self.is_primitive():
            return s
        if s in self.MISSING_VALUES:
            return Unknown
        raise RuntimeError(
            "primitive variable descriptors must overload to_val()")

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

    def copy(self, compute_value=None, new_name=None):
        """Make a deep copy of this variable."""
        var = Variable(new_name or self.name, compute_value)
        var.attributes = dict(self.attributes)
        return var


class ContinuousVariable(Variable):
    """
    A descriptor for continuous variables.

    Attributes
    ----------
    number_of_decimals : int, default 3
        The number of decimals when the value is printed out.
    adjust_decimals : int, default 2
        A flag regulating whether the `number_of_decimals` is being adjusted by :obj:`to_val`.
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
        if isinstance(s, pd.Series):
            return s.astype(float)
        else:
            return float(s)

    def repr_val(self, val):
        """Return the value as a string with the prescribed number of decimals."""
        if isnan(val):
            return "?"
        return self._out_format % val

    str_val = repr_val

    def copy(self, compute_value=None, new_name=None):
        var = ContinuousVariable(new_name or self.name, self.number_of_decimals, compute_value)
        var.attributes = dict(self.attributes)
        return var


class DiscreteVariable(Variable):
    """
    Descriptor for symbolic, discrete variables. Values of discrete variables
    are stored as floats; the numbers corresponds to indices in the list of
    values.

    Attributes
    ----------
    name : str
        The name of the variable.
    values : list
        A list of variable's values.
    ordered : bool, default False
        Some algorithms (and, in particular, visualizations) may
        sometime reorder the values of the variable, e.g. alphabetically.
        This flag hints that the given order of values is "natural"
        (e.g. "small", "middle", "large") and should not be changed.
    base_value : int, default -1
        The index of the base value, or -1 if there is none. The base value is
        used in some methods like, for instance, when creating dummy variables
        for regression.
    compute_value : Callable, default None
        A function to compute this variable's values from a source table.
    """
    TYPE_HEADERS = ('discrete', 'd')

    _all_vars = collections.defaultdict(list)

    def __init__(self, name="", values=(), ordered=False, base_value=-1, compute_value=None):
        """Construct a discrete variable descriptor with the given values."""
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
        """Give a string representation of the variable.

        Returns
        -------
        str
            The string represtation of the variable.

        Examples
        --------
        `"DiscreteVariable('Gender', values=['male', 'female'])"`
        """
        args = "values=[{}]".format(
            ", ".join([repr(x) for x in self.values[:5]] +
                      ["..."] * (len(self.values) > 5)))
        if self.ordered:
            args += ", ordered=True"
        if self.base_value >= 0:
            args += ", base_value={}".format(self.base_value)
        return "{}('{}', {})".format(self.__class__.__name__, self.name, args)

    def to_val(self, c):
        def transform_func(s):
            if pd.isnull(s):
                return np.nan
            # performs better than a dict; at least for a reasonable amount of categories
            return self.values.index(s)
        if isinstance(c, pd.Series):
            # compared to the complicated to_val, this is orders of magnitude faster
            # testing on adult.occupation:
            # list comprehension to_val: 3.498
            # .map(reverse dict of .values): 0.237
            # .apply(to_val): 0.059
            # col.cat.codes: 0.007
            # col.cat.codes.replace(-1, np.nan): 0.225
            # it would be great to use pandas' codes, but that returns -1 for unknown values,
            # whereas we need np.nan - and converting those to nan later is slower than applying
            # I suspect this is because application is sped up with an accelerator
            return c.apply(transform_func)
        else:
            return transform_func(c)

    def add_value(self, s):
        """Add a value `s` to the list of values."""
        self.values.append(s)

    def repr_val(self, val):
        if (isinstance(val, Number) and isnan(val)) or not val:
            return "?"
        return str(val)

    str_val = repr_val

    def __reduce__(self):
        if not self.name:
            raise PickleError("Variables without names cannot be pickled")
        return make_variable, (self.__class__, self._compute_value, self.name,
                               self.values, self.ordered, self.base_value), \
            self.__dict__

    @classmethod
    def make(cls, name, values=(), ordered=False, base_value=-1):
        """Return a variable with the given name and other properties.

        The method first looks for a compatible existing variable: the existing
        variable must have the same name and both variables must have either
        ordered or unordered values. If values are ordered, the order must be
        compatible: all common values must have the same order. If values are
        unordered, the existing variable must have at least one common value
        with the new one, except when any of the two lists of values is empty.

        If a compatible variable is find, it is returned, with missing values
        appended to the end of the list. If there is no explicit order, the
        values are ordered using sorted. Otherwise, it
        constructs and returns a new variable descriptor.

        Parameters
        ----------
        name : str
            The name of the variable.
        values : list
            Symbolic values for the variable.
        ordered : bool, default False
            Whether the order of the values is significant.
        base_value : int, default -1
            The index of the base value or -1 if there is none.

        Returns
        -------
        Variable
            An existing compatible variable or a new variable.
        """
        if not name:
            raise ValueError("Variables without names cannot be stored or made")
        var = cls._find_compatible(name, values, ordered, base_value)
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
        """Return a compatible existing variable if it exists.

        Parameters
        ----------
        name : str
            The name of the variable.
        values : list
            Symbolic values for the variable.
        ordered : bool, default False
            Whether the order of the values is significant.
        base_value : int, default -1
            The index of the base value or -1 if there is none.

        Returns
        -------
        Variable
            A compatible existing variable if it exists.

        See Also
        --------
        See :obj:`make` for details; this function differs by returning `None`
        instead of constructing a new descriptor. (Method :obj:`make` calls
        this function.)
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

    def copy(self, compute_value=None, new_name=None):
        var = DiscreteVariable(new_name or self.name, self.values, self.ordered,
                               self.base_value, compute_value)
        var.attributes = dict(self.attributes)
        return var

    @classmethod
    def generate_unique_values(cls, column):
        """Generate a sorted set of unique values.

        Takes into account values we consider missing (Variable.MISSING_VALUES).

        Parameters
        ----------
        column : pd.Series
            The column to generate unique values of.

        Returns
        -------
        list
            A sorted list of unique values.
        """
        # comparing np.nan doesn't always work, use the appropriate mechanism
        raw = column[~column.isnull()].unique()
        return sorted([v for v in raw if v not in Variable.MISSING_VALUES])


class StringVariable(Variable):
    """Descriptor for string variables. String variables can only appear as meta attributes."""
    Unknown = ""
    TYPE_HEADERS = ('string', 's', 'text')

    def to_val(self, c):
        def transform_func(s):
            if s is None or (isinstance(s, Number) and np.isnan(s)):
                return ""
            if isinstance(s, str):
                return s
            return str(s)
        if isinstance(c, pd.Series):
            return c.apply(transform_func)
        else:
            return transform_func(c)

    @staticmethod
    def str_val(val):
        if val in Variable.MISSING_VALUES or pd.isnull(val):
            return "?"
        return str(val)

    def repr_val(self, val):
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
        # True if the time component exists (and will be displayed), parsing in column_to_datetime
        self.has_time_component = True

    @staticmethod
    def column_looks_like_time(column):
        """Determine whether a column looks like it should be a TimeVariable.

        Parameters
        ----------
        column : pd.Series
            The column to check.

        Returns
        -------
        bool
            True if the column's values look like they are times.
        """
        # all values must be strings (otherwise integers under 10e5 would be years)
        # and be able to be parsed with python's datetime module
        return all((isinstance(val, str) and TimeVariable._matches_iso_format(val))
                   or val in Variable.MISSING_VALUES
                   for val in column)

    @classmethod
    def _detect_timezone(cls, date_string):
        """Detect a timezone from a date string.

        Parameters
        ----------
        date_string : str
            The date string to check.

        Returns
        -------
        pytz.timezone
            An appropriate pytz timezone object or None if no timezone.
        """
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
        """Convert a column to a pandas datetime column.

        Takes note of the source timezone to display it correctly later.

        Parameters
        ----------
        column : pd.Series
            The column to transform.

        Returns
        -------
        A transformed pd.Series of the datetime type, with timezone information.
        """
        for val in column:
            # handle missing values like they don't exist
            if val in Variable.MISSING_VALUES or (isinstance(val, Number) and np.isnan(val)):
                continue
            # for multiple timezones, use the last one for display
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

        # .apply(str) because parsing a float discards the fractional part for some reason
        # utc=True: make timezone aware
        # .values: return a DatetimeIndex so we can actually localize to UTC
        result = pd.to_datetime(column.apply(str).values, errors='raise', exact=True, utc=True,
                                infer_datetime_format=True, **kwargs)

        # determine whether we should display the time part (HH:MM:SS.MS)
        # only display it when hours, minutes and seconds are all 0 in all cases
        # this is the most robust way as it doesn't depend on regexes
        # (we might not even have strings as inputs) and offloads any parsing to pandas
        # addition works because everything is non-negative
        self.has_time_component = (result.hour + result.minute + result.second + result.microsecond).sum() != 0
        return result

    def repr_val(self, val):
        if val is pd.NaT:
            return "?"
        else:
            tzval = val.tz_convert(self.timezone)
            return str(tzval) if self.has_time_component else tzval.strftime("%Y-%m-%d")

    str_val = repr_val

    def to_val(self, s):
        # unix float seconds
        if isinstance(s, pd.Series):
            return s.ts.timestamp
        else:
            return s.timestamp()
