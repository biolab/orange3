from numbers import Real
from ..misc.enum import Enum
import threading
from ..data.value import Value, Unknown
import collections
from math import isnan, floor


class Variable:
    """
    The base class for variable descriptors, which contains the variable's
    name, type and basic properties.

    .. attribute:: name

        The name of the variable.

    .. attribute:: ordered

        A flag which tells whether the variable`s values are ordered.

    .. attribute:: unknown_str

        A set of values that represent unknowns in conversion form textual
        formats. Default is :obj:`Variable.DefaultUnknownStr`, which contains
        "?", ".", "", "NA" and "~" (and, for technical reasons `None`).

    .. attribute:: get_value_from

        A function for computing the variable's value when converting from
        another domain which does not have this variable. Details are described
        below.

    .. attribute:: source_variable

        If a variable is computed (via :obj:`getValueFrom` from another
        variable, this attribute contains its descriptor.

    .. attribute:: attributes

        A dictionary with user-defined attributes of the variable
    """
    MakeStatus = Enum("OK", "MissingValues", "NoRecognizedValues",
                      "Incompatible", "NotFound")
    DefaultUnknownStr = {"?", ".", "", "NA", "~", None}

    variable_types = []

    def __init__(self, name="", ordered=False):
        """
        Construct a variable descriptor and store the general properties of
        variables.
        """
        self.name = name
        self.ordered = ordered
        self.unknown_str = set(Variable.DefaultUnknownStr)
        self.source_variable = None
        self.attributes = {}
        self.get_value_from = None
        self._get_value_lock = threading.Lock()
        # TODO: do we need locking? Don't we expect the code to be reentrant?

        # TODO: the original intention was to prevent deadlocks. Locks block,
        #       we would need more like a semaphore. But then - which
        #       reasonable use of get_value_from can lead to deadlocks?!

    def compute_value(self, inst):
        """
        Call get_value_from if it exists; return :obj:`Unknown` otherwise.

        :param inst: Data instance from the original domain
        :type inst: data.Instance
        :rtype: float
        """
        if self.get_value_from is None:
            return Unknown
        with self._get_value_lock:
            return self.get_value_from(inst)

    @staticmethod
    def is_primitive():
        """
        Tell whether the value is primitive, that is, represented as a float,
        not a Python object. Primitive variables are
        :obj:`~data.DiscreteVariable` and :obj:`~data.ContinuousVariable`.

        Derived classes should overload the function.
        """
        raise RuntimeError(
            "variable descriptors should overload is_primitive()")

    def repr_val(self, val):
        """
        Return a textual representation of variable's value `val`. Argument
        `val` can be a float (for primitive variables) or a an arbitrary
        Python object (for non-primitives).

        Derived classes should overload the function.
        """
        raise RuntimeError("variable descriptors should overload repr_val()")

    str_val = repr_val

    def __str__(self):
        """
        Return a representation of the variable, like,
        `'DiscreteVariable("gender")'`. Derived classes may overload this
        method to provide a more informative representation.
        """
        return "{}('{}')".format(self.__class__.__name__, self.name)

    __repr__ = __str__

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_get_value_lock")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._get_value_lock = threading.Lock()

    @classmethod
    def clear_cache(cls):
        for tpe in cls.variable_types:
            tpe.clear_cache()

class DiscreteVariable(Variable):
    """
    Descriptor for symbolic, discrete variables. Besides the inherited
    attributes, it stores a list of values. Discrete variables are stored
    as floats; the number corresponds to the index in the list of values.

    .. attribute:: values

        A list of values for the attribute.

    .. attribute:: base_value

        The index of the base value, or -1 if there is none. The base value is
        used in some methods like, for instance, when creating dummy variables
        for regression.
    """
    all_discrete_vars = collections.defaultdict(set)
    has_numeric_values = False
    presorted_values = []

    def __init__(self, name="", values=(), ordered=False, base_value=-1):
        """ Construct a discrete variable descriptor with the given values. """
        super().__init__(name, ordered)
        self.values = list(values)
        self.base_value = base_value
        DiscreteVariable.all_discrete_vars[name].add(self)

    def __str__(self):
        """
        Give a string representation of the variable, for instance,
        `"DiscreteVariable('Gender', values=['male', 'female'])"`.
        """
        args = "values=[" + ", ".join(self.values[:5]) +\
               "..." * (len(self.values) > 5) + "]"
        if self.ordered:
            args += ", ordered=True"
        if self.base_value >= 0:
            args += ", base_value={}".format(self.base_value)
        return "{}('{}', {})".format(self.__class__.__name__, self.name, args)

    @staticmethod
    def is_primitive():
        """ Return `True`: discrete variables are stored as floats. """
        return True

    def to_val(self, s):
        """
        Convert the given argument to a value of the variable (`float`).
        If the argument is numeric, its value is returned without checking that
        it is integer and withing bounds. `Unknown` is returned if the argument
        is one of the representations for unknown values. Otherwise, the
        argument must be a string and the method returns its index in
        :obj:`values`.

        :param s: values, represented as a number, string or `None`
        :rtype: float
        """
        if self.has_numeric_values:
            s = str(s)

        if isinstance(s, int):
            return s
        if isinstance(s, Real):
            return s if isnan(s) else floor(s + 0.25)
        if s in self.unknown_str:
            return Unknown
        if not isinstance(s, str):
            raise TypeError('Cannot convert {} to value of "{}"'.format(
                type(s).__name__, self.name))
        return self.values.index(s)

    def add_value(self, s):
        """ Add a value to the list of values.
        """
        self.values.append(s)

    def val_from_str_add(self, s):
        """
        Similar to :obj:`to_val`, except that it accepts only strings and that
        it adds the value to the list if it does not exist yet.

        :param s: symbolic representation of value
        :type s: str
        :rtype: float
        """
        try:
            return Unknown if s in self.unknown_str else self.values.index(s)
        except ValueError:
            self.add_value(s)
            return len(self.values) - 1

    def repr_val(self, val):
        """
        Return a textual representation of a value. The result is a "?" for
        unknowns and the symbolic representation from :obj:`values`
        (that is, `self.values[int(val)]`) otherwise.

        :param val: value
        :type val: float (should be whole number)
        :rtype: str
        """
        if isnan(val):
            return "?"
        return '{}'.format(self.values[int(val)])

    str_val = repr_val

    @staticmethod
    def make(name, values=(), ordered=False, base_value=-1):
        """
        Return a variable with the given name and other properties. The method
        first looks for a useful existing variable. First, the existing
        variable must have the same name and both variables must have either
        ordered or unordered values. If values are ordered, the order must be
        compatible: all common values must have the same order. If values are
        unordered, the existing variable must have at least one common value
        with the new one, except when any of the two lists of values is empty.

        If a compatible variable is find, it is returned, with missing values
        appended to the end of the list. If there is no explicit order, the
        values are ordered using :obj:`ordered_values`. Otherwise, it
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
        var = DiscreteVariable.find_compatible(name, values, ordered,
                                               base_value)
        if var:
            return var
        if not ordered:
            base_value_rep = base_value != -1 and values[base_value]
            values = DiscreteVariable.ordered_values(values)
            if base_value != -1:
                base_value = values.index(base_value_rep)
        return DiscreteVariable(name, values, ordered, base_value)

    @staticmethod
    def find_compatible(name, values=(), ordered=False, base_value=-1):
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
        existing = DiscreteVariable.all_discrete_vars.get(name)
        if not existing:
            return None
        if not ordered:
            values = DiscreteVariable.ordered_values(values)
        for var in existing:
            if (var.ordered != ordered or
                    var.base_value != -1
                    and var.values[var.base_value] != base_rep):
                continue
            if ordered:
                i = 0
                for val in var.values:
                    if values[i] == val:
                        i += 1
                        if i == len(values):
                            break  # we have all the values
                else:  # We have some remaining values: check them, add them
                    if set(ordered[i:]) & set(var.values):
                        continue  # next var in existing
                    for val in values[i:]:
                        var.add_value(val)
                break  # we have the variable
            elif not var.values or not values or set(var.values) & set(values):
                vv = set(var.values)
                for val in values:
                    if val not in vv:
                        var.add_value(val)
                break
        else:
            return None
        if base_value != -1 and var.base_value == -1:
            var.base_value = var.values.index(base_rep)
        return var

    @classmethod
    def clear_cache(cls):
        """
        Cleans the list of variables for reuse by :obj:`make`.
        """
        cls.all_discrete_vars.clear()

    @staticmethod
    def ordered_values(values):
        """
        Return a sorted list of values. If there exists a prescribed order for
        such set of values, it is returned. Otherwise, values are sorted
        alphabetically.
        """
        for presorted in DiscreteVariable.presorted_values:
            if values == set(presorted):
                return presorted
        return sorted(values)


class ContinuousVariable(Variable):
    """
    Descriptor for continuous variables. Additional attributes describe the
    output format.

    .. attribute:: number_of_decimals

        The number of decimals in the textual representation

    .. attribute:: adjust_decimals

        A flag telling whether the number of decimals need to be adjusted
        according to strings passed to :obj:`to_val`: 0 for no adjustment,
        1 for increasing the number of decimals whenever a value with a
        larger number of decimals is passed to :obj:`to_val`, and 2 if the
        number of decimals is still at the default (3) and needs to be set
        at the first call to :obj:`to_val`.
    """
    all_continuous_vars = {}

    def __init__(self, name=""):
        """
        Construct a new continuous variable. The number of decimals is set to
        three, but adjusted at the first call of :obj:`to_val`.
        """
        super().__init__(name)
        self.number_of_decimals = 3
        self.adjust_decimals = 2
        ContinuousVariable.all_continuous_vars[name] = self

    @property
    def number_of_decimals(self):
        return self._number_of_decimals

    # noinspection PyAttributeOutsideInit
    @number_of_decimals.setter
    def number_of_decimals(self, x):
        self._number_of_decimals = x
        self._out_format = "%.{}f".format(self.number_of_decimals)

    @staticmethod
    def make(name):
        """
        Return an existing continuous variable with the given name, or
        construct and return a new one.
        """
        existing_var = ContinuousVariable.all_continuous_vars.get(name)
        return existing_var or ContinuousVariable(name)

    @classmethod
    def clear_cache(cls):
        """
        Cleans the list of variables for reuse by :obj:`make`.
        """
        cls.all_continuous_vars.clear()

    @staticmethod
    def is_primitive():
        """ Return `True`: continuous variables are stored as floats."""
        return True

    def to_val(self, s):
        """
        Convert a value, given as an instance of an arbitrary type, to a float.
        If the value is given as a string, it adjusts the number of decimals
        if needed.
        """
        if s in self.unknown_str:
            return Unknown
        if self.adjust_decimals and isinstance(s, str):
            #TODO: This may significantly slow down file reading.
            #      Is there something we can do about it?
            ndec = s.strip().rfind(".")
            ndec = len(s) - ndec - 1 if ndec != -1 else 0
            if self.adjust_decimals == 2 or ndec > self.number_of_decimals:
                self.number_of_decimals = ndec
        return float(s)

    val_from_str_add = to_val

    def repr_val(self, val):
        """
        Return the value as a string with the prescribed number of decimals.
        """
        if isnan(val):
            return "?"
        return self._out_format % val

    str_val = repr_val


class StringVariable(Variable):
    """
    Descriptor for string variables.
    """
    all_string_vars = {}

    def __init__(self, name="", default_col=-1):
        """Construct a new descriptor."""
        super().__init__(name, default_col)
        StringVariable.all_string_vars[name] = self

    @staticmethod
    def is_primitive():
        """Return `False`: string variables are not stored as floats."""
        return False

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

    def str_val(self, val):
        """Return a string representation of the value."""
        if isinstance(val, Value):
            if val.value is None:
                return "None"
            val = val.value
        return str(val)

    def repr_val(self, val):
        """Return a string representation of the value."""
        return '"{}"'.format(self.str_val(val))

    @staticmethod
    def make(name):
        """
        Return an existing string variable with the given name, or construct
        and return a new one.
        """
        existing_var = StringVariable.all_string_vars.get(name)
        return existing_var or StringVariable(name)

    @classmethod
    def clear_cache(cls):
        """
        Cleans the list of variables for reuse by :obj:`make`.
        """
        cls.all_string_vars.clear()

Variable.variable_types += [DiscreteVariable, ContinuousVariable, StringVariable
    ]
