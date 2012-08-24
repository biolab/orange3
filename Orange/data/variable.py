from ..misc.enum import Enum
import threading
from ..data.value import Value, Unknown
import collections

class Variable:
    VarTypes = Enum("None", "Discrete", "Continuous", "String")
    MakeStatus = Enum("OK", "MissingValues", "NoRecognizedValues",
        "Incompatible", "NotFound")
    DefaultUnknownStr =  {"?", ".", "", "NA", "~", None}

    def __init__(self, var_type, name="", ordered=False, default_col = -1):
        self.var_type = var_type
        self.name = name
        self.ordered = ordered
        self.default_col = default_col
        self.random_generator = None
        self.source_variable = None
        self.get_value_from = None
        self.get_value_lock = threading.Lock()
        self.unknown_str = set(Variable.DefaultUnknownStr)

    def computeValue(self, inst):
        if self.get_value_from is None:
            return Value(self)
        with self.get_value_lock:
            return Value(self, self.get_value_from(inst))

    def is_primitive(self):
        return False

    def repr_val(self, val):
        return str(val)

    str_val = repr_val

    def __str__(self):
        return "{}('{}')".format(self.__class__.__name__, self.name)

    __repr__ = __str__



class DiscreteVariable(Variable):
    all_discrete_vars = collections.defaultdict(set)
    presorted_values = [["no", "yes"], ["No", "Yes"]]

    def __init__(self, name="", values=(), ordered=False, default_col=-1, base_value=-1):
        super().__init__(Variable.VarTypes.Discrete,
            name, ordered, default_col)
        self.values = list(values)
        self.base_value = base_value
        DiscreteVariable.all_discrete_vars[name].add(self)

    def __str__(self):
        args = "values=[" + ", ".join(self.values[:5]) + \
               "..."*(len(self.values)>5) + "]"
        if self.ordered:
            args += ", ordered=True"
        if self.base_value >= 0:
            args += ", base_value={}".format(self.base_value)
        return "{}('{}', {})".format(self.__class__.__name__, self.name, args)

    @staticmethod
    def is_primitive(self):
        return True

    def to_val(self, s):
        if isinstance(s, int):
            return s
        if s in self.unknown_str:
            return Unknown
        if not isinstance(s, str):
            raise TypeError('Cannot convert {} to value of "{}"'
                            .format(type(s).__name__, self.name))
        return self.values.index(s)

    def val_from_str_add(self, s):
        try:
            return Unknown if s in self.unknown_str else self.values.index(s)
        except ValueError:
            self.values.append(s)
            return len(self.values) - 1

    def repr_val(self, val):
        return '{}'.format(self.values[int(val)])

    str_val = repr_val

    def check_values_order(self, values):
        return self.values[:len(values)] == values[:len(self.values)]

    @staticmethod
    def make(name, values=(), ordered=False, default_meta_id=-1, base_value=-1):
        var = DiscreteVariable.find_compatible(
            name, values, ordered, default_meta_id, base_value)
        if var:
            return var
        return DiscreteVariable(name, values, ordered, default_meta_id, base_value)

    @staticmethod
    def find_compatible(name, values=(), ordered=False, default_col=-1, base_value=-1):
        existing = DiscreteVariable.all_discrete_vars.get(name)
        if existing:
            for var in existing:
                if var.n_existing_values(values) >= 0 and var.ordered == ordered:
                    if base_value != -1:
                        if var.base_value == -1:
                            var.base_value = base_value
                        elif var.base_value != base_value:
                            continue
                    if default_col != -1:
                        if var.default_col == -1:
                            var.default_col = default_col
                        elif var.default_col != default_col:
                            continue
                    return var

    @staticmethod
    def order_values(values):
        values = set(values)
        for presorted in DiscreteVariable.presorted_values:
            if values == set(presorted):
                return presorted
        return sorted(values)

    def add_values(self, unordered, ordered=()):
        used = self.n_existing_values(ordered)
        if used < 0:
            raise ValueError("Mismatching order of values")
        self.values += ordered[used:]
        remaining = set(unordered) - set(self.values)
        self.values += self.order_values(remaining)

    def n_existing_values(self, ordered):
        i = 0
        for val in self.values:
            if i < len(ordered) and ordered[i] == val:
                i += 1
        if set(ordered[i:]) & set(self.values):
            return -1
        return i



class ContinuousVariable(Variable):
    all_continuous_vars = {}

    def __init__(self, name="", default_col=-1):
        super().__init__(Variable.VarTypes.Continuous, name, default_col)
        self.number_of_decimals = 3
        self.scientific_format = False
        self.adjustDecimals = 2
        ContinuousVariable.all_continuous_vars[name] = self

    @staticmethod
    def make(name, default_meta_id=-1):
        return ContinuousVariable.all_continuous_vars.get(name) or \
               ContinuousVariable(name, default_meta_id)

    def is_primitive(self):
        return False

    def to_val(self, s):
        return Unknown if s in self.unknown_str else float(s)

    val_from_str_add = to_val

    def repr_val(self, val):
        return "%.*f" % (self.number_of_decimals, val)

    str_val = repr_val


class StringVariable(Variable):
    all_string_vars = {}

    def __init__(self, name="", default_col=-1):
        super().__init__(Variable.VarTypes.String, name, default_col)
        StringVariable.all_string_vars[name] = self

    def to_val(self, s):
        if s is None:
            return ""
        if isinstance(s, str):
            return s
        return str(s)

    val_from_str_add = to_val

    def str_val(self, val):
        if isinstance(val, Value):
            if val.value is None:
                return "None"
            val = val.value
        return str(val)

    def repr_val(self, val):
        return '"{}"'.format(self.str_val(val))

    @staticmethod
    def make(name, default_meta_id=-1):
        return StringVariable.all_string_vars.get(name) or \
               StringVariable(name, default_meta_id)
