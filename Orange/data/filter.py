# Filters will be defined as follows:
# - when given a data instance, they return True/False, as usual
# - when given a data collection (Table, SQL proxy...)
#   -- they will inquire whether the collection implements a faster filter and
#      use it if possible;
#   -- if not, they will find out whether the collection provides a method to
#      construct a new collection or view based on row indices; if possible,
#      they will construct such a list and pass it to collection
#   -- otherwise, they construct a new collection and add examples one by one???

from ..misc.enum import Enum

class Filter:
    def __init__(self, negate=False):
        self.negate = negate


class Values(Filter):
    def __init__(self, conjunction=True, conditions=[]):
        self.conjunction = conjunction
        self.conditions = conditions


class ValueFilter:
    Operators = Enum("Equal", "NotEqual",
                     "Less", "LessEqual", "Greater", "GreaterEqual",
                     "Between", "Outside")

    def __init__(self, position):
        self.position = position


class ValueFilterDiscrete(ValueFilter):
    def __init__(self, position, values):
        super().__init__(self, position)
        self.values = values


class ValueFilterContinuous(ValueFilter):
    def __init__(self, position, min, max=None, oper=0, case_sensitive=True):
        super().__init__(self, position)
        self.min = min
        self.max = max
        self.oper = oper
        self.case_sensitive = True

    def get_ref(self):
        return self.min
    def set_ref(self, value):
        self.min = value
    ref = property(get_ref, set_ref)

class ValueFilterString(ValueFilter):
    def __init__(self, position, min, max=None, oper=0, case_sensitive=True):
        super().__init__(self, position)
        self.min = min
        self.max = max
        self.oper = oper
        self.case_sensitive = True


class ValueFilterStringList(ValueFilter):
    def __init__(self, position, values, case_sensitive=True):
        super().__init__(self, position)
        self.min = min
        self.max = max
        self.oper = oper
        self.case_sensitive = True

"""

from orange import\
    Filter as Filter,\
    FilterList as FilterList,\
    Filter_random as Random,\
    Filter_isDefined as IsDefined,\
    Filter_hasClassValue as HasClassValue,\
    Filter_hasMeta as HasMeta,\
    Filter_sameValue as SameValue,\
    Filter_values as Values,\
    Filter_hasSpecial as HasSpecial,\
    Filter_conjunction as Conjunction,\
    Filter_disjunction as Disjunction
"""