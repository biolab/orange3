# Filters will be defined as follows:
# - when given a data instance, they return True/False, as usual
# - when given a data collection (Table, SQL proxy...)
#   -- they will inquire whether the collection implements a faster filter and
#      use it if possible;
#   -- if not, they will find out whether the collection provides a method to
#      construct a new collection or view based on row indices; if possible,
#      they will construct such a list and pass it to collection
#   -- otherwise, they construct a new collection and add examples one by one???

# Hint: parameters can be set using introspection into method arguments
# (set the arguments whose names match the filter's attributes)

from ..misc.enum import Enum

class Filter:
    def __init__(self, negate=False):
        self.negate = negate


class Values(Filter):
    def __init__(self, conjunction=True, conditions=[]):
        self.conjunction = conjunction
        self.conditions = conditions


class ValueFilter:
    Operator = Enum("Equal", "NotEqual",
                     "Less", "LessEqual", "Greater", "GreaterEqual",
                     "Between", "Outside",
                     "Contains", "BeginsWith", "EndsWith")

    def __init__(self, position):
        self.position = position


class FilterDiscrete(ValueFilter):
    def __init__(self, position, values):
        super().__init__(position)
        self.values = values


class FilterContinuous(ValueFilter):
    def __init__(self, position, oper, min=None, max=None, case_sensitive=True, **a):
        super().__init__(position)
        if a:
            if len(a) != 1 or "ref" not in a:
                raise TypeError("FilterContinuous got unexpected keyword arguments")
            else:
                min = a["ref"]
        self.min = min
        self.max = max
        self.oper = oper
        self.case_sensitive = True

    def get_ref(self):
        return self.min
    def set_ref(self, value):
        self.min = value
    ref = property(get_ref, set_ref)

class FilterString(ValueFilter):
    def __init__(self, position, oper, min=None, max=None, case_sensitive=True, **a):
        super().__init__(position)
        if a:
            print(a, "X")
            if len(a) != 1 or "ref" not in a:
                raise TypeError("FilterContinuous got unexpected keyword arguments")
            else:
                min = a["ref"]
        self.min = min
        self.max = max
        self.oper = oper
        self.case_sensitive = True

    def get_ref(self):
        return self.min
    def set_ref(self, value):
        self.min = value
    ref = property(get_ref, set_ref)

class FilterStringList(ValueFilter):
    def __init__(self, position, values, case_sensitive=True):
        super().__init__(position)
        self.min = min
        self.max = max
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