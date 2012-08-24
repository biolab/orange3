from ..data.variable import *
from collections import Iterable

class Domain:
    class MetaDescriptor:
        def __init__(self, variable, optional):
            self.variable = variable
            self.optional = optional

    version = 0

    def __init__(self, variables, class_variables=None, source=None):
        if isinstance(class_variables, Variable):
            attributes = list(variables)
            class_vars = [class_variables]
        elif isinstance(class_variables, Iterable):
            attributes = list(variables)
            class_vars = list(class_variables)
        else:
            variables = list(variables)
            if class_variables:
                attributes = variables[:-1]
                class_vars = variables[-1:]
            else:
                attributes = variables
                class_vars = []
        for lst in (attributes, class_vars):
            for i, var in enumerate(lst):
                if not isinstance(var, Variable):
                    lst[i] = source[var]
        self.attributes = tuple(attributes)
        self.class_vars = tuple(class_vars)
        self.variables = self.attributes + self.class_vars
        self.class_var = self.class_vars[0] if len(self.class_vars)==1 else None

        if not all(var.is_primitive for var in self.variables):
            raise TypeError("variables must be primitive")

        self.metas = []
        self.anonymous = False

        Domain.version += 1
        self.domain_version = Domain.version


    def var_from_domain(self, var, check_included=False, no_index=False):
        if isinstance(var, str):
            for each in self.variables:
                if each.name == var:
                    return each
            for each in self.metas:
                if each.name == var:
                    return each
            raise IndexError("Variable '%s' is not in the domain", var)
        if not no_index and isinstance(var, int):
            return self.variables[var] if var >= 0 else self.metas[-1-var]
        if isinstance(var, Variable):
            if check_included:
                for each in self.variables:
                    if each is var:
                        return var
                for each in self.metas:
                    if each is var:
                        return var
                raise IndexError("Variable '%s' is not in the domain", var.name)
            else:
                return var
        raise TypeError("Expected str, int or Variable, got '%s'" %
                        type(var).__name__)

    def __len__(self):
        return len(self.variables)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.variables[i]
                    for i in range(*index.indices(len(self.variables)))]
        return self.var_from_domain(index, True)

    def __contains__(self, item):
        try:
            self.var_from_domain(item, True)
            return True
        except IndexError:
            return False

    def __iter__(self):
        return iter(self.variables)


    def index(self, var):
        if isinstance(var, str):
            for i, each in enumerate(self.variables):
                if each.name == var:
                    return i
            for i, each in enumerate(self.metas):
                if each.name == var:
                    return -1-i
            raise ValueError("'%s' is not in domain" % var)
        if isinstance(var, Variable):
            if var in self.variables:
                return self.variables.index(var)
            if var in self.metas:
                return -1-self.metas.index(var)
            raise ValueError("'%s' is not in domain" % var.name)
        if isinstance(var, int):
            if -len(self.metas) <= var < len(self.variables):
                return var
            raise ValueError("there is no variable with index '%i'" % var)
        raise TypeError("Expected str, int or Variable, got '%s'" %
                        type(var).__name__)


    def has_discrete_attributes(self, include_class=False):
        return any(isinstance(var, DiscreteVariable)
                   for var in self.attributes) \
            or include_class and any(isinstance(var, DiscreteVariable)
                                     for var in self.class_vars)

    def has_continuous_attributes(self, include_class=False):
        return any(isinstance(var, ContinuousVariable)
                   for var in self.attributes) \
            or include_class and any(isinstance(var, ContinuousVariable)
                                     for var in self.class_vars)



    #TODO fast mapping of entire example tables, not just examples