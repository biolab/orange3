from collections import Iterable
import weakref
from .variable import *
from .instance import *

class DomainConversion:
    def __init__(self, source, destination):
        self.domain = destination
        self.attributes = [source.index(var) if var in source else var.get_value_from
                                for var in destination.attributes]
        self.classes = [source.index(var) if var in source else var.get_value_from
                                for var in destination.classes]
        self.variables = self.attributes + self.classes
        self.metas = [source.index(var) if var in source else var.get_value_from
                        for var in destination]

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

        self.known_domains = weakref.WeakKeyDictionary()
        self.last_conversion = None


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


    def get_conversion(self, domain):
        # the method is thread-safe
        c = self.last_conversion
        if c.domain is example.domain:
            return c
        c = self.known_domains.get(example.domain, None)
        if not c:
            c = DomainConversion(self, example.domain)
            self.known_domains[example.domain] = self.last_conversion = c
        return c

    def convert_as_list(self, example):
        if isinstance(example, Instance):
            if example.domain == self:
                return example.values, example.metas
            c = self.get_conversion(example.domain)
            attributes = [example._values[i] if isinstance(i, int) else
                      (Unknown if not i else i(example)) for i in c.attributes]
            classes = [example._values[i] if isinstance(i, int) else
                      (Unknown if not i else i(example)) for i in c.classes]
            metas = [example._values[i] if isinstance(i, int) else
                     (Unknown if not i else i(example)) for i in c.metas]
            return attributes, classes, metas
        return [var.to_val(val) for var, val in zip(self.attributes, example)], \
               [var.to_val(val) for var, val in zip(self.classes, example)], \
               []

    def convert(self, example, dst=None):
        if dst is not None:
            if isinstance(dst, FreeInstance):
                dst.domain = self
            else:
                raise ValueError(
                    "Destination is a row in a table from a different domain")

        attributes, classes, metas = self.convert_to_values(example)
        if dst is None:
            dst = FreeInstance(self, attributes + classes)
            dst.metas = metas
        else:
            dst.values = attributes + classes
            dst.metas = metas
            if isinstance(dst, RowInstance):
                dst._x[:] = attributes
                dst._y[:] = classes
                dst._metas[:] = metas
        return dst

    def convert_to_row(self, example, x, y, metas):
        if isinstance(example, Instance):
            if example.domain == self:
                if isinstance(example, RowInstance):
                    x[:] = example._x
                    y[:] = example._y
                else:
                    x[:] = example._values[:len(self.attributes)]
                    y[:] = example._values[len(self.attributes):]
                metas[:] = example._metas
                return
            c = self.get_conversion(example.domain)
            x[:] = [example._values[i] if isinstance(i, int) else
                    (Unknown if not i else i(example)) for i in c.attributes]
            y[:] = [example._values[i] if isinstance(i, int) else
                    (Unknown if not i else i(example)) for i in c.classes]
            metas[:] = [example._values[i] if isinstance(i, int) else
                    (Unknown if not i else i(example)) for i in c.metas]
        else:
            x[:] = [var.to_val(val)
                    for var, val in zip(self.attributes, example)]
            y[:] = [var.to_val(val)
                    for var, val in zip(self.class_vars, example[len(self.attributes):])]
            metas[:] = Unknown



    #TODO fast mapping of entire example tables, not just examples