from collections import Iterable
from itertools import chain
import weakref
from .variable import *
import numpy as np

class DomainConversion:
    """
    Indices and functions for conversion between domains.

    Every list contains indices (instances of int) of variables in the
    source domain, or the variable's get_value_from function if the source
    domain does not contain the variable.

    .. attribute:: source

        The source domain. The destination is not specifically stored since
        destination domain is the one which contains the instance of
        DomainConversion.

    .. attribute:: attributes

        Indices for attribute values.

    .. attribute:: class_vars

        Indices for class variables

    .. attribute:: variables

        Indices for attributes and class variables
        (:obj:`attributes`+:obj:`class_vars`).

    .. attribute:: metas

        Indices for meta attributes
    """

    def __init__(self, source, destination):
        """
        Compute the conversion indices from the given source to destination
        """
        self.source = source
        self.attributes = [source.index(var) if var in source else var.get_value_from
                                for var in destination.attributes]
        self.class_vars = [source.index(var) if var in source else var.get_value_from
                                for var in destination.class_vars]
        self.variables = self.attributes + self.class_vars
        self.metas = [source.index(var) if var in source else var.get_value_from
                        for var in destination.metas]


class Domain:
    """
    Description of a domain. It stores a list of attribute, class and meta
    attribute descriptors with methods for indexing, printing out and
    conversion between domain.

    .. attribute:: attributes

        A list of descriptors (instances of :class:`Orange.data.Variable`)
        for attributes (features, independent variables).

    .. attribute:: class_vars

        A list of descriptors for class attributes (outcomes, dependent
        variables).

    .. attribute:: variables

        A list of attributes and class attributes (the concatenation of
        the above).

    .. attribute:: class_var

        Class variable if the domain has a single class; `None` otherwise.

    .. attribute:: metas

        List of meta attributes.

    .. attribute:: anonymous

        True if the domain was constructed when converting numpy array to
        :class:`Orange.data.Table`. Such domains can be converted to and
        from other domains even if they consist of different variables for
        as long as their types match.

    .. attribute:: known_domains

        A weak dictionary containing instances of :class:`DomainConversion` for
        conversion from other domains into this one. Source domains are used
        as keys. The dictionary is used for caching the conversions constructed
        by :method:`get_conversion`.

    .. attribute:: last_conversion

        The last conversion returned by :method:`get_conversion`. Most
        conversions into the domain used the same domain as a source, so
        storing the last conversions saves a lookup into
        :attribute:`known_domains`.
    """

    def __init__(self, variables, class_variables=None, source=None):
        """
        Initialize a new domain descriptor. Arguments give the features and
        the class attribute(s). Feature and attributes can be given by
        descriptors (instances of :class:`Variable`) or by indices and names
        if the source domain is given.

        :param variables: a list of variables; if sole argument, it includes the class
        :param class_variables: a list of class variables
        :param source: the source domain for attributes
        :return: a new instance of :class:`domain`
        """
        #TODO use source if provided!
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

        self.known_domains = weakref.WeakKeyDictionary()
        self.last_conversion = None


    def var_from_domain(self, var, check_included=False, no_index=False):
        """
        Return a variable descriptor from the given argument, which can be
        a descriptor, index or name. If `var` is a descriptor, the function
        return it

        :param var: an instance of :class:`Variable`, int or str
        :param check_included: if `var` is an instance of :class:`Variable`,
            this flags tells whether to check that the domain contains this
            variable
        :param no_index: if True, `var` must not be an int
        :return: an instance of :class:`Variable` described by `var`
        """
        if isinstance(var, str):
            for each in chain(self.variables, self.metas):
                if each.name == var:
                    return each
            raise IndexError("Variable '%s' is not in the domain", var)

        if not no_index and isinstance(var, int):
            return self.variables[var] if var >= 0 else self.metas[-1-var]

        if isinstance(var, Variable):
            if check_included:
                for each in chain(self.variables, self.metas):
                    if each is var:
                        return var
                raise IndexError("Variable '%s' is not in the domain", var.name)
            else:
                return var

        raise TypeError("Expected str, int or Variable, got '%s'" %
                        type(var).__name__)

    def __len__(self):
        """Return the number of variables (features and class attributes)"""
        return len(self.variables)

    def __getitem__(self, index):
        """
        Same as var_from_domain. Index can be a slice, an int, str or
        instance of :class:`Variable`.
        """
        if isinstance(index, slice):
            return self.variables[index]
        return self.var_from_domain(index, True)

    def __contains__(self, item):
        """
        Return true if the item (str, int, :class:`Variable`) is in the domain.
        """
        try:
            self.var_from_domain(item, True)
            return True
        except IndexError:
            return False

    def __iter__(self):
        """
        Return an iterator through variables (features and class attributes)
        """
        return iter(self.variables)

    def __str__(self):
        """
        Return a list-like string with the domain's features, class attributes
        and meta attributes
        """
        s = "[" + ", ".join(attr.name for attr in self.attributes)
        if self.class_vars:
            s += " | " + ", ".join(cls.name for cls in self.class_vars)
        s += "]"
        if self.metas:
            s += "{" + ", ".join(meta.name for meta in self.metas) + "}"
        return s

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("known_domains")
        return state

    def index(self, var):
        """
        Return the index of the given variable or meta attribute, represented
        with an instance of :class:`Variable`, int or str.
        """
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
        """
        Return True if domain has any discrete attributes.

        :param include_class: if True (default is False), the check includes
            the class attribute(s)
        """
        return any(isinstance(var, DiscreteVariable)
                   for var in self.attributes) \
            or include_class and any(isinstance(var, DiscreteVariable)
                                     for var in self.class_vars)

    def has_continuous_attributes(self, include_class=False):
        """
        Return True if domain has any continuous attributes.

        :param include_class: if True (default is False), the check includes
            the class attribute(s)
        """
        return any(isinstance(var, ContinuousVariable)
                   for var in self.attributes) \
            or include_class and any(isinstance(var, ContinuousVariable)
                                     for var in self.class_vars)

    def get_conversion(self, source):
        """
        Return an instance of :class:`DomainConversion` for conversion from the
        given source domain to this domain. Domain conversions are cached to
        avoid unnecessary construction in the common case in which the domain
        is based on another domain, for instance, when the domain contains
        discretized variables from another domain.
        """
        # the method is thread-safe
        c = self.last_conversion
        if c and c.source is source:
            return c
        c = self.known_domains.get(source, None)
        if not c:
            c = DomainConversion(source, self)
            self.known_domains[source] = self.last_conversion = c
        return c

    def convert(self, inst):
        """
        Convert a data instance from another domain.
        :param inst: The data instance to be converted
        :return: The data instance in this domain
        """
        from .instance import Instance
        if isinstance(inst, Instance):
            if inst.domain == self:
                return inst._values, inst._metas
            c = self.get_conversion(inst.domain)
            values = [inst._values[i] if isinstance(i, int) else
                      (Unknown if not i else i(inst)) for i in c.variables]
            metas = [inst._values[i] if isinstance(i, int) else
                     (Unknown if not i else i(inst)) for i in c.metas]
        else:
            values = [var.to_val(val) for var, val in zip(self.variables, inst)]
            metas = [Unknown if var.is_primitive else None for var in self.metas]
        # Let np.array decide dtype for values
        return np.array(values), np.array(metas, dtype=object)
