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

        The source domain. The destination is not stored since destination
        domain is the one which contains the instance of DomainConversion.

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
        Compute the conversion indices from the given `source` to `destination`
        """
        self.source = source
        self.attributes = [
        source.index(var) if var in source
        else var.get_value_from for var in destination.attributes]
        self.class_vars = [
        source.index(var) if var in source
        else var.get_value_from for var in destination.class_vars]
        self.variables = self.attributes + self.class_vars
        self.metas = [
        source.index(var) if var in source
        else var.get_value_from for var in destination.metas]


class Domain:
    def __init__(self, attributes, class_vars=None, metas=None, source=None):
        """
        Initialize a new domain descriptor. Arguments give the features and
        the class attribute(s). They can be described by descriptors (instances
        of :class:`Variable`), or by indices or names if the source domain is
        given.

        :param attributes: a list of attributes
        :param class_vars: the target variable or a list of target variables
        :param metas: a list of meta attributes
        :param source: the source domain for attributes
        :return: a new instance of :class:`domain`
        """

        if class_vars is None:
            class_vars = []
        elif isinstance(class_vars, (Variable, int, str)):
            class_vars = [class_vars]
        elif isinstance(class_vars, Iterable):
            class_vars = list(class_vars)

        metas = list(metas) if metas else []

        # Replace str's and int's with descriptors if 'source' is given;
        # complain otherwise
        if source:
            if not isinstance(attributes, list):
                attributes = list(attributes)
            if metas and not isinstance(metas, list):
                metas = list(metas)
        if metas is None:
            metas = []
        for lst in (attributes, class_vars, metas):
            for i, var in enumerate(lst):
                if not isinstance(var, Variable):
                    if source and isinstance(var, (str, int)):
                        lst[i] = source[var]
                    else:
                        raise TypeError(
                            "descriptors must be instances of Variable, "
                            "not '%s'" % type(var).__name__)

        # Store everything
        self.attributes = tuple(attributes)
        self.class_vars = tuple(class_vars)
        self._variables = self.attributes + self.class_vars
        self._metas = tuple(metas)
        self.class_var =\
        self.class_vars[0] if len(self.class_vars) == 1 else None
        if not all(var.is_primitive() for var in self._variables):
            raise TypeError("variables must be primitive")

        self.indices = {var.name: idx
                        for idx, var in enumerate(self._variables)}
        self.indices.update((var.name, -1 - idx)
            for idx, var in enumerate(metas))

        self.anonymous = False
        self.known_domains = weakref.WeakKeyDictionary()
        self.last_conversion = None


    @classmethod
    def from_numpy(cls, X, Y=None, metas=None):
        """
        Create a domain corresponding to the given numpy arrays.

        All attributes are assumed to be continuous and are named
        "Feature <n>". Target variables are discrete if the all values are
        whole numbers between 0 and 19; otherwise they are continuous. Discrete
        classes are named "Class <n>" and continuous are named "Target <n>".
        Domain is marked as anonymous, so data from any other domain of the
        same shape can be converted into this one and vice-versa.

        :param X: attributes
        :param Y: class variables
        :param metas: meta attributes
        :return: a new domain
        :rtype: Orange.data.Domain
        """
        attr_vars = [ContinuousVariable(name="Feature %i" % (a + 1))
                     for a in range(X.shape[1])]
        class_vars = []
        if Y is not None:
            for i, class_ in enumerate(Y.T):
                mn, mx = np.min(class_), np.max(class_)
                if 0 <= mn and mx <= 20:
                    values = np.unique(class_)
                    if all(int(x) == x and 0 <= x <= 19 for x in values):
                        mx = int(mx)
                        places = 1 + (mx >= 10)
                        values = ["v%*i" % (places, i + 1)
                                  for i in range(mx + 1)]
                        name = "Class %i" % (i + 1)
                        class_vars.append(DiscreteVariable(name, values))
                        continue
                class_vars.append(
                    ContinuousVariable(name="Target %i" % (i + 1)))
        if metas is not None:
            meta_vars = [StringVariable(name="Meta %i" % m)
                         for m in range(metas.shape[1])]
        else:
            meta_vars = []

        domain = cls(attr_vars, class_vars, meta_vars)
        domain.anonymous = True
        return domain


    def var_from_domain(self, var, check_included=False, no_index=False):
        """
        Return a variable descriptor from the given argument, which can be
        a descriptor, index or name. If `var` is a descriptor, the function
        returns it

        :param var: an instance of :class:`Variable`, `int` or `str`
        :param check_included: if `var` is an instance of :class:`Variable`,
            this flags tells whether to check that the domain contains this
            variable
        :param no_index: if `True`, `var` must not be an `int`
        :return: an instance of :class:`Variable` described by `var`
        """
        if isinstance(var, str):
            if not var in self.indices:
                raise IndexError("Variable '%s' is not in the domain", var)
            idx = self.indices[var]
            return self._variables[idx] if idx >= 0 else self._metas[-1 - idx]

        if not no_index and isinstance(var, int):
            return self._variables[var] if var >= 0 else self._metas[-1 - var]

        if isinstance(var, Variable):
            if check_included:
                for each in chain(self.variables, self.metas):
                    if each is var:
                        return var
                raise IndexError(
                    "Variable '%s' is not in the domain", var.name)
            else:
                return var

        raise TypeError(
            "Expected str, int or Variable, got '%s'" % type(var).__name__)


    @property
    def variables(self):
        return self._variables


    @property
    def metas(self):
        return self._metas


    def __len__(self):
        """Return the number of variables (features and class attributes)"""
        return len(self._variables)


    def __getitem__(self, index):
        """
        Same as var_from_domain. Index can be a slice, an int, str or
        instance of :class:`Variable`. Slices apply to variables, not meta
        attributes.
        """
        if isinstance(index, slice):
            return self._variables[index]
        else:
            return self.var_from_domain(index, True)


    def __contains__(self, item):
        """
        Return `True` if the item (`str`, `int`, :class:`Variable`) is
        in the domain.
        """
        if isinstance(item, str):
            return item in self.indices
        if isinstance(item, Variable) and not item.name in self.indices:
            return False
            # ... but not the opposite!
            # It may just be a variable with the same name
        try:
            self.var_from_domain(item, True)
            return True
        except IndexError:
            return False


    def __iter__(self):
        """
        Return an iterator through variables (features and class attributes)
        """
        return iter(self._variables)


    def __str__(self):
        """
        Return a list-like string with the domain's features, class attributes
        and meta attributes
        """
        s = "[" + ", ".join(attr.name for attr in self.attributes)
        if self.class_vars:
            s += " | " + ", ".join(cls.name for cls in self.class_vars)
        s += "]"
        if self._metas:
            s += " {" + ", ".join(meta.name for meta in self._metas) + "}"
        return s


    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("known_domains")
        return state


    def index(self, var):
        """
        Return the index of the given variable or meta attribute, represented
        with an instance of :class:`Variable`, `int` or `str`.
        """
        if isinstance(var, str):
            idx = self.indices.get(var, None)
            if idx is None:
                raise ValueError("'%s' is not in domain" % var)
            else:
                return idx
        if isinstance(var, Variable):
            if var in self._variables:
                return self._variables.index(var)
            if var in self._metas:
                return -1 - self._metas.index(var)
            raise ValueError("'%s' is not in domain" % var.name)
        if isinstance(var, int):
            if -len(self._metas) <= var < len(self._variables):
                return var
            raise ValueError("there is no variable with index '%i'" % var)
        raise TypeError("Expected str, int or Variable, got '%s'" %
                        type(var).__name__)


    def has_discrete_attributes(self, include_class=False):
        """
        Return `True` if domain has any discrete attributes.

        :param include_class: if `True` (default is `False`), the check
            includes the class attribute(s)
        """
        if not include_class:
            return any(isinstance(var, DiscreteVariable)
                for var in self.attributes)
        else:
            return any(isinstance(var, DiscreteVariable)
                for var in self.variables)


    def has_continuous_attributes(self, include_class=False):
        """
        Return `True` if domain has any continuous attributes.

        :param include_class: if `True` (default is `False`), the check
            includes the class attribute(s)
        """
        if not include_class:
            return any(isinstance(var, ContinuousVariable)
                for var in self.attributes)
        else:
            return any(isinstance(var, ContinuousVariable)
                for var in self.variables)


    def get_conversion(self, source):
        """
        Return an instance of :class:`DomainConversion` for conversion from the
        given source domain to this domain. Domain conversions are cached to
        speed-up the conversion in the common case in which the domain
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
            values = [var.to_val(val)
                      for var, val in zip(self._variables, inst)]
            metas = [Unknown if var.is_primitive else None
                     for var in self._metas]
            # Let np.array decide dtype for values
        return np.array(values), np.array(metas, dtype=object)

    def select_columns(self, col_idx):
        attributes, col_indices = self._compute_col_indices(col_idx)
        if attributes is not None:
            n_attrs = len(self.attributes)
            r_attrs = [attributes[i]
                       for i, col in enumerate(col_indices)
                       if 0 <= col < n_attrs]
            r_classes = [attributes[i]
                         for i, col in enumerate(col_indices)
                         if col >= n_attrs]
            r_metas = [attributes[i]
                       for i, col in enumerate(col_indices) if col < 0]
            return Domain(r_attrs, r_classes, r_metas)
        else:
            return self

    def _compute_col_indices(self, col_idx):
        if col_idx is Ellipsis:
            return None, None
        if isinstance(col_idx, np.ndarray) and col_idx.dtype == bool:
            return ([attr for attr, c in zip(self, col_idx) if c],
                    np.nonzero(col_idx))
        elif isinstance(col_idx, slice):
            s = len(self.variables)
            start, end, stride = col_idx.indices(s)
            if col_idx.indices(s) == (0, s, 1):
                return None, None
            else:
                return (self.variables[col_idx],
                        np.arange(start, end, stride))
        elif isinstance(col_idx, Iterable) and not isinstance(col_idx, str):
            attributes = [self[col] for col in col_idx]
            if attributes == self.attributes:
                return None, None
            return attributes, np.fromiter(
                (self.index(attr) for attr in attributes), int)
        elif isinstance(col_idx, int):
            attr = self[col_idx]
        else:
            attr = self[col_idx]
            col_idx = self.index(attr)
        return [attr], np.array([col_idx])
