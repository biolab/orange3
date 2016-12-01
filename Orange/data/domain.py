import weakref

from math import log
from collections import Iterable
from itertools import chain
from numbers import Integral

import numpy as np

from Orange.data import (
    Unknown, Variable, ContinuousVariable, DiscreteVariable, StringVariable
)

__all__ = ["DomainConversion", "Domain"]


class DomainConversion:
    """
    Indices and functions for conversion between domains.

    Every list contains indices (instances of int) of variables in the
    source domain, or the variable's compute_value function if the source
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
            else var.compute_value for var in destination.attributes]
        self.class_vars = [
            source.index(var) if var in source
            else var.compute_value for var in destination.class_vars]
        self.variables = self.attributes + self.class_vars
        self.metas = [
            source.index(var) if var in source
            else var.compute_value for var in destination.metas]


def filter_visible(feats):
    """
    Args:
        feats (iterable): Features to be filtered.

    Returns: A filtered tuple of features that are visible (i.e. not hidden).
    """
    return (f for f in feats if not f.attributes.get('hidden', False))


class Domain:
    def __init__(self, attributes, class_vars=None, metas=None, source=None):
        """
        Initialize a new domain descriptor. Arguments give the features and
        the class attribute(s). They can be described by descriptors (instances
        of :class:`Variable`), or by indices or names if the source domain is
        given.

        :param attributes: a list of attributes
        :type attributes: list of :class:`Variable`
        :param class_vars: target variable or a list of target variables
        :type class_vars: :class:`Variable` or list of :class:`Variable`
        :param metas: a list of meta attributes
        :type metas: list of :class:`Variable`
        :param source: the source domain for attributes
        :type source: Orange.data.Domain
        :return: a new domain
        :rtype: :class:`Domain`
        """

        if class_vars is None:
            class_vars = []
        elif isinstance(class_vars, (Variable, Integral, str)):
            class_vars = [class_vars]
        elif isinstance(class_vars, Iterable):
            class_vars = list(class_vars)

        if not isinstance(attributes, list):
            attributes = list(attributes)
        metas = list(metas) if metas else []

        # Replace str's and int's with descriptors if 'source' is given;
        # complain otherwise
        for lst in (attributes, class_vars, metas):
            for i, var in enumerate(lst):
                if not isinstance(var, Variable):
                    if source is not None and isinstance(var, (str, int)):
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
        self.class_var = \
            self.class_vars[0] if len(self.class_vars) == 1 else None
        if not all(var.is_primitive() for var in self._variables):
            raise TypeError("variables must be primitive")

        self._indices = dict(chain.from_iterable(
            ((var, idx), (var.name, idx), (idx, idx))
            for idx, var in enumerate(self._variables)))
        self._indices.update(chain.from_iterable(
            ((var, -1-idx), (var.name, -1-idx), (-1-idx, -1-idx))
            for idx, var in enumerate(self.metas)))

        self.anonymous = False
        self._known_domains = weakref.WeakKeyDictionary()
        self._last_conversion = None

        # Precompute hash, which is frequently used in domain conversions.
        self._hash = hash(self.attributes) ^ hash(self.class_vars) ^ hash(self.metas)

    # noinspection PyPep8Naming
    @classmethod
    def from_numpy(cls, X, Y=None, metas=None):
        """
        Create a domain corresponding to the given numpy arrays. This method
        is usually invoked from :meth:`Orange.data.Table.from_numpy`.

        All attributes are assumed to be continuous and are named
        "Feature <n>". Target variables are discrete if the only two values
        are 0 and 1; otherwise they are continuous. Discrete
        targets are named "Class <n>" and continuous are named "Target <n>".
        Domain is marked as :attr:`anonymous`, so data from any other domain of
        the same shape can be converted into this one and vice-versa.

        :param `numpy.ndarray` X: 2-dimensional array with data
        :param Y: 1- of 2- dimensional data for target
        :type Y: `numpy.ndarray` or None
        :param `numpy.ndarray` metas: meta attributes
        :type metas: `numpy.ndarray` or None
        :return: a new domain
        :rtype: :class:`Domain`
        """
        def get_places(max_index):
            return 0 if max_index == 1 else int(log(max_index, 10)) + 1

        def get_name(base, index, places):
            return base if not places \
                else "{} {:0{}}".format(base, index + 1, places)

        if X.ndim != 2:
            raise ValueError('X must be a 2-dimensional array')
        n_attrs = X.shape[1]
        places = get_places(n_attrs)
        attr_vars = [ContinuousVariable(name=get_name("Feature", a, places))
                     for a in range(n_attrs)]
        class_vars = []
        if Y is not None:
            if Y.ndim == 1:
                Y = Y.reshape(len(Y), 1)
            elif Y.ndim != 2:
                raise ValueError('Y has invalid shape')
            n_classes = Y.shape[1]
            places = get_places(n_classes)
            for i, values in enumerate(Y.T):
                if set(values) == {0, 1}:
                    name = get_name('Class', i, places)
                    values = ['v1', 'v2']
                    class_vars.append(DiscreteVariable(name, values))
                else:
                    name = get_name('Target', i + 1, places)
                    class_vars.append(ContinuousVariable(name))
        if metas is not None:
            n_metas = metas.shape[1]
            places = get_places(n_metas)
            meta_vars = [StringVariable(get_name("Meta", m, places))
                         for m in range(n_metas)]
        else:
            meta_vars = []

        domain = cls(attr_vars, class_vars, meta_vars)
        domain.anonymous = True
        return domain

    @property
    def variables(self):
        return self._variables

    @property
    def metas(self):
        return self._metas

    def __len__(self):
        """The number of variables (features and class attributes)."""
        return len(self._variables)

    def __getitem__(self, idx):
        """
        Return a variable descriptor from the given argument, which can be
        a descriptor, index or name. If `var` is a descriptor, the function
        returns this same object.

        :param idx: index, name or descriptor
        :type idx: int, str or :class:`Variable`
        :return: an instance of :class:`Variable` described by `var`
        :rtype: :class:`Variable`
        """
        if isinstance(idx, slice):
            return self._variables[idx]

        idx = self._indices[idx]
        if idx >= 0:
            return self.variables[idx]
        else:
            return self.metas[-1-idx]

    def __contains__(self, item):
        """
        Return `True` if the item (`str`, `int`, :class:`Variable`) is
        in the domain.
        """
        return item in self._indices

    def __iter__(self):
        """
        Return an iterator through variables (features and class attributes).
        """
        return iter(self._variables)

    def __str__(self):
        """
        Return a list-like string with the domain's features, class attributes
        and meta attributes.
        """
        s = "[" + ", ".join(attr.name for attr in self.attributes)
        if self.class_vars:
            s += " | " + ", ".join(cls.name for cls in self.class_vars)
        s += "]"
        if self._metas:
            s += " {" + ", ".join(meta.name for meta in self._metas) + "}"
        return s

    __repr__ = __str__

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_known_domains", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._known_domains = weakref.WeakKeyDictionary()

    def index(self, var):
        """
        Return the index of the given variable or meta attribute, represented
        with an instance of :class:`Variable`, `int` or `str`.
        """

        try:
            return self._indices[var]
        except KeyError:
            raise ValueError("'%s' is not in domain" % var)

    def has_discrete_attributes(self, include_class=False, include_metas=False):
        """
        Return `True` if domain has any discrete attributes. If
        `include_class` is set, the check includes the class attribute(s). If
        `include_metas` is set, the check includes the meta attributes.
        """
        vars = self.variables if include_class else self.attributes
        vars += self.metas if include_metas else ()
        return any(var.is_discrete for var in vars)

    def has_continuous_attributes(self, include_class=False, include_metas=False):
        """
        Return `True` if domain has any continuous attributes. If
        `include_class` is set, the check includes the class attribute(s). If
        `include_metas` is set, the check includes the meta attributes.
        """
        vars = self.variables if include_class else self.attributes
        vars += self.metas if include_metas else ()
        return any(var.is_continuous for var in vars)

    @property
    def has_continuous_class(self):
        return bool(self.class_var and self.class_var.is_continuous)

    @property
    def has_discrete_class(self):
        return bool(self.class_var and self.class_var.is_discrete)

    def get_conversion(self, source):
        """
        Return an instance of :class:`DomainConversion` for conversion from the
        given source domain to this domain. Domain conversions are cached to
        speed-up the conversion in the common case in which the domain
        is based on another domain, for instance, when the domain contains
        discretized variables from another domain.

        :param source: the source domain
        :type source: Orange.data.Domain
        """
        # the method is thread-safe
        c = self._last_conversion
        if c and c.source is source:
            return c
        c = self._known_domains.get(source, None)
        if not c:
            c = DomainConversion(source, self)
            self._known_domains[source] = self._last_conversion = c
        return c

    # noinspection PyProtectedMember
    def convert(self, inst):
        """
        Convert a data instance from another domain to this domain.

        :param inst: The data instance to be converted
        :return: The data instance in this domain
        """
        from .instance import Instance

        if isinstance(inst, Instance):
            if inst.domain == self:
                return inst._x, inst._y, inst._metas
            c = self.get_conversion(inst.domain)
            l = len(inst.domain.attributes)
            values = [(inst._x[i] if 0 <= i < l
                       else inst._y[i - l] if i >= l
                       else inst._metas[-i - 1])
                      if isinstance(i, int)
                      else (Unknown if not i else i(inst))
                      for i in c.variables]
            metas = [(inst._x[i] if 0 <= i < l
                      else inst._y[i - l] if i >= l
                      else inst._metas[-i - 1])
                     if isinstance(i, int)
                     else (Unknown if not i else i(inst))
                     for i in c.metas]
        else:
            nvars = len(self._variables)
            nmetas = len(self._metas)
            if len(inst) != nvars and len(inst) != nvars + nmetas:
                raise ValueError("invalid data length for domain")
            values = [var.to_val(val)
                      for var, val in zip(self._variables, inst)]
            if len(inst) == nvars + nmetas:
                metas = [var.to_val(val)
                         for var, val in zip(self._metas, inst[nvars:])]
            else:
                metas = [var.Unknown for var in self._metas]
        nattrs = len(self.attributes)
        # Let np.array decide dtype for values
        return np.array(values[:nattrs]), np.array(values[nattrs:]),\
               np.array(metas, dtype=object)

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
        if col_idx is ...:
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
                return (self[col_idx],
                        np.arange(start, end, stride))
        elif isinstance(col_idx, Iterable) and not isinstance(col_idx, str):
            attributes = [self[col] for col in col_idx]
            if attributes == self.attributes:
                return None, None
            return attributes, np.fromiter(
                (self.index(attr) for attr in attributes), int)
        elif isinstance(col_idx, Integral):
            attr = self[col_idx]
        else:
            attr = self[col_idx]
            col_idx = self.index(attr)
        return [attr], np.array([col_idx])

    def checksum(self):
        return hash(self)

    def __eq__(self, other):
        if not isinstance(other, Domain):
            return False

        return (self.attributes == other.attributes and
                self.class_vars == other.class_vars and
                self.metas == other.metas)

    def __hash__(self):
        return self._hash
