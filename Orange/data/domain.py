import logging
from math import log
from collections import Iterable
from itertools import chain
from numbers import Integral

import weakref
from .variable import *
import numpy as np

from Orange.util import deprecated


LOG = logging.getLogger()


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


class Domain:
    def __init__(self, attributes=(), class_vars=(), metas=(), source=None):
        """
        Initialize a new domain descriptor from features and class
        attribute(s), which are instances of :class:`Variable` or indices or
        names in the `source` domain if one is provided.

        Parameters
        ----------
        attributes : list of Variable
            A list of feature variables. Any non-primitive variables (i.e.
            those that can't be coerced to ``float``) are appended to
            `metas` instead.
        class_vars : Variable or list of Variable
            A list of class/target variables. Target variables must be
            primitive (instances of :class:`DiscreteVariable` or
            :class:`ContinuousVariable`).
        metas : list of Variables
            A list of (meta) variables that describe the data but aren't used
            for learning (e.g. :class:`StringVariable` holding individual
            names or example annotations).
        source : Domain
            Source domain from which to pull variables if they are specified
            as ``int`` or ``str`` in `attributes`, `class_vars`, or `metas`.

        Returns
        -------
        Domain
            A new domain.
        """

        if isinstance(class_vars, (Variable, int, str)):
            class_vars = [class_vars]
        try:
            class_vars = list(class_vars)
        except TypeError:
            raise TypeError('class_vars must be a list of Variable')
        try:
            attributes = list(attributes)
        except TypeError:
            raise TypeError('attributes must be a list of Variable')
        try:
            metas = list(metas)
        except TypeError:
            raise TypeError('metas must be a list of Variable')

        # Replace str's and int's with variables if 'source' is given
        if source is not None:
            def from_source(iterable):
                return [var if isinstance(var, Variable) else source[var]
                        for var in iterable]
            try:
                attributes = from_source(attributes)
                class_vars = from_source(class_vars)
                metas = from_source(metas)
            except (ValueError, TypeError):
                raise ValueError("Can't convert all indices "
                                 "(attrs={}, class_vars={}, metas={}) "
                                 "from source domain {}".format(
                                     attributes, class_vars, metas, source))

        if not all(isinstance(var, Variable)
                   for var in chain(attributes, class_vars, metas)):
            raise TypeError('Parameters attributes, class_vars, and metas '
                            'must be lists of Variable objects')

        # Put non-primitive Variables into metas instead
        append_metas = [var for var in chain(attributes, class_vars)
                        if not var.is_primitive]

        self.attributes = tuple(var for var in attributes if var.is_primitive)
        self.class_vars = tuple(var for var in class_vars if var.is_primitive)
        self.metas = tuple(chain(metas, append_metas))
        if append_metas:
            LOG.warning("Non-primitive variables '{}', passed as attributes "
                        "or class_vars, added to metas instead".format(
                            "', '".join(v.name for v in append_metas)))

        self._indices = dict(chain.from_iterable(
            ((var, i), (var.name, i)) for i, var in enumerate(self)))

        self._known_domains = weakref.WeakKeyDictionary()
        self._last_conversion = None

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return self.__class__(self.attributes, self.class_vars, self.metas)

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
        if Y is not None and Y.size:
            if Y.ndim == 1:
                Y = Y.reshape(len(Y), 1)
            elif Y.ndim != 2:
                raise ValueError('Y has invalid shape')
            n_classes = Y.shape[1]
            places = get_places(n_classes)
            for i, values in enumerate(Y.T):
                values = is_discrete_values(values)
                if values:
                    name = get_name('Class', i, places)
                    class_vars.append(DiscreteVariable(name, sorted(values)))
                else:
                    name = get_name('Target', i + 1, places)
                    class_vars.append(ContinuousVariable(name))
        if metas is not None and metas.size:
            n_metas = metas.shape[1]
            places = get_places(n_metas)
            meta_vars = [StringVariable(get_name("Meta", m, places))
                         for m in range(n_metas)]
        else:
            meta_vars = []

        domain = cls(attr_vars, class_vars, meta_vars)
        return domain

    @property
    def variables(self):
        """Return list of ```attributes` + `class_vars```."""
        return tuple(chain(self.attributes, self.class_vars))

    @property
    def class_var(self):
        return self.class_vars[0] if self.class_vars else None

    def __len__(self):
        """The number of variables in the domain."""
        return len(self.attributes) + len(self.class_vars) + len(self.metas)

    def __getitem__(self, idx):
        """
        Return the domain variable corresponding to given `idx`, which can
        be a numerical index or variable's name.

        In case `idx` represents multiple values (iterable, slice, ellipsis),
        a domain object containing those variables is returned.

        Parameters
        ----------
        idx : int or str or Variable or Iterable or slice or Ellipsis
            Index or name of variable to return. If ``Iterable``, a new
            domain is returned with matching attributes, class variables, and
            metas. If ``slice``, a new domain is returned with matching
            variables set as **attributes**. If ``Iterable``, it can contain
            integer indexes, names, or variables. If ``slice``, it returns
            the domain with its attributes matching that slice. If `Ellipsis`,
            self is returned.

        Returns
        -------
        var : Variable or Domain
            The (Domain of) variable(s) corresponding to index(es) `idx`.
        """
        try:
            if isinstance(idx, (int, np.integer)):
                if idx < 0:
                    idx += len(self)
                which = idx

            elif isinstance(idx, str):
                which = idx = self._indices[idx]

            elif isinstance(idx, Iterable):
                vars = [self[val] for val in idx]
                X = tuple(v for v in vars if v in self.attributes)
                Y = tuple(v for v in vars if v in self.class_vars)
                M = tuple(v for v in vars if v in self.metas)
                return self.__class__(X, Y, M)

            elif idx is Ellipsis:
                return self

            elif isinstance(idx, slice):
                idx = range(*idx.indices(len(self)))
                if idx == range(len(self)):
                    return self.__class__(self)
                return self.__class__(self[i] for i in idx)

            elif isinstance(idx, Variable):
                if idx not in self:
                    raise KeyError
                return idx

            else:
                raise TypeError("Can't get variable by index type '{}'. Names "
                                "and integers are supported.".format(
                                    idx.__class__))
        except KeyError:
            raise ValueError("Variable '{}' is not in domain".format(idx))

        # Find which group this index belongs to; equivalent to
        # tuple(self)[idx], but much faster for large domains
        which -= len(self.attributes)
        if which < 0:
            return self.attributes[idx]
        idx, which = which, which - len(self.class_vars)
        if which < 0:
            return self.class_vars[idx]
        idx, which = which, which - len(self.metas)
        assert which < 0
        return self.metas[idx]

    def __contains__(self, item):
        """
        Return `True` if the item (`str`, `int`, :class:`Variable`) is
        in the domain.
        """
        return item in self._indices

    def __iter__(self):
        """Return an iterator through all variables in the domain."""
        return chain(self.attributes, self.class_vars, self.metas)

    def __str__(self):
        """
        Return a list-like string with the domain's features, class attributes
        and meta attributes.
        """
        s = "[" + ", ".join(attr.name for attr in self.attributes)
        if self.class_vars:
            s += " | " + ", ".join(cls.name for cls in self.class_vars)
        s += "]"
        if self.metas:
            s += " {" + ", ".join(meta.name for meta in self.metas) + "}"
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
        Return the index of the given variable (represented as
        `int`, `str`, or instance of :class:`Variable`) in the domain.
        """
        if isinstance(var, (int, np.integer)):
            return int(var)
        try:
            return self._indices[var]
        except KeyError:
            raise ValueError("Variable '{}' is not in domain".format(var))

    @property
    def has_discrete(self):
        """Return `True` if domain has any discrete attributes."""
        return any(var.is_discrete for var in self.attributes)

    @property
    def has_continuous(self):
        """Return `True` if domain has any continuous attributes."""
        return any(var.is_continuous for var in self.attributes)

    @property
    def has_continuous_class(self):
        """
        Returrn `True` if the first class variable of domain is continuous.
        """
        return bool(self.class_var and self.class_var.is_continuous)

    @property
    def has_discrete_class(self):
        """
        Returrn `True` if the first class variable of domain is discrete.
        """
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
            lc = len(inst.domain.class_vars)
            values = [(inst._x[i] if 0 <= i < l
                       else inst._y[i - l] if l <= i < l + lc
                       else inst._metas[i - l - lc])
                      if isinstance(i, int)
                      else (Unknown if not i else i(inst))
                      for i in c.variables]
            metas = [(inst._x[i] if 0 <= i < l
                      else inst._y[i - l] if l <= i < l + lc
                      else inst._metas[i - l - lc])
                     if isinstance(i, int)
                     else (Unknown if not i else i(inst))
                     for i in c.metas]
        else:
            nvars = len(self.variables)
            nmetas = len(self.metas)
            if len(inst) != nvars and len(inst) != nvars + nmetas:
                raise ValueError("invalid data length for domain")
            values = [var.to_val(val)
                      for var, val in zip(self.variables, inst)]
            if len(inst) == nvars + nmetas:
                metas = [var.to_val(val)
                         for var, val in zip(self.metas, inst[nvars:])]
            else:
                metas = [var.Unknown for var in self.metas]
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

    @deprecated
    def checksum(self):
        return hash(self)

    def __eq__(self, other):
        return (isinstance(other, Domain) and
                self.attributes == other.attributes and
                self.class_vars == other.class_vars and
                self.metas == other.metas)

    def __hash__(self):
        return hash(self.attributes) ^ hash(self.class_vars) ^ hash(self.metas)
