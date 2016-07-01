from math import log
from collections import Iterable
from itertools import chain
from numbers import Integral

import weakref

from util import deprecated
from .variable import *
import numpy as np
from pandas import DataFrame


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

    def has_discrete_attributes(self, include_class=False):
        """
        Return `True` if domain has any discrete attributes. If `include_class`
                is set, the check includes the class attribute(s).
        """
        if not include_class:
            return any(var.is_discrete for var in self.attributes)
        else:
            return any(var.is_discrete for var in self.variables)

    def has_continuous_attributes(self, include_class=False):
        """
        Return `True` if domain has any continuous attributes. If
        `include_class` is set, the check includes the class attribute(s).
        """
        if not include_class:
            return any(var.is_continuous for var in self.attributes)
        else:
            return any(var.is_continuous for var in self.variables)

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

        if isinstance(inst, Instance):

            # TODO: transform this to work with Tables

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
        return hash(self.attributes) ^ hash(self.class_vars) ^ hash(self.metas)

    @classmethod
    def _is_discrete_column(cls, column, discrete_max_values=3):
        """
        Determine whether a column can be considered as a DiscreteVariable.
        The default value for discrete_max_value is 3 for 2 + nan.
        """
        if not len(column):
            return None
        # If the first few values are, or can be converted to, floats,
        # the type is numeric
        try:
            isinstance(next(iter(column)), Number) or \
            [float(v) for _, v in zip(range(min(3, len(column))), column)]
        except ValueError:
            is_numeric = False
            max_values = int(round(len(column) ** .7))
        else:
            is_numeric = True
            max_values = discrete_max_values

        # If more than max values => not discrete
        unique = column.unique()
        if len(unique) > max_values:
            return False

        # Strip NaN from unique
        unique = {i for i in unique
                  if (not i in Variable.MISSING_VALUES and
                      not (isinstance(i, Number) and np.isnan(i)))}

        # All NaNs => indeterminate
        if not unique:
            return False

        # Strings with |values| < max_unique
        if not is_numeric:
            return unique

        # Handle numbers
        try:
            unique_float = set(map(float, unique))
        except ValueError:
            # Converting all the values to floats resulted in an error.
            # Since the values have enough unique values, they are probably
            # string values and discrete.
            return unique

        # If only values are {0, 1} or {1, 2} (or a subset of those sets) => discrete
        return (not (unique_float - {0, 1}) or
                not (unique_float - {1, 2})) and unique

    @classmethod
    def infer_type_role(cls, column, force_role=None):
        """
        Infer a variable type and the column role for the given column (pd.Series).
        Return a tuple of (type, role), where type is one of the subclasses of Variable
        and role is one of {'x', 'y', 'meta'}.
        """
        # if the column looks like it has times, this has precedence
        # also allow for pandas to parse dates before us (such as when reading a file)
        if np.issubdtype(column.dtype, np.datetime64) \
                or TimeVariable.column_looks_like_time(column):
            return TimeVariable, (force_role or 'x')
        # if there are at most 3 different values of any kind, they are discrete,
        # but there have to be at least 6 values in total
        elif Domain._is_discrete_column(column):
            return DiscreteVariable, (force_role or 'x')
        # all other all-number columns are continuous features
        elif np.issubdtype(column.dtype, np.number):
            return ContinuousVariable, (force_role or 'x')
        # all others (including those we can't determine) are string metas
        else:
            return StringVariable, (force_role or 'meta')

    @classmethod
    def infer_name(cls, column_type, column_role, force_name=None,
                   existing_attrs=(), existing_classes=(), existing_metas=()):
        """
        Infers a name for the given column type and role, taking into account
        existing variables and an existing name, if passed.

        if force_name is None or a number, a new name is generated, otherwise
        the existing name is returned.

        existing_X is a list of existing variables of the type.
        """
        if not force_name or isinstance(force_name, Integral) \
                or (isinstance(force_name, str) and force_name.isdigit()):
            # choose a new name, we don't want numbers
            if column_role == 'x':
                return "Feature {}".format(len(existing_attrs) + 1)
            elif column_role == 'y' and column_type is ContinuousVariable:
                # TimeVariable is ContinuousVariable
                return "Target {}".format(len(existing_classes) + 1)
            elif column_role == 'y' and column_type is DiscreteVariable:
                return "Class {}".format(len(existing_classes) + 1)
            else:
                return "Meta {}".format(len(existing_metas) + 1)
        else:
            return force_name
