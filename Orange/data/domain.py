from math import log
from collections import Iterable
from itertools import chain
from numbers import Integral

import weakref

from .variable import *
import numpy as np


class DomainConversion:
    """Indices and functions for conversion between domains.

    Every list contains indices (instances of int) of variables in the
    source domain, or the variable's compute_value function if the source
    domain does not contain the variable.

    Attributes
    ----------
    source : int or Callable
        The source domain. The destination is not stored since destination
        domain is the one which contains the instance of DomainConversion.
    attributes : int or Callable
        Indices for attribute values.
    class_vars : int or Callable
        Indices for class variables.
    variables : int or Callable
        Indices for attributes and class variables (:obj:`attributes`+:obj:`class_vars`).
    metas : int or Callable
        Indices for meta attributes.
    """
    def __init__(self, source, destination):
        """Compute the conversion indices from the given source to the destination domain.

        Parameters
        ----------
        source : Domain
            The source domain.
        destination : Domain
            The destination domain.
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
    """Filter only visible (not hidden) features.

    Parameters
    ----------
    feats : Iterable
        Features to be filtered.

    Returns
    -------
    generator
        A filtered generator of features that are visible (i.e. not hidden).
    """
    return (f for f in feats if not f.attributes.get('hidden', False))


class Domain:
    """A container for Variables that classifies them into roles.

    The domain is intended immutable by design.

    A note on indexing:
     - Indices [0, len(attributes)) are indices of attributes.
     - Indices [len(attributes), len(attributes) + len(class_vars))
       are indices of class vars.
     - Indices (-(len(metas) + 1), -1] are indices of meta attributes.
       Note that these are reversed.
    """

    def __init__(self, attributes, class_vars=None, metas=None, source=None):
        """Initialize a new domain descriptor.

        Arguments give the features and the class attribute(s).
        They can be described by descriptors (instances Variable),
        or by indices or names if the source domain is given.

        Parameters
        ----------
        attributes : list[str or Variable or int]
            A list of attributes.
        class_vars : list[str or Variable or int], optional, default None
            A list of class variables.
        metas : list[str or Variable or int], optional, default None
            A list of meta attributes.
        source : Domain, optional, default None
            The source domain for the attributes.
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
        """Return the variables (attributes and class vars) of this domain."""
        return self._variables

    @property
    def metas(self):
        """Return the meta attributes of this domain."""
        return self._metas

    def __len__(self):
        """The number of variables (features and class attributes)."""
        return len(self._variables)

    def __getitem__(self, idx):
        """Get a variable descriptor.

        Parameters
        ----------
        idx : str or Variable or int
            A variable name, variable or domain index of the requested Variable.

        Returns
        -------
        Variable
            The requested variable.
        """
        if isinstance(idx, slice):
            return self._variables[idx]

        idx = self._indices[idx]
        if idx >= 0:
            return self.variables[idx]
        else:
            return self.metas[-1-idx]

    def __contains__(self, item):
        """Check whether this item is in the domain.

        Parameters
        ----------
        item : str, Variable, int
            The variable name, descriptor or index.

        Returns
        -------
        bool
            Whether the item exists in the domain.
        """
        return item in self._indices

    def __iter__(self):
        """Return an iterator through variables (features and class attributes)."""
        # TODO: make domain iterate over the whole domain - including metas
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
        """Get the index of the given variable.

        Parameters
        ----------
        var : str or Variable or int

        Returns
        -------
        int
            The index of the variable.
        """
        try:
            return self._indices[var]
        except KeyError:
            raise ValueError("'%s' is not in domain" % var)

    def has_discrete_attributes(self, include_class=False):
        """Determine whether the domain has any discrete attributes.

        Parameters
        ----------
        include_class : bool, optional, default False
            Whether to include class attributes in the check.

        Returns
        -------
        bool
            Return True if the domain has any discrete attributes.
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
        """Determine whether the domain has a continuous class."""
        return bool(self.class_var and self.class_var.is_continuous)

    @property
    def has_discrete_class(self):
        """Determine whether the domain has a discrete class."""
        return bool(self.class_var and self.class_var.is_discrete)

    def get_conversion(self, source):
        """Get a conversion from the source to this domain.

        Domain conversions are cached to speed-up the conversion in the common case
        in which the domain is based on another domain, for instance,
        when the domain contains discretized variables from another domain.

        Parameters
        ----------
        source : Domain
            The source domain.

        Returns
        -------
        DomainConversion
            A new conversion object.
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

    def convert(self, inst):
        """Convert a data instance from another domain to this domain.

        Parameters
        ----------
        inst : SeriesBase
            The data instance to be converted.

        Returns
        -------
        (np.ndarray, np.ndarray, np.ndarray)
            A tuple of converted X, Y and metas.
        """

        nvars = len(self._variables)
        nmetas = len(self._metas)
        if len(inst) != nvars and len(inst) != nvars + nmetas:
            raise ValueError("invalid data length for domain")
        # SQL table workaround
        values = []
        for var, val in zip(self._variables, inst):
            try:
                values.append(var.to_val(val))
            except:
                values.append(val)
        if len(inst) == nvars + nmetas:
            metas = [var.to_val(val)
                     for var, val in zip(self._metas, inst[nvars:])]
        else:
            metas = [var.Unknown for var in self._metas]
        nattrs = len(self.attributes)
        # Let np.array decide dtype for values
        return np.array(values[:nattrs]), \
               np.array(values[nattrs:]), \
               np.array(metas, dtype=object)

    def select_columns(self, col_idx):
        """Select specific columns from the domain.

        Parameters
        ----------
        col_idx
            The column indices.

        Returns
        -------
        Domain
            A domain with only the selected columns.
        """

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
        """Determine whether a column can be considered as a DiscreteVariable.

        Parameters
        ----------
        column : SeriesBase or np.array
            The column to check.
        discrete_max_values : int, optional, default 3
            The maximum number of discrete values to allow.
            3 = 2 + nan.
        Returns
        -------
        set or bool
            A set of unique values for the column if the variable can
            be considered discrete, False or none otherwise.
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
        """Infer a variable type and the column role for the given column.

        Parameters
        ----------
        column : pd.Series
            The column to infer the role for.
        force_role : str, optional, default None
            The column role to force (skip inference).

        Returns
        -------
        (T <= Variable, str)
            A tuple of the inferred type and the inferred role.
            Type is one of {'x', 'y', 'meta'}.
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
        """Infer a name for the given column type and role.

        Takes into account existing variables and an existing name, if passed.

        Parameters
        ----------
        column_type : T <= Variable
            The column type.
        column_role : str
            The column role. One of {'x', 'y', 'meta'}.
        force_name : str, optional, default None
            An existing column name to force. If this is None or a number,
            a new name is generated, otherwise the existing name is returned.
        existing_attrs : list
            A list of existing attribute names.
        existing_classes : list
            A list of existing class names.
        existing_metas : list
            A list of existing meta names.

        Returns
        -------
        str
            The inferred name for the variable.
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
