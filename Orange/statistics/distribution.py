from collections.abc import Iterable
from numbers import Real
import zlib

import numpy as np

from Orange import data


def _get_variable(dat, variable, expected_type=None, expected_name=""):
    """Get the variable instance from data."""
    failed = False
    if isinstance(variable, data.Variable):
        datvar = getattr(dat, "variable", None)
        if datvar is not None and datvar is not variable:
            raise ValueError("variable does not match the variable in the data")
    elif hasattr(dat, "domain"):
        variable = dat.domain[variable]
    elif hasattr(dat, "variable"):
        variable = dat.variable
    else:
        failed = True
    if failed or (expected_type is not None and not isinstance(variable, expected_type)):
        if isinstance(variable, data.Variable):
            raise ValueError("expected %s variable not %s" % (expected_name, variable))
        else:
            raise ValueError("expected %s, not '%s'" % (
                expected_type.__name__, type(variable).__name__))
    return variable


class Distribution(np.ndarray):
    def __array_finalize__(self, obj):
        # defined in derived classes,
        # pylint: disable=attribute-defined-outside-init
        """See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html"""
        if obj is None:
            return
        self.variable = getattr(obj, 'variable', None)
        self.unknowns = getattr(obj, 'unknowns', 0)

    def __reduce__(self):
        state = super().__reduce__()
        newstate = state[2] + (self.variable, self.unknowns)
        return state[0], state[1], newstate

    def __setstate__(self, state):
        # defined in derived classes,
        # pylint: disable=attribute-defined-outside-init
        super().__setstate__(state[:-2])
        self.variable, self.unknowns = state[-2:]

    def __eq__(self, other):
        return (
            np.array_equal(self, other) and
            (not hasattr(other, "unknowns") or self.unknowns == other.unknowns)
        )

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return zlib.adler32(self) ^ hash(self.unknowns)

    def sample(self, size=None, replace=True):
        """Get a random sample from the distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]]
        replace : bool

        Returns
        -------
        Union[float, data.Value, np.ndarray]

        """
        raise NotImplementedError

    def normalize(self):
        """Normalize the distribution to a probability distribution."""
        raise NotImplementedError

    def min(self):
        """Get the smallest value for the distribution.

        If the variable is not ordinal, return None.

        """
        raise NotImplementedError

    def max(self):
        """Get the largest value for the distribution.

        If the variable is not ordinal, return None.

        """
        raise NotImplementedError


class Discrete(Distribution):
    def __new__(cls, dat, variable=None, unknowns=None):
        if isinstance(dat, data.Storage):
            if unknowns is not None:
                raise TypeError("incompatible arguments (data storage and 'unknowns'")
            return cls.from_data(dat, variable)

        if variable is not None:
            variable = _get_variable(dat, variable)
            n = len(variable.values)
        else:
            n = len(dat)

        self = super().__new__(cls, n)
        self.variable = variable
        if dat is None:
            self[:] = 0
            self.unknowns = unknowns or 0
        else:
            self[:] = dat
            self.unknowns = unknowns if unknowns is not None else getattr(dat, "unknowns", 0)
        return self

    @classmethod
    def from_data(cls, data, variable):
        variable = _get_variable(data, variable)
        try:
            dist, unknowns = data._compute_distributions([variable])[0]
            self = super().__new__(cls, len(dist))
            self[:] = dist
            self.unknowns = unknowns
        except NotImplementedError:
            self = super().__new__(cls, len(variable.values))
            self[:] = np.zeros(len(variable.values))
            self.unknowns = 0
            if data.has_weights():
                for inst, w in zip(data, data.W):
                    val = inst[variable]
                    if not np.isnan(val):
                        self[int(val)] += w
                    else:
                        self.unknowns += w
            else:
                for inst in data:
                    val = inst[variable]
                    if val == val:
                        self[int(val)] += 1
                    else:
                        self.unknowns += 1
        self.variable = variable
        return self

    @property
    def array_with_unknowns(self):
        """
        This property returns a distribution array with unknowns added
        at the end

        Returns
        -------
        np.array
            Array with appended unknowns at the end of the row.
        """
        return np.append(np.array(self), self.unknowns)

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.variable.to_val(index)
        return super().__getitem__(index)

    def __setitem__(self, index, value):
        if isinstance(index, str):
            index = self.variable.to_val(index)
        super().__setitem__(index, value)

    def __add__(self, other):
        s = super().__add__(other)
        s.unknowns = self.unknowns + getattr(other, "unknowns", 0)
        return s

    def __iadd__(self, other):
        super().__iadd__(other)
        self.unknowns += getattr(other, "unknowns", 0)
        return self

    def __sub__(self, other):
        s = super().__sub__(other)
        s.unknowns = self.unknowns - getattr(other, "unknowns", 0)
        return s

    def __isub__(self, other):
        super().__isub__(other)
        self.unknowns -= getattr(other, "unknowns", 0)
        return self

    def __mul__(self, other):
        s = super().__mul__(other)
        if isinstance(other, Real):
            s.unknowns = self.unknowns / other
        return s

    def __imul__(self, other):
        super().__imul__(other)
        if isinstance(other, Real):
            self.unknowns *= other
        return self

    def __div__(self, other):
        s = super().__mul__(other)
        if isinstance(other, Real):
            s.unknowns = self.unknowns / other
        return s

    def __idiv__(self, other):
        super().__imul__(other)
        if isinstance(other, Real):
            self.unknowns /= other
        return self

    def normalize(self):
        t = np.sum(self)
        if t > 1e-6:
            self[:] /= t
            self.unknowns /= t
        elif self.shape[0]:
            self[:] = 1 / self.shape[0]

    def modus(self):
        val = np.argmax(self)
        return data.Value(self.variable, val) if self.variable is not None else val

    def sample(self, size=None, replace=True):
        value_indices = np.random.choice(range(len(self)), size, replace, self.normalize())
        if isinstance(value_indices, Iterable):
            to_value = np.vectorize(lambda idx: data.Value(self.variable, idx))
            return to_value(value_indices)
        return data.Value(self.variable, value_indices)

    def min(self):
        return None

    def max(self):
        return None

    def sum(self, *args, **kwargs):
        res = super().sum(*args, **kwargs)
        res.unknowns = self.unknowns
        return res


class Continuous(Distribution):
    def __new__(cls, dat, variable=None, unknowns=None):
        if isinstance(dat, data.Storage):
            if unknowns is not None:
                raise TypeError("incompatible arguments (data storage and 'unknowns'")
            return cls.from_data(variable, dat)
        if isinstance(dat, int):
            self = super().__new__(cls, (2, dat))
            self[:] = 0
            self.unknowns = unknowns or 0
        else:
            if not isinstance(dat, np.ndarray):
                dat = np.asarray(dat)
            self = super().__new__(cls, dat.shape)
            self[:] = dat
            self.unknowns = (unknowns if unknowns is not None else getattr(dat, "unknowns", 0))
        self.variable = variable
        return self

    @classmethod
    def from_data(cls, variable, data):
        variable = _get_variable(data, variable)
        try:
            dist, unknowns = data._compute_distributions([variable])[0]
        except NotImplementedError:
            col = data[:, variable]
            dtype = col.dtype
            if data.has_weights():
                if not "float" in dtype.name and "float" in col.dtype.name:
                    dtype = col.dtype.name
                dist = np.empty((2, len(col)), dtype=dtype)
                dist[0, :] = col
                dist[1, :] = data.W
            else:
                dist = np.ones((2, len(col)), dtype=dtype)
                dist[0, :] = col
            dist.sort(axis=0)
            dist = np.array(_orange.valuecount(dist))
            unknowns = len(col) - dist.shape[1]

        self = super().__new__(cls, dist.shape)
        self[:] = dist
        self.unknowns = unknowns
        self.variable = variable
        return self

    def normalize(self):
        t = np.sum(self[1, :])
        if t > 1e-6:
            self[1, :] /= t
            self.unknowns /= t
        elif self.shape[1]:
            self[1, :] = 1 / self.shape[1]

    def modus(self):
        val = np.argmax(self[1, :])
        return self[0, val]

    def min(self):
        return self[0, 0]

    def max(self):
        return self[0, -1]

    def sample(self, size=None, replace=True):
        normalized = Continuous(self, self.variable, self.unknowns)
        normalized.normalize()
        return np.random.choice(self[0, :], size, replace, normalized[1, :])

    def mean(self):
        return np.average(np.asarray(self[0]), weights=np.asarray(self[1]))

    def variance(self):
        mean = self.mean()
        return np.dot((self[0] - mean) ** 2, self[1]) / np.sum(self[1])

    def standard_deviation(self):
        return np.sqrt(self.variance())


def class_distribution(data):
    """Get the distribution of the class variable(s)."""
    if data.domain.class_var:
        return get_distribution(data, data.domain.class_var)
    elif data.domain.class_vars:
        return [get_distribution(data, cls) for cls in data.domain.class_vars]
    else:
        raise ValueError("domain has no class attribute")


def get_distribution(dat, variable, unknowns=None):
    """Get the distribution of the given variable."""
    variable = _get_variable(dat, variable)
    if variable.is_discrete:
        return Discrete(dat, variable, unknowns)
    elif variable.is_continuous:
        return Continuous(dat, variable, unknowns)
    else:
        raise TypeError("cannot compute distribution of '%s'" % type(variable).__name__)


def get_distributions(dat, skipDiscrete=False, skipContinuous=False):
    """Get the distributions of all variables in the data."""
    vars = dat.domain.variables
    if skipDiscrete:
        if skipContinuous:
            return []
        columns = [i for i, var in enumerate(vars) if var.is_continuous]
    elif skipContinuous:
        columns = [i for i, var in enumerate(vars) if var.is_discrete]
    else:
        columns = None
    try:
        dist_unks = dat._compute_distributions(columns)
        if columns is None:
            columns = np.arange(len(vars))
        distributions = []
        for col, (dist, unks) in zip(columns, dist_unks):
            distributions.append(get_distribution(dist, vars[col], unks))
    except NotImplementedError:
        if columns is None:
            columns = np.arange(len(vars))
        distributions = [get_distribution(dat, i) for i in columns]
    return distributions


def get_distributions_for_columns(data, columns):
    """Compute the distributions for columns.

    Parameters
    ----------
    data : data.Table
        List of column indices into the `data.domain` (indices can be
        :class:`int` or instances of `Orange.data.Variable`)

    """
    domain = data.domain
    # Normailze the columns to int indices
    columns = [col if isinstance(col, int) else domain.index(col) for col in columns]
    try:
        # Try the optimized code path (query the table|storage directly).
        dist_unks = data._compute_distributions(columns)
    except NotImplementedError:
        # Use default slow(er) implementation.
        return [get_distribution(data, i) for i in columns]
    else:
        # dist_unkn is a list of (values, unknowns)
        return [get_distribution(dist, domain[col], unknown)
                for col, (dist, unknown) in zip(columns, dist_unks)]
