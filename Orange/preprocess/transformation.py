from typing import TYPE_CHECKING, Mapping, Optional

import numpy as np
import scipy.sparse as sp
from pandas import isna

from Orange.data import Instance, Table, Domain, Variable
from Orange.misc.collections import DictMissingConst
from Orange.util import Reprable, nan_eq, nan_hash_stand, frompyfunc

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


class Transformation(Reprable):
    """
    Base class for simple transformations of individual variables. Derived
    classes are used in continuization, imputation, discretization...
    """
    def __init__(self, variable):
        """
        :param variable: The variable whose transformed value is returned.
        :type variable: int or str or :obj:`~Orange.data.Variable`
        """
        self.variable = variable
        self._create_cached_target_domain()

    def _create_cached_target_domain(self):
        """ If the same domain is used everytime this allows better caching of
        domain transformations in from_table"""
        if self.variable is not None:
            if self.variable.is_primitive():
                self._target_domain = Domain([self.variable])
            else:
                self._target_domain = Domain([], metas=[self.variable])

    def __getstate__(self):
        # Do not pickle the cached domain; rather recreate it after unpickling
        state = self.__dict__.copy()
        state.pop("_target_domain")
        return state

    def __setstate__(self, state):
        # Ensure that cached target domain is created after unpickling.
        # This solves the problem of unpickling old pickled models.
        self.__dict__.update(state)
        self._create_cached_target_domain()

    def __call__(self, data):
        """
        Return transformed column from the data by extracting the column view
        from the data and passing it to the `transform` method.
        """
        inst = isinstance(data, Instance)
        if inst:
            # TODO deprecate this
            data = Table.from_list(data.domain, [data])
        data = data.transform(self._target_domain)
        if self.variable.is_primitive():
            col = data.X
        else:
            col = data.metas
        if not sp.issparse(col) and col.ndim > 1:
            col = col.squeeze(axis=1)
        transformed = self.transform(col)
        if inst:
            transformed = transformed[0]
        return transformed

    def transform(self, c):
        """
        Return the transformed value of the argument `c`, which can be a number
        of a vector view.
        """
        raise NotImplementedError(
            "ColumnTransformations must implement method 'transform'.")

    def __eq__(self, other):
        return type(other) is type(self) and self.variable == other.variable

    def __hash__(self):
        return hash((type(self), self.variable))


class Identity(Transformation):
    """Return an untransformed value of `c`.
    """
    InheritEq = True

    def transform(self, c):
        return c

    def __eq__(self, other):  # pylint: disable=useless-parent-delegation
        return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()


# pylint: disable=abstract-method
class _Indicator(Transformation):
    def __init__(self, variable, value):
        """
        :param variable: The variable whose transformed value is returned.
        :type variable: int or str or :obj:`~Orange.data.Variable`

        :param value: The value to which the indicator refers
        :type value: int or float
        """
        super().__init__(variable)
        self.value = value

    def __eq__(self, other):
        return super().__eq__(other) and self.value == other.value

    def __hash__(self):
        return hash((type(self), self.variable, self.value))

    @staticmethod
    def _nan_fixed(c, transformed):
        if np.isscalar(c):
            if c != c:  # pylint: disable=comparison-with-itself
                transformed = np.nan
            else:
                transformed = float(transformed)
        else:
            transformed = transformed.astype(float)
            transformed[np.isnan(c)] = np.nan
        return transformed


class Indicator(_Indicator):
    """
    Return an indicator value that equals 1 if the variable has the specified
    value and 0 otherwise.
    """

    InheritEq = True

    def transform(self, c):
        if sp.issparse(c):
            if self.value != 0:
                # If value is nonzero, the matrix will become sparser:
                # we transform the data and remove zeros
                transformed = c.copy()
                transformed.data = self.transform(c.data)
                transformed.eliminate_zeros()
                return transformed
            else:
                # Otherwise, it becomes dense anyway (or it wasn't really sparse
                # before), so we just convert it to sparse before transforming
                c = c.toarray().ravel()
        return self._nan_fixed(c, c == self.value)

    def __eq__(self, other):  # pylint: disable=useless-parent-delegation
        return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()


class Indicator1(_Indicator):
    """
    Return an indicator value that equals 1 if the variable has the specified
    value and -1 otherwise.
    """

    InheritEq = True

    def transform(self, column):
        # The result of this is always dense
        if sp.issparse(column):
            column = column.toarray().ravel()
        return self._nan_fixed(column, (column == self.value) * 2 - 1)


class Normalizer(Transformation):
    """
    Return a normalized variable; for the given `value`, the transformed value
    if `(value - self.offset) * self.factor`.
    """

    def __init__(self, variable, offset, factor):
        """
        :param variable: The variable whose transformed value is returned.
        :type variable: int or str or :obj:`~Orange.data.Variable`
        :param offset:
        :type offset: float
        :param factor:
        :type factor: float
        """
        super().__init__(variable)
        self.offset = offset
        self.factor = factor

    def transform(self, c):
        if sp.issparse(c):
            if self.offset != 0:
                raise ValueError('Normalization does not work for sparse data.')
            return c * self.factor
        else:
            return (c - self.offset) * self.factor

    def __eq__(self, other):
        return super().__eq__(other) \
               and self.offset == other.offset and self.factor == other.factor

    def __hash__(self):
        return hash((type(self), self.variable, self.offset, self.factor))


class Lookup(Transformation):
    """
    Transform a discrete variable according to lookup table (`self.lookup`).
    """
    def __init__(self, variable, lookup_table, unknown=np.nan):
        """
        :param variable: The variable whose transformed value is returned.
        :type variable: int or str or :obj:`~Orange.data.DiscreteVariable`
        :param lookup_table: transformations for each value of `self.variable`
        :type lookup_table: np.array
        :param unknown: The value to be used as unknown value.
        :type unknown: float or int
        """
        super().__init__(variable)
        self.lookup_table = lookup_table
        self.unknown = unknown

    def transform(self, column):
        # Densify DiscreteVariable values coming from sparse datasets.
        if sp.issparse(column):
            column = column.toarray().ravel()
        mask = np.isnan(column)
        column = column.astype(int)
        column[mask] = 0
        values = self.lookup_table[column]
        return np.where(mask, self.unknown, values)

    def __eq__(self, other):
        return super().__eq__(other) \
               and np.allclose(self.lookup_table, other.lookup_table,
                               equal_nan=True) \
               and np.allclose(self.unknown, other.unknown, equal_nan=True)

    def __hash__(self):
        return hash(
            (
                type(self),
                self.variable,
                # nan value does not have constant hash in Python3.10
                # to avoid different hashes for the same array change to None
                # issue: https://bugs.python.org/issue43475#msg388508
                tuple(None if isna(x) else x for x in self.lookup_table),
                nan_hash_stand(self.unknown),
            )
        )


class MappingTransform(Transformation):
    """
    Map values via a dictionary lookup.

    Parameters
    ----------
    variable: Variable
    mapping: Mapping
        The mapping (for the non NA values).
    dtype: Optional[DTypeLike]
        The optional target dtype.
    unknown: Any
        The constant with whitch to replace unknown values in input.
    """
    def __init__(
            self,
            variable: Variable,
            mapping: Mapping,
            dtype: Optional['DTypeLike'] = None,
            unknown=np.nan,
    ) -> None:
        super().__init__(variable)
        if any(nan_eq(k, np.nan) for k in mapping.keys()):  # ill-defined mapping
            raise ValueError("'nan' value in mapping.keys()")
        self.mapping = mapping
        self.dtype = dtype
        self.unknown = unknown
        self._mapper = self._make_dict_mapper(
            DictMissingConst(unknown, mapping), dtype
        )

    @staticmethod
    def _make_dict_mapper(mapping, dtype):
        return frompyfunc(mapping.__getitem__, 1, 1, dtype)

    def transform(self, c):
        return self._mapper(c)

    def __reduce_ex__(self, protocol):
        return type(self), (self.variable, self.mapping, self.dtype, self.unknown)

    def __eq__(self, other):
        return super().__eq__(other) \
               and nan_mapping_eq(self.mapping, other.mapping) \
               and self.dtype == other.dtype \
               and nan_eq(self.unknown, other.unknown)

    def __hash__(self):
        return hash((type(self), self.variable, nan_mapping_hash(self.mapping),
                     self.dtype, nan_hash_stand(self.unknown)))


def nan_mapping_hash(a: Mapping) -> int:
    return hash(tuple((k, nan_hash_stand(v)) for k, v in a.items()))


def nan_mapping_eq(a: Mapping, b: Mapping) -> bool:
    if len(a) != len(b):
        return False
    try:
        for k, va in a.items():
            vb = b[k]
            if not nan_eq(va, vb):
                return False
    except LookupError:
        return False
    return True
