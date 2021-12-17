import numpy as np
import scipy.sparse as sp

from Orange.data import Instance, Table, Domain
from Orange.util import Reprable


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

        if self.variable is not None:
            if self.variable.is_primitive():
                self.need_domain = Domain([self.variable])
            else:
                self.need_domain = Domain([], metas=[self.variable])

    def __call__(self, data):
        """
        Return transformed column from the data by extracting the column view
        from the data and passing it to the `transform` method.
        """
        inst = isinstance(data, Instance)
        if inst:
            data = Table.from_list(data.domain, [data])
        data = data.transform(self.need_domain)
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
    def transform(self, c):
        return c


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


class Indicator(_Indicator):
    """
    Return an indicator value that equals 1 if the variable has the specified
    value and 0 otherwise.
    """
    def transform(self, c):
        return c == self.value


class Indicator1(_Indicator):
    """
    Return an indicator value that equals 1 if the variable has the specified
    value and -1 otherwise.
    """
    def transform(self, c):
        return (c == self.value) * 2 - 1


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
        return hash((type(self), self.variable,
                     tuple(self.lookup_table), self.unknown))
