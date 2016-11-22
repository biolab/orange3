import scipy.sparse as sp

from Orange.data import Instance, Table, Domain


class Transformation:
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

    def __call__(self, data):
        """
        Return transformed column from the data by extracting the column view
        from the data and passing it to the `transform` method.
        """
        inst = isinstance(data, Instance)
        if inst:
            data = Table(data.domain, [data])
        if self.variable.is_primitive():
            domain = Domain([self.variable])
            data = Table.from_table(domain, data)
            col = data.X
        else:
            domain = Domain([], metas=[self.variable])
            data = Table.from_table(domain, data)
            col = data.metas
        if not sp.issparse(col):
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


class Identity(Transformation):
    """Return an untransformed value of `c`.
    """
    def transform(self, c):
        return c


class Indicator(Transformation):
    """
    Return an indicator value that equals 1 if the variable has the specified
    value and 0 otherwise.
    """
    def __init__(self, variable, value):
        """
        :param variable: The variable whose transformed value is returned.
        :type variable: int or str or :obj:`~Orange.data.Variable`

        :param value: The value to which the indicator refers
        :type value: int or float
        """
        super().__init__(variable)
        self.value = value

    def transform(self, c):
        return c == self.value


class Indicator1(Transformation):
    """
    Return an indicator value that equals 1 if the variable has the specified
    value and -1 otherwise.
    """
    def __init__(self, variable, value):
        """
        :param variable: The variable whose transformed value is returned.
        :type variable: int or str or :obj:`~Orange.data.Variable`

        :param value: The value to which the indicator refers
        :type value: int or float
        """
        super().__init__(variable)
        self.value = value

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
                raise ValueError('Non-zero offset in normalization '
                                 'of sparse data')
            return c * self.factor
        else:
            return (c - self.offset) * self.factor


class Lookup(Transformation):
    """
    Transform a discrete variable according to lookup table (`self.lookup`).
    """
    def __init__(self, variable, lookup_table):
        """
        :param variable: The variable whose transformed value is returned.
        :type variable: int or str or :obj:`~Orange.data.DiscreteVariable`
        :param lookup_table: transformations for each value of `self.variable`
        :type lookup_table: np.array or list or tuple
        """
        super().__init__(variable)
        self.lookup_table = lookup_table

    def transform(self, c):
        return self.lookup_table[c]
