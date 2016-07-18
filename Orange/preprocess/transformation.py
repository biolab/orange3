import numpy as np

from Orange.data import Table


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
        self._last_domain = None

    def __call__(self, data):
        """
        Return transformed column from the data by extracting the column view
        from the data and passing it to the `transform` method.
        """
        if self._last_domain != data.domain:
            try:
                self.attr_index = data.domain.index(self.variable)
            except ValueError:
                if self.variable.compute_value is None:
                    raise ValueError("{} is not in domain".
                                     format(self.variable.name))
                self.attr_index = None
            self._last_domain = data.domain
        if self.attr_index is None:
            data = self.variable.compute_value(data)
        else:
            data = data[data.domain[self.attr_index]]
        transformed = self.transform(data)
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


class Ordinalize(Transformation):
    """
    Used for discrete variables; return the value as it appears in e.g. t.X.
    """
    def transform(self, c):
        return c.apply(self.variable.to_val).astype(int)


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
        if self.variable.is_discrete:
            c = c.apply(self.variable.to_val)
        return (c == self.value) * 1


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
        if self.variable.is_discrete:
            c = c.apply(self.variable.to_val)
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
        # we need to map the values to their numerical representation,
        # only then can we do numerical computations
        # on c.apply(...), the categorical persists, but the categories
        # automatically change (so we don't need to worry about that)
        # we then need to use a number dtype (because we got indices)
        # to support mathematical operations
        return (c.apply(self.variable.to_val).astype(float) - self.offset) * self.factor


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
        return c.apply(lambda val: self.lookup_table[val])
