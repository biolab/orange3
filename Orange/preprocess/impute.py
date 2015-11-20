import numpy

import Orange.data
from Orange.statistics import distribution, basic_stats
from .transformation import Transformation, Lookup

__all__ = ["ReplaceUnknowns", "Average"]


class ReplaceUnknowns(Transformation):
    """
    A column transformation which replaces unknown values with a fixed `value`.

    Parameters
    ----------
    variable : Orange.data.Variable
        The target variable for imputation.
    value : int or float
        The value with which to replace the unknown values
    """
    def __init__(self, variable, value=0):
        super().__init__(variable)
        self.value = value

    def transform(self, c):
        return numpy.where(numpy.isnan(c), self.value, c)


class Average:
    def __call__(self, data, variable, value=None):
        variable = data.domain[variable]
        if value is None:
            if variable.is_continuous:
                stats = basic_stats.BasicStats(data, variable)
                value = stats.mean
            elif variable.is_discrete:
                dist = distribution.get_distribution(data, variable)
                value = dist.modus()
            else:
                raise TypeError("Variable must be continuous or discrete")

        a = variable.copy(compute_value=ReplaceUnknowns(variable, value))
        a.to_sql = ImputeSql(variable, value)
        return a


class ImputeSql:
    def __init__(self, var, default):
        self.var = var
        self.default = default

    def __call__(self):
        return 'coalesce(%s, %s)' % (self.var.to_sql(), str(self.default))


class Default:
    def __init__(self, default=0):
        self.default = default

    def __call__(self, data, variable, *, default=None):
        variable = data.domain[variable]
        default = default if default is not None else self.default
        return variable.copy(
            compute_value=ReplaceUnknowns(variable, default))


class ReplaceUnknownsModel:
    """
    Replace unknown values with predicted values using a `Orange.base.Model`

    Parameters
    ----------
    variable : Orange.data.Variable
        The target variable for the imputation.
    model : Orange.base.Model
        A fitted model predicting `variable`.
    """
    def __init__(self, variable, model):
        assert model.domain.class_var == variable
        self.variable = variable
        self.model = model

    def __call__(self, data):
        if isinstance(data, Orange.data.Instance):
            column = numpy.array([float(data[self.variable])])
        else:
            column = numpy.array(data.get_column_view(self.variable)[0],
                                 copy=True)

        mask = numpy.isnan(column)
        if not numpy.any(mask):
            return column

        if isinstance(data, Orange.data.Instance):
            predicted = self.model(data)
        else:
            predicted = self.model(data[mask])
        column[mask] = predicted
        return column


class Model:
    def __init__(self, learner):
        self.learner = learner

    def __call__(self, data, variable):
        variable = data.domain[variable]
        domain = domain_with_class_var(data.domain, variable)
        data = data.from_table(domain, data)
        model = self.learner(data)
        assert model.domain.class_var == variable
        return variable.copy(
            compute_value=ReplaceUnknownsModel(variable, model))


def domain_with_class_var(domain, class_var):
    """
    Return a domain with class_var as output domain.class_var.

    If class_var is in the input domain's attributes it is removed from the
    output's domain.attributes.
    """
    if domain.class_var is class_var:
        return domain
    elif class_var in domain.attributes:
        attrs = [var for var in domain.attributes
                 if var is not class_var]
    else:
        attrs = domain.attributes
    return Orange.data.Domain(attrs, class_var)


class IsDefined(Transformation):
    def transform(self, c):
        return ~numpy.isnan(c)


class Lookup(Lookup):
    def __init__(self, variable, lookup_table, unknown=None):
        super().__init__(variable, lookup_table)
        self.unknown = unknown

    def transform(self, column):
        if self.unknown is None:
            unknown = numpy.nan
        else:
            unknown = self.unknown

        mask = numpy.isnan(column)
        column_valid = numpy.where(mask, 0, column)
        values = self.lookup_table[numpy.array(column_valid, dtype=int)]
        return numpy.where(mask, unknown, values)


class AsValue:
    def __call__(self, data, variable):
        variable = data.domain[variable]
        if variable.is_discrete:
            fmt = "{var.name}"
            value = "N/A"
            var = Orange.data.DiscreteVariable(
                fmt.format(var=variable),
                values=variable.values + [value],
                base_value=variable.base_value,
                compute_value=Lookup(
                    variable,
                    numpy.arange(len(variable.values), dtype=int),
                    unknown=len(variable.values))
                )
            return var

        elif variable.is_continuous:
            fmt = "{var.name}_def"
            indicator_var = Orange.data.DiscreteVariable(
                fmt.format(var=variable),
                values=("undef", "def"),
                compute_value=IsDefined(variable))
            stats = basic_stats.BasicStats(data, variable)
            return (variable.copy(compute_value=ReplaceUnknowns(variable,
                                                                stats.mean)),
                    indicator_var)
        else:
            raise TypeError(type(variable))


class ReplaceUnknownsRandom(Transformation):
    """
    A column transformation replacing unknowns with values drawn randomly from
    an empirical distribution.

    Parameters
    ----------
    variable : Orange.data.Variable
        The target variable for imputation.
    distribution : Orange.statistics.distribution.Distribution
        The corresponding sampling distribution
    """
    def __init__(self, variable, distribution):
        assert distribution.size > 0
        assert distribution.variable == variable
        super().__init__(variable)
        self.distribution = distribution

        if variable.is_discrete:
            counts = numpy.array(distribution)
        elif variable.is_continuous:
            counts = numpy.array(distribution)[1, :]
        else:
            raise TypeError("Only discrete and continuous "
                            "variables are supported")
        csum = numpy.sum(counts)
        if csum > 0:
            self.sample_prob = counts / csum
        else:
            self.sample_prob = numpy.ones_like(counts) / len(counts)

    def transform(self, c):
        c = numpy.array(c, copy=True)
        nanindices = numpy.flatnonzero(numpy.isnan(c))

        if self.variable.is_discrete:
            sample = numpy.random.choice(
                len(self.variable.values), size=len(nanindices),
                replace=True, p=self.sample_prob)
        else:
            sample = numpy.random.choice(
                numpy.asarray(self.distribution)[0, :], size=len(nanindices),
                replace=True, p=self.sample_prob)

        c[nanindices] = sample
        return c


class Random:
    def __call__(self, data, variable):
        variable = data.domain[variable]
        dist = distribution.get_distribution(data, variable)
        # A distribution is invalid if a continuous variable's column does not
        # contain any known values or if a discrete variable's .values == []
        isinvalid = dist.size == 0
        if isinvalid and variable.is_discrete:
            assert len(variable.values) == 0
            raise ValueError("'{}' has no values".format(variable))
        elif isinvalid and variable.is_continuous:
            raise ValueError("'{}' has an unknown distribution"
                             .format(variable))

        if variable.is_discrete and numpy.sum(dist) == 0:
            dist += 1 / len(dist)
        elif variable.is_continuous and numpy.sum(dist[1, :]) == 0:
            dist[1, :] += 1 / dist.shape[1]
        return variable.copy(
            compute_value=ReplaceUnknownsRandom(variable, dist))
