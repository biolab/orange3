import numpy as np
import scipy.sparse as sp

import Orange.data
from Orange.statistics import distribution, basic_stats
from Orange.util import Reprable
from .transformation import Transformation, Lookup

__all__ = ["ReplaceUnknowns", "Average", "DoNotImpute", "DropInstances",
           "Model", "AsValue", "Random", "Default", "FixedValueByType"]


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
        if sp.issparse(c):
            c.data = np.where(np.isnan(c.data), self.value, c.data)
            return c
        else:
            return np.where(np.isnan(c), self.value, c)

    def __eq__(self, other):
        return super().__eq__(other) and self.value == other.value

    def __hash__(self):
        return hash((type(self), self.variable, float(self.value)))


class BaseImputeMethod(Reprable):
    name = ""
    short_name = ""
    description = ""
    format = "{var.name} -> {self.short_name}"
    columns_only = False

    def __call__(self, data, variable):
        """ Imputes table along variable column.

        Args:
            data (Table): A table to impute.
            variable (Variable): Variable for completing missing values.

        Returns:
            A new Variable instance with completed missing values or
            a array mask of rows to drop out.
        """
        raise NotImplementedError

    def format_variable(self, var):
        return self.format.format(var=var, self=self)

    def __str__(self):
        return self.name

    def copy(self):
        return self

    @classmethod
    def supports_variable(cls, variable):
        return True


class DoNotImpute(BaseImputeMethod):
    name = "Don't impute"
    short_name = "leave"
    description = ""

    def __call__(self, data, variable):
        return variable


class DropInstances(BaseImputeMethod):
    name = "Remove instances with unknown values"
    short_name = "drop"
    description = ""

    def __call__(self, data, variable):
        col, _ = data.get_column_view(variable)
        return np.isnan(col)


class Average(BaseImputeMethod):
    name = "Average/Most frequent"
    short_name = "average"
    description = "Replace with average/mode of the column"

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
                raise TypeError("Variable must be numeric or categorical.")

        a = variable.copy(compute_value=ReplaceUnknowns(variable, value))
        a.to_sql = ImputeSql(variable, value)
        return a

    @staticmethod
    def supports_variable(variable):
        return variable.is_primitive()


class ImputeSql(Reprable):
    def __init__(self, var, default):
        self.var = var
        self.default = default

    def __call__(self):
        return 'coalesce(%s, %s)' % (self.var.to_sql(), str(self.default))


class Default(BaseImputeMethod):
    name = "Fixed value"
    short_name = "value"
    description = ""
    columns_only = True
    format = '{var} -> {self.default}'

    def __init__(self, default=0):
        self.default = default

    def __call__(self, data, variable, *, default=None):
        variable = data.domain[variable]
        default = default if default is not None else self.default
        return variable.copy(compute_value=ReplaceUnknowns(variable, default))

    def copy(self):
        return Default(self.default)


class FixedValueByType(BaseImputeMethod):
    name = "Fixed value"
    short_name = "Fixed Value"
    format = "{var.name}"

    def __init__(self,
                 default_discrete=np.nan, default_continuous=np.nan,
                 default_string=None, default_time=np.nan):
        # If you change the order of args or in dict, also fix method copy
        self.defaults = {
            Orange.data.DiscreteVariable: default_discrete,
            Orange.data.ContinuousVariable: default_continuous,
            Orange.data.StringVariable: default_string,
            Orange.data.TimeVariable: default_time
        }

    def __call__(self, data, variable, *, default=None):
        variable = data.domain[variable]
        if default is None:
            default = self.defaults[type(variable)]
        return variable.copy(compute_value=ReplaceUnknowns(variable, default))

    def copy(self):
        return FixedValueByType(*self.defaults.values())


class ReplaceUnknownsModel(Reprable):
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
            data = Orange.data.Table.from_list(data.domain, [data])
        domain = data.domain
        column = np.array(data.get_column_view(self.variable)[0], copy=True)

        mask = np.isnan(column)
        if not np.any(mask):
            return column

        if domain.class_vars:
            # cannot have class var in domain (due to backmappers in model)
            data = data.transform(
                Orange.data.Domain(domain.attributes, None, domain.metas)
            )
        predicted = self.model(data[mask])
        column[mask] = predicted
        return column


class Model(BaseImputeMethod):
    _name = "Model-based imputer"
    short_name = "model"
    description = ""
    format = BaseImputeMethod.format + " ({self.learner.name})"
    @property
    def name(self):
        return "{} ({})".format(self._name, getattr(self.learner, 'name', ''))

    def __init__(self, learner):
        self.learner = learner

    def __call__(self, data, variable):
        variable = data.domain[variable]
        domain = domain_with_class_var(data.domain, variable)

        if self.learner.check_learner_adequacy(domain):
            data = data.transform(domain)
            model = self.learner(data)
            assert model.domain.class_var == variable
            return variable.copy(
                compute_value=ReplaceUnknownsModel(variable, model))
        else:
            raise ValueError("`{}` doesn't support domain type"
                             .format(self.learner.name))

    def copy(self):
        return Model(self.learner)

    def supports_variable(self, variable):
        domain = Orange.data.Domain([], class_vars=variable)
        return self.learner.check_learner_adequacy(domain)


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
        if sp.issparse(c):
            c = c.toarray()
        return ~np.isnan(c)


class AsValue(BaseImputeMethod):
    name = "As a distinct value"
    short_name = "new value"
    description = ""

    def __call__(self, data, variable):
        variable = data.domain[variable]
        if variable.is_discrete:
            fmt = "{var.name}"
            value = "N/A"
            var = Orange.data.DiscreteVariable(
                fmt.format(var=variable),
                values=variable.values + (value, ),
                compute_value=Lookup(
                    variable,
                    np.arange(len(variable.values), dtype=int),
                    unknown=len(variable.values)),
                sparse=variable.sparse,
                )
            return var

        elif variable.is_continuous:
            fmt = "{var.name}_def"
            indicator_var = Orange.data.DiscreteVariable(
                fmt.format(var=variable),
                values=("undef", "def"),
                compute_value=IsDefined(variable),
                sparse=variable.sparse,
            )
            stats = basic_stats.BasicStats(data, variable)
            return (variable.copy(compute_value=ReplaceUnknowns(variable,
                                                                stats.mean)),
                    indicator_var)
        else:
            raise TypeError(type(variable))

    @staticmethod
    def supports_variable(variable):
        return variable.is_primitive()

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
            counts = np.array(distribution)
        elif variable.is_continuous:
            counts = np.array(distribution)[1, :]
        else:
            raise TypeError("Only categorical and numeric "
                            "variables are supported.")
        csum = np.sum(counts)
        if csum > 0:
            self.sample_prob = counts / csum
        else:
            self.sample_prob = np.ones_like(counts) / len(counts)

    def transform(self, c):
        if not sp.issparse(c):
            c = np.array(c, copy=True)
        else:
            c = c.toarray().ravel()
        nanindices = np.flatnonzero(np.isnan(c))

        if self.variable.is_discrete:
            sample = np.random.choice(
                len(self.variable.values), size=len(nanindices),
                replace=True, p=self.sample_prob)
        else:
            sample = np.random.choice(
                np.asarray(self.distribution)[0, :], size=len(nanindices),
                replace=True, p=self.sample_prob)

        c[nanindices] = sample
        return c

    def __eq__(self, other):
        return super().__eq__(other) and self.distribution == other.distribution

    def __hash__(self):
        return hash((type(self), self.variable, self.distribution))


class Random(BaseImputeMethod):
    name = "Random values"
    short_name = "random"
    description = "Replace with a random value"

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

        if variable.is_discrete and np.sum(dist) == 0:
            dist += 1 / len(dist)
        elif variable.is_continuous and np.sum(dist[1, :]) == 0:
            dist[1, :] += 1 / dist.shape[1]
        return variable.copy(
            compute_value=ReplaceUnknownsRandom(variable, dist))

    @staticmethod
    def supports_variable(variable):
        return variable.is_primitive()
