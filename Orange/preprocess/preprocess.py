"""
Preprocess
----------

"""
import Orange.data
from . import impute, discretization, continuizer

__all__ = ["Preprocess", "Continuize", "Discretize"]


def is_continuous(var):
    return isinstance(var, Orange.data.ContinuousVariable)


def is_discrete(var):
    return isinstance(var, Orange.data.DiscreteVariable)


class Preprocess(object):
    def __call__(self, data):
        raise NotImplementedError("Subclasses need to implement __call__")


class Continuize(Preprocess):
    def __init__(self, zero_based=True,
                 multinomial_treatment=continuizer.DomainContinuizer.NValues,
                 normalize_continuous=
                 continuizer.DomainContinuizer.NormalizeBySD):
        self.zero_based = zero_based
        self.multinimial_treatment = multinomial_treatment

    def __call__(self, data):
        dc = continuizer.DomainContinuizer(
            zero_based=self.zero_based,
            multinomial_treatment=self.multinimial_treatment,
        )
        domain = dc(data)
        return data.from_table(domain, data)


class Discretize(Preprocess):
    def __init__(self, method=discretization.EqualFreq()):
        self.method = method

    def __call__(self, data):
        def transform(var):
            if is_continuous(var):
                newvar = self.method(data, var)
                if newvar is not None and len(newvar.values) >= 2:
                    return newvar
                else:
                    return None
            else:
                return var

        newattrs = [transform(var) for var in data.domain.attributes]
        newattrs = [var for var in newattrs if var is not None]
        domain = Orange.data.Domain(
            newattrs, data.domain.class_vars, data.domain.metas)

        return data.from_table(domain, data)


class Impute(Preprocess):
    def __init__(self, method=impute.Average()):
        self.method = method

    def __call__(self, data):
        newattrs = [self.method(data, var) for var in data.domain.attributes]
        domain = Orange.data.Domain(
            newattrs, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class PreprocessorList(object):
    def __init__(self, preprocessors):
        self.preprocessors = tuple(preprocessors)

    def __call__(self, data):
        for pp in self.preprocessors:
            data = pp(data)
        return data
