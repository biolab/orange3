"""
Preprocess
----------

"""
import numpy as np
import sklearn.preprocessing as skl_preprocessing

import Orange.data
from . import impute, discretize
from ..misc.enum import Enum

__all__ = ["Continuize", "Discretize", "Impute", "SklImpute"]


def is_continuous(var):
    return isinstance(var, Orange.data.ContinuousVariable)


def is_discrete(var):
    return isinstance(var, Orange.data.DiscreteVariable)


class Preprocess(object):
    def __new__(cls, data=None, *args, **kwargs):
        self = super().__new__(cls)
        if isinstance(data, Orange.data.Storage):
            self.__init__(*args, **kwargs)
            return self(data)
        else:
            return self

    def __call__(self, data):
        raise NotImplementedError("Subclasses need to implement __call__")


class Continuize(Preprocess):
    MultinomialTreatment = Enum(
        "Indicators", "FirstAsBase", "FrequentAsBase",
        "Remove", "RemoveMultinomial", "ReportError", "AsOrdinal",
        "AsNormalizedOrdinal", "Leave", "NormalizeBySpan",
        "NormalizeBySD"
    )

    (Indicators, FirstAsBase, FrequentAsBase, Remove, RemoveMultinomial,
     ReportError, AsOrdinal, AsNormalizedOrdinal, Leave,
     NormalizeBySpan, NormalizeBySD) = MultinomialTreatment

    def __init__(self, zero_based=True, multinomial_treatment=Indicators,
                 normalize_continuous=NormalizeBySD):
        self.zero_based = zero_based
        self.multinomial_treatment = multinomial_treatment
        self.normalize_continuous = normalize_continuous

    def __call__(self, data):
        from . import continuize
        continuizer = continuize.DomainContinuizer(
            zero_based=self.zero_based,
            multinomial_treatment=self.multinomial_treatment,
            normalize_continuous=self.normalize_continuous)
        domain = continuizer(data)
        return data.from_table(domain, data)


class Discretize(Preprocess):
    def __init__(self, method=None):
        """
        Construct a discretizer.

        :param method: discretization method
        :type method: Orange.preprocess.discretiza.Discretization
        """
        self.method = method

    def __call__(self, data):
        """
        Compute and apply discretization of the given data. Returns a new
        data table.

        :param data: data
        :type data: Orange.data.Table
        :return: Orange.data.Table
        """
        def transform(var):
            if is_continuous(var):
                newvar = method(data, var)
                if newvar is not None and len(newvar.values) >= 2:
                    return newvar
                else:
                    return None
            else:
                return var

        method = self.method or discretize.EqualFreq()
        newattrs = [transform(var) for var in data.domain.attributes]
        newattrs = [var for var in newattrs if var is not None]
        domain = Orange.data.Domain(
            newattrs, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class Impute(Preprocess):
    def __init__(self, method=None):
        self.method = method

    def __call__(self, data):
        method = self.method or impute.Average()
        newattrs = [method(data, var) for var in data.domain.attributes]
        domain = Orange.data.Domain(
            newattrs, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class SklImpute(Preprocess):
    def __init__(self, strategy='mean', force=True):
        self.strategy = strategy
        self.force = force

    def __call__(self, data):
        if not self.force and not np.isnan(data.X).any():
            return data
        self.imputer = skl_preprocessing.Imputer(strategy=self.strategy)
        X = self.imputer.fit_transform(data.X)
        features = [impute.Average()(data, var, value) for var, value in
                    zip(data.domain.attributes, self.imputer.statistics_)]
        domain = Orange.data.Domain(features, data.domain.class_vars,
                                    data.domain.metas)
        return Orange.data.Table(domain, X, data.Y, data.metas)


class PreprocessorList(object):
    def __init__(self, preprocessors):
        self.preprocessors = tuple(preprocessors)

    def __call__(self, data):
        for pp in self.preprocessors:
            data = pp(data)
        return data
