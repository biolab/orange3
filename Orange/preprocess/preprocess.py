"""
Preprocess
----------

"""
import numpy as np
import sklearn.preprocessing as skl_preprocessing
import bottlechest

import Orange.data
from . import impute, discretize
from ..misc.enum import Enum

__all__ = ["Continuize", "Discretize", "Impute", "SklImpute"]


def is_continuous(var):
    return isinstance(var, Orange.data.ContinuousVariable)


def is_discrete(var):
    return isinstance(var, Orange.data.DiscreteVariable)


class Preprocess(object):
    """
    A generic preprocessor class. All preprocessors need to inherit this
    class. Preprocessors can be instantiated without the data set to return
    data preprocessor, or can be given a data set to return the preprocessed
    data.

    Parameters
    ----------
    data : a data table (default=None)
        An optional data set to be preprocessed.
    """

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
    """
    Construct a discretizer, a preprocessor for discretization of
    continuous fatures.

    Parameters
    ----------
    method : discretization method (default: Orange.preprocess.discretize.Discretization)
    """

    def __init__(self, method=None):
        self.method = method

    def __call__(self, data):
        """
        Compute and apply discretization of the given data. Returns a new
        data table.

        Parameters
        ----------
        data : Orange.data.Table
            A data table to be discretized.
        """

        def transform(var):
            if is_continuous(var):
                new_var = method(data, var)
                if new_var is not None and len(new_var.values) >= 2:
                    return new_var
                else:
                    return None
            else:
                return var

        method = self.method or discretize.EqualFreq()
        attributes = [transform(var) for var in data.domain.attributes]
        attributes = [var for var in attributes if var is not None]
        domain = Orange.data.Domain(
            attributes, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class Impute(Preprocess):
    """
    Construct a imputer, a preprocessor for imputation of missing values in
    the data table.

    Parameters
    ----------
    method : imputation method (default: Orange.preprocess.impute.Average())
    """

    def __init__(self, method=Orange.preprocess.impute.Average()):
        self.method = method

    def __call__(self, data):
        """
        Apply an imputation method to the given data set. Returns a new
        data table with missing values replaced by their imputations.

        Parameters
        ----------
        data : Orange.data.Table
            An input data table.
        """

        method = self.method or impute.Average()
        newattrs = [method(data, var) for var in data.domain.attributes]
        domain = Orange.data.Domain(
            newattrs, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class SklImpute(Preprocess):
    __wraps__ = skl_preprocessing.Imputer

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


class RemoveConstant(Preprocess):
    """
    Construct a preprocessor that removes features with constant values
    from the data set.
    """
    def __call__(self, data):
        """
        Remove columns with constant values from the data set and return
        the resulting data table.

        Parameters
        ----------
        data : an input data set
        """

        oks = bottlechest.nanmin(data.X, axis=0) != \
              bottlechest.nanmax(data.X, axis=0)
        atts = [data.domain.attributes[i] for i, ok in enumerate(oks) if ok]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return Orange.data.Table(domain, data)


class PreprocessorList(object):
    """
    Store a list of preprocessors and on call apply them to the data set.

    Parameters
    ----------
    preprocessors : list
        A list of preprocessors.
    """

    def __init__(self, preprocessors):
        self.preprocessors = list(preprocessors)

    def __call__(self, data):
        """
        Applies a list of preprocessors to the data set.

        Parameters
        ----------
        data : an input data table
        """

        for pp in self.preprocessors:
            data = pp(data)
        return data

