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
from collections.abc import Iterable

__all__ = ["Continuize", "Discretize", "Impute", "SklImpute", "Normalize"]


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
        "AsNormalizedOrdinal", "Leave"
    )

    (Indicators, FirstAsBase, FrequentAsBase, Remove, RemoveMultinomial,
     ReportError, AsOrdinal, AsNormalizedOrdinal, Leave) = MultinomialTreatment

    def __init__(self, zero_based=True, multinomial_treatment=Indicators):
        self.zero_based = zero_based
        self.multinomial_treatment = multinomial_treatment

    def __call__(self, data):
        from . import continuize

        continuizer = continuize.DomainContinuizer(
            zero_based=self.zero_based,
            multinomial_treatment=self.multinomial_treatment)
        domain = continuizer(data)
        return data.from_table(domain, data)


class Discretize(Preprocess):
    """
    Construct a discretizer, a preprocessor for discretization of
    continuous features.

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
            if var.is_continuous:
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


class Normalize(Preprocess):
    """
    Construct a preprocessor for normalization of features.
    Given a data table, preprocessor returns a new table in
    which the continuous attributes are normalized.

    Parameters
    ----------
    zero_based : bool (default=True)
        Determines the value used as the “low” value of the variable.
        It determines the interval for normalized continuous variables
        (either [-1, 1] or [0, 1]).

    norm_type : NormTypes (default: Normalize.NormalizeBySD)
        Normalization type. If Normalize.NormalizeBySD, the values are
        replaced with standardized values by subtracting the average
        value and dividing by the standard deviation.
        Attribute zero_based has no effect on this standardization.

        If Normalize.NormalizeBySpan, the values are replaced with
        normalized values by subtracting min value of the data and
        dividing by span (max - min).

    transform_class : bool (default=False)
        If True the class is normalized as well.

    Examples
    --------
    >>> from Orange.data import Table
    >>> from Orange.preprocess import Normalize
    >>> data = Table("iris")
    >>> normalizer = Normalize(Normalize.NormalizeBySpan)
    >>> normalized_data = normalizer(data)
    """

    NormTypes = Enum("NormalizeBySpan", "NormalizeBySD")
    (NormalizeBySpan, NormalizeBySD) = NormTypes

    def __init__(self,
                 zero_based=True,
                 norm_type=NormalizeBySD,
                 transform_class=False):
        self.zero_based = zero_based
        self.norm_type = norm_type
        self.transform_class = transform_class

    def __call__(self, data):
        """
        Compute and apply normalization of the given data. Returns a new
        data table.

        Parameters
        ----------
        data : Orange.data.Table
            A data table to be normalized.

        Returns
        -------
        data : Orange.data.Table
            Normalized data table.
        """
        from . import normalize

        normalizer = normalize.Normalizer(
            zero_based=self.zero_based,
            norm_type=self.norm_type,
            transform_class=self.transform_class)
        return normalizer(data)


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


def is_same_type(a, b):
    """
    Return True if input preprocessors a and b are of the same kind.
    """
    return type(a) == type(b) or \
        (isinstance(a, (SklImpute, Impute)) and isinstance(b, (SklImpute, Impute)))


def add_preprocessors(preprocessors, old=None):
    """ Add preprocessors to a list of preprocessors.
    
    If new preprocessors are in the same order than the old, 
    just replace old preprocessors with a matching new one.
    
    If the order is different remove matching old preprocessors
    and append new ones in the given order.
    """

    def to_list(preproc):
        if isinstance(preproc, PreprocessorList):
            preproc = preproc.preprocessors
        if not preproc:
            preproc = []
        if not isinstance(preproc, Iterable):
            preproc = [ preproc]
        return preproc

    old = to_list(old)
    preprocessors = to_list(preprocessors)

    add_preprocessors = []

    #check the order of new preprocessors
    same_type_ind = []
    for a in preprocessors:
        for i,b in enumerate(old):
            if is_same_type(a,b):
                same_type_ind.append(i)

    #is order the same?
    replace = same_type_ind == sorted(same_type_ind)

    current = [ [] for a in old ] #allow multiple of the same type

    for a in preprocessors:
        same_type_ind = []
        for i,b in enumerate(old):
            if is_same_type(a,b):
                same_type_ind.append(i)
        if same_type_ind:
            if replace:
                current[same_type_ind[0]].append(a)
            else:
                for s in same_type_ind:
                    current[s] = None #mark a preprocessor for removal
                add_preprocessors.append(a)
        else:
            add_preprocessors.append(a)
    
    #keep those that were not changed
    for i,(a,o) in enumerate(zip(current, old)):
        if a == []:
            current[i] = [o]

    currentl = []
    for c in current:
        if c: #skip preprocesors marked for removal
            currentl += c

    return currentl + add_preprocessors
