"""
Preprocess
----------

"""
import numpy as np
import sklearn.preprocessing as skl_preprocessing
import bottleneck as bn

import Orange.data
from Orange.data import Table
from . import impute, discretize
from ..misc.enum import Enum

__all__ = ["Continuize", "Discretize", "Impute", "SklImpute",
           "Normalize", "Randomize", "RemoveNaNClasses",
           "ProjectPCA", "ProjectCUR"]


class Preprocess:
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

    remove_const : bool (default=True)
        Determines whether the features with constant values are removed
        during discretization.
    """

    def __init__(self, method=None, remove_const=True,
                 discretize_classes=False, discretize_metas=False):
        self.method = method
        self.remove_const = remove_const
        self.discretize_classes = discretize_classes
        self.discretize_metas = discretize_metas

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
                if new_var is not None and \
                        (len(new_var.values) >= 2 or not self.remove_const):
                    return new_var
                else:
                    return None
            else:
                return var

        def discretized(vars, do_discretize):
            if do_discretize:
                vars = (transform(var) for var in vars)
                vars = [var for var in vars if var is not None]
            return vars

        method = self.method or discretize.EqualFreq()
        domain = Orange.data.Domain(
            discretized(data.domain.attributes, True),
            discretized(data.domain.class_vars, self.discretize_classes),
            discretized(data.domain.metas, self.discretize_metas))
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

    def __init__(self, strategy='mean'):
        self.strategy = strategy

    def __call__(self, data):
        from Orange.data.sql.table import SqlTable
        if isinstance(data, SqlTable):
            return Impute()(data)
        self.imputer = skl_preprocessing.Imputer(strategy=self.strategy)
        X = self.imputer.fit_transform(data.X)
        # Create new variables with appropriate `compute_value`, but
        # drop the ones which do not have valid `imputer.statistics_`
        # (i.e. all NaN columns). `sklearn.preprocessing.Imputer` already
        # drops them from the transformed X.
        features = [impute.Average()(data, var, value)
                    for var, value in zip(data.domain.attributes,
                                          self.imputer.statistics_)
                    if not np.isnan(value)]
        assert X.shape[1] == len(features)
        domain = Orange.data.Domain(features, data.domain.class_vars,
                                    data.domain.metas)
        new_data = Orange.data.Table(domain, X, data.Y, data.metas, W=data.W)
        new_data.attributes = getattr(data, 'attributes', {})
        return new_data


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

        oks = bn.nanmin(data.X, axis=0) != \
              bn.nanmax(data.X, axis=0)
        atts = [data.domain.attributes[i] for i, ok in enumerate(oks) if ok]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return Orange.data.Table(domain, data)


class RemoveNaNClasses(Preprocess):
    """
    Construct preprocessor that removes examples with missing class
    from the data set.
    """

    def __call__(self, data):
        """
        Remove rows that contain NaN in any class variable from the data set
        and return the resulting data table.

        Parameters
        ----------
        data : an input data set

        Returns
        -------
        data : data set without rows with missing classes
        """
        if len(data.Y.shape) > 1:
            nan_cls = np.any(np.isnan(data.Y), axis=1)
        else:
            nan_cls = np.isnan(data.Y)
        return Table(data.domain, data, np.where(nan_cls == False))


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
    >>> normalizer = Normalize(norm_type=Normalize.NormalizeBySpan)
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

        if all(a.attributes.get('skip-normalization', False)
               for a in data.domain.attributes if a.is_continuous):
            # Skip normalization for data sets where all features are marked as already normalized.
            # Required for SVMs (with normalizer as their default preprocessor) on sparse data to
            # retain sparse structure. Normalizing sparse data would otherwise result in a dense matrix,
            # which requires too much memory. For example, this is used for Bag of Words models where
            # normalization is not really needed.
            return data
        normalizer = normalize.Normalizer(
            zero_based=self.zero_based,
            norm_type=self.norm_type,
            transform_class=self.transform_class)
        return normalizer(data)


class Randomize(Preprocess):
    """
    Construct a preprocessor for randomization of classes,
    attributes or metas.
    Given a data table, preprocessor returns a new table in
    which the data is shuffled.

    Parameters
    ----------

    rand_type : RandTypes (default: Randomize.RandomizeClasses)
        Randomization type. If Randomize.RandomizeClasses, classes
        are shuffled.
        If Randomize.RandomizeAttributes, attributes are shuffled.
        If Randomize.RandomizeMetas, metas are shuffled.

    Examples
    --------
    >>> from Orange.data import Table
    >>> from Orange.preprocess import Randomize
    >>> data = Table("iris")
    >>> randomizer = Randomize(Randomize.RandomizeClasses)
    >>> randomized_data = randomizer(data)
    """

    RandTypes = Enum("RandomizeClasses", "RandomizeAttributes",
                     "RandomizeMetas")
    (RandomizeClasses, RandomizeAttributes, RandomizeMetas) = RandTypes

    def __init__(self, rand_type=RandomizeClasses):
        self.rand_type = rand_type

    def __call__(self, data):
        """
        Apply randomization of the given data. Returns a new
        data table.

        Parameters
        ----------
        data : Orange.data.Table
            A data table to be randomized.

        Returns
        -------
        data : Orange.data.Table
            Randomized data table.
        """
        new_data = Table(data)
        new_data.ensure_copy()

        if self.rand_type == Randomize.RandomizeClasses:
            self.randomize(new_data.Y)
        elif self.rand_type == Randomize.RandomizeAttributes:
            self.randomize(new_data.X)
        elif self.rand_type == Randomize.RandomizeMetas:
            self.randomize(new_data.metas)
        else:
            raise TypeError('Unsupported type')

        return new_data

    def randomize(self, table):
        if len(table.shape) > 1:
            for i in range(table.shape[1]):
                np.random.shuffle(table[:,i])
        else:
            np.random.shuffle(table)


class ProjectPCA(Preprocess):

    def __init__(self, n_components=None):
        self.n_components = n_components

    def __call__(self, data):
        pca = Orange.projection.PCA(n_components=self.n_components)(data)
        return pca(data)


class ProjectCUR(Preprocess):

    def __init__(self, rank=3, max_error=1):
        self.rank = rank
        self.max_error = max_error

    def __call__(self, data):
        rank = min(self.rank, min(data.X.shape)-1)
        cur = Orange.projection.CUR(
            rank=rank, max_error=self.max_error,
            compute_U=False,
        )(data)
        return cur(data)


class PreprocessorList:
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

