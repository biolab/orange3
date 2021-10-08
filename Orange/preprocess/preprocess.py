"""
Preprocess
----------

"""
import numpy as np
import bottleneck as bn
import scipy.sparse as sp
from sklearn.impute import SimpleImputer

import Orange.data
from Orange.data.filter import HasClass
from Orange.statistics import distribution
from Orange.util import Reprable, Enum, deprecated
from . import impute, discretize, transformation

__all__ = ["Continuize", "Discretize", "Impute", "RemoveNaNRows",
           "SklImpute", "Normalize", "Randomize", "Preprocess",
           "RemoveConstant", "RemoveNaNClasses", "RemoveNaNColumns",
           "ProjectPCA", "ProjectCUR", "Scale", "RemoveSparse",
           "AdaptiveNormalize", "PreprocessorList"]


class Preprocess(Reprable):
    """
    A generic preprocessor base class.

    Methods
    -------
    __call__(data: Table) -> Table
        Return preprocessed data.
    """
    def __call__(self, data):
        raise NotImplementedError("Subclasses need to implement __call__")


class Continuize(Preprocess):
    MultinomialTreatment = Enum(
        "Continuize",
        ("Indicators", "FirstAsBase", "FrequentAsBase", "Remove",
         "RemoveMultinomial", "ReportError", "AsOrdinal", "AsNormalizedOrdinal",
         "Leave"),
        qualname="Continuize.MultinomialTreatment")
    (Indicators, FirstAsBase, FrequentAsBase, Remove, RemoveMultinomial,
     ReportError, AsOrdinal, AsNormalizedOrdinal, Leave) = MultinomialTreatment

    def __init__(self, zero_based=True,
                 multinomial_treatment=Indicators):
        self.zero_based = zero_based
        self.multinomial_treatment = multinomial_treatment

    def __call__(self, data):
        from . import continuize

        continuizer = continuize.DomainContinuizer(
            zero_based=self.zero_based,
            multinomial_treatment=self.multinomial_treatment)
        domain = continuizer(data)
        return data.transform(domain)


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

        def discretized(vars_, do_discretize):
            if do_discretize:
                vars_ = (transform(var) for var in vars_)
                vars_ = [var for var in vars_ if var is not None]
            return vars_

        method = self.method or discretize.EqualFreq()
        domain = Orange.data.Domain(
            discretized(data.domain.attributes, True),
            discretized(data.domain.class_vars, self.discretize_classes),
            discretized(data.domain.metas, self.discretize_metas))
        return data.transform(domain)


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
        Apply an imputation method to the given dataset. Returns a new
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
        return data.transform(domain)


class SklImpute(Preprocess):
    __wraps__ = SimpleImputer

    def __init__(self, strategy='mean'):
        self.strategy = strategy

    def __call__(self, data):
        from Orange.data.sql.table import SqlTable
        if isinstance(data, SqlTable):
            return Impute()(data)
        imputer = SimpleImputer(strategy=self.strategy)
        X = imputer.fit_transform(data.X)
        # Create new variables with appropriate `compute_value`, but
        # drop the ones which do not have valid `imputer.statistics_`
        # (i.e. all NaN columns). `sklearn.preprocessing.Imputer` already
        # drops them from the transformed X.
        features = [impute.Average()(data, var, value)
                    for var, value in zip(data.domain.attributes,
                                          imputer.statistics_)
                    if not np.isnan(value)]
        assert X.shape[1] == len(features)
        domain = Orange.data.Domain(features, data.domain.class_vars,
                                    data.domain.metas)
        new_data = data.transform(domain)
        with new_data.unlocked(new_data.X):
            new_data.X = X
        return new_data


class RemoveConstant(Preprocess):
    """
    Construct a preprocessor that removes features with constant values
    from the dataset.
    """

    def __call__(self, data):
        """
        Remove columns with constant values from the dataset and return
        the resulting data table.

        Parameters
        ----------
        data : an input dataset
        """

        oks = np.logical_and(~bn.allnan(data.X, axis=0),
                             bn.nanmin(data.X, axis=0) != bn.nanmax(data.X, axis=0))
        atts = [data.domain.attributes[i] for i, ok in enumerate(oks) if ok]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.transform(domain)


class RemoveNaNRows(Preprocess):
    _reprable_module = True

    def __call__(self, data):
        mask = np.isnan(data.X)
        mask = np.any(mask, axis=1)
        return data[~mask]


class RemoveNaNColumns(Preprocess):
    """
    Remove features from the data domain if they contain
    `threshold` or more unknown values.

    `threshold` can be an integer or a float in the range (0, 1) representing
    the fraction of the data size. When not provided, columns with only missing
    values are removed (default).
    """
    def __init__(self, threshold=None):
        self.threshold = threshold

    def __call__(self, data, threshold=None):
        # missing entries in sparse data are treated as zeros so we skip removing NaNs
        if sp.issparse(data.X):
            return data

        if threshold is None:
            threshold = data.X.shape[0] if self.threshold is None else \
                        self.threshold
        if isinstance(threshold, float):
            threshold = threshold * data.X.shape[0]
        nans = np.sum(np.isnan(data.X), axis=0)
        att = [a for a, n in zip(data.domain.attributes, nans) if n < threshold]
        domain = Orange.data.Domain(att, data.domain.class_vars,
                                    data.domain.metas)
        return data.transform(domain)


@deprecated("Orange.data.filter.HasClas")
class RemoveNaNClasses(Preprocess):
    """
    Construct preprocessor that removes examples with missing class
    from the dataset.
    """

    def __call__(self, data):
        """
        Remove rows that contain NaN in any class variable from the dataset
        and return the resulting data table.

        Parameters
        ----------
        data : an input dataset

        Returns
        -------
        data : dataset without rows with missing classes
        """
        return HasClass()(data)


class Normalize(Preprocess):
    """
    Construct a preprocessor for normalization of features.
    Given a data table, preprocessor returns a new table in
    which the continuous attributes are normalized.

    Parameters
    ----------
    zero_based : bool (default=True)
        Only used when `norm_type=NormalizeBySpan`.

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

    center : bool (default=True)
        Only used when `norm_type=NormalizeBySD`.

        Whether or not to center the data so it has mean zero.

    normalize_datetime : bool (default=False)


    Examples
    --------
    >>> from Orange.data import Table
    >>> from Orange.preprocess import Normalize
    >>> data = Table("iris")
    >>> normalizer = Normalize(norm_type=Normalize.NormalizeBySpan)
    >>> normalized_data = normalizer(data)
    """
    Type = Enum("Normalize", ("NormalizeBySpan", "NormalizeBySD"),
                qualname="Normalize.Type")
    NormalizeBySpan, NormalizeBySD = Type

    def __init__(self,
                 zero_based=True,
                 norm_type=NormalizeBySD,
                 transform_class=False,
                 center=True,
                 normalize_datetime=False):
        self.zero_based = zero_based
        self.norm_type = norm_type
        self.transform_class = transform_class
        self.center = center
        self.normalize_datetime = normalize_datetime

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
            # Skip normalization for datasets where all features are marked as already normalized.
            # Required for SVMs (with normalizer as their default preprocessor) on sparse data to
            # retain sparse structure. Normalizing sparse data would otherwise result in a dense
            # matrix, which requires too much memory. For example, this is used for Bag of Words
            # models where normalization is not really needed.
            return data

        normalizer = normalize.Normalizer(
            zero_based=self.zero_based,
            norm_type=self.norm_type,
            transform_class=self.transform_class,
            center=self.center,
            normalize_datetime=self.normalize_datetime
        )
        return normalizer(data)


class Randomize(Preprocess):
    """
    Construct a preprocessor for randomization of classes,
    attributes and/or metas.
    Given a data table, preprocessor returns a new table in
    which the data is shuffled.

    Parameters
    ----------

    rand_type : RandTypes (default: Randomize.RandomizeClasses)
        Randomization type. If Randomize.RandomizeClasses, classes
        are shuffled.
        If Randomize.RandomizeAttributes, attributes are shuffled.
        If Randomize.RandomizeMetas, metas are shuffled.

    rand_seed : int (optional)
        Random seed

    Examples
    --------
    >>> from Orange.data import Table
    >>> from Orange.preprocess import Randomize
    >>> data = Table("iris")
    >>> randomizer = Randomize(Randomize.RandomizeClasses)
    >>> randomized_data = randomizer(data)
    """
    Type = Enum("Randomize",
                dict(RandomizeClasses=1,
                     RandomizeAttributes=2,
                     RandomizeMetas=4),
                type=int,
                qualname="Randomize.Type")
    RandomizeClasses, RandomizeAttributes, RandomizeMetas = Type

    def __init__(self, rand_type=RandomizeClasses, rand_seed=None):
        self.rand_type = rand_type
        self.rand_seed = rand_seed

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
        new_data = data.copy()
        rstate = np.random.RandomState(self.rand_seed)
        # ensure the same seed is not used to shuffle X and Y at the same time
        r1, r2, r3 = rstate.randint(0, 2 ** 32 - 1, size=3, dtype=np.int64)
        with new_data.unlocked():
            if self.rand_type & Randomize.RandomizeClasses:
                new_data.Y = self.randomize(new_data.Y, r1)
            if self.rand_type & Randomize.RandomizeAttributes:
                new_data.X = self.randomize(new_data.X, r2)
            if self.rand_type & Randomize.RandomizeMetas:
                new_data.metas = self.randomize(new_data.metas, r3)
        return new_data

    @staticmethod
    def randomize(table, rand_state=None):
        rstate = np.random.RandomState(rand_state)
        if sp.issparse(table):
            table = table.tocsc()  # type: sp.spmatrix
            for i in range(table.shape[1]):
                permutation = rstate.permutation(table.shape[0])
                col_indices = \
                    table.indices[table.indptr[i]: table.indptr[i + 1]]
                col_indices[:] = permutation[col_indices]
        elif len(table.shape) > 1:
            for i in range(table.shape[1]):
                rstate.shuffle(table[:, i])
        else:
            rstate.shuffle(table)
        return table


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


class Scale(Preprocess):
    """
    Scale data preprocessor.  Scales data so that its distribution remains
    the same but its location on the axis changes.
    """
    class _MethodEnum(Enum):
        def __call__(self, *args, **kwargs):
            return getattr(Scale, '_' + self.name)(*args, **kwargs)

    CenteringType = _MethodEnum("Scale", ("NoCentering", "Mean", "Median"),
                                qualname="Scale.CenteringType")
    ScalingType = _MethodEnum("Scale", ("NoScaling", "Std", "Span"),
                              qualname="Scale.ScalingType")
    NoCentering, Mean, Median = CenteringType
    NoScaling, Std, Span = ScalingType

    @staticmethod
    def _Mean(dist):
        values, counts = np.array(dist)
        return np.average(values, weights=counts)

    @staticmethod
    def _Median(dist):
        values, counts = np.array(dist)
        cumdist = np.cumsum(counts)
        if cumdist[-1] > 0:
            cumdist /= cumdist[-1]
        return np.interp(.5, cumdist, values)

    @staticmethod
    def _Std(dist):
        values, counts = np.array(dist)
        mean = np.average(values, weights=counts)
        diff = values - mean
        return np.sqrt(np.average(diff ** 2, weights=counts))

    @staticmethod
    def _Span(dist):
        values = np.array(dist[0])
        return np.max(values) - np.min(values)

    def __init__(self, center=Mean, scale=Std):
        self.center = center
        self.scale = scale

    def __call__(self, data):
        if self.center is None and self.scale is None:
            return data

        def transform(var):
            dist = distribution.get_distribution(data, var)
            if self.center != self.NoCentering:
                c = self.center(dist)
                dist[0, :] -= c
            else:
                c = 0

            if self.scale != self.NoScaling:
                s = self.scale(dist)
                if s < 1e-15:
                    s = 1
            else:
                s = 1
            factor = 1 / s
            transformed_var = var.copy(
                compute_value=transformation.Normalizer(var, c, factor))
            return transformed_var

        newvars = []
        for var in data.domain.attributes:
            if var.is_continuous:
                newvars.append(transform(var))
            else:
                newvars.append(var)
        domain = Orange.data.Domain(newvars, data.domain.class_vars,
                                    data.domain.metas)
        return data.transform(domain)


class PreprocessorList(Preprocess):
    """
    Store a list of preprocessors and on call apply them to the dataset.

    Parameters
    ----------
    preprocessors : list
        A list of preprocessors.
    """

    def __init__(self, preprocessors=()):
        self.preprocessors = list(preprocessors)

    def __call__(self, data):
        """
        Applies a list of preprocessors to the dataset.

        Parameters
        ----------
        data : an input data table
        """

        for pp in self.preprocessors:
            data = pp(data)
        return data

class RemoveSparse(Preprocess):
    """
    Filter out the features with too many (>threshold) zeros or missing values. Threshold is user defined.

    Parameters
    ----------
    threshold: int or float
        if >= 1, the argument represents the allowed number of 0s or NaNs;
        if below 1, it represents the allowed proportion of 0s or NaNs
    filter0: bool
        if True (default), preprocessor counts 0s, otherwise NaNs
    """
    def __init__(self, threshold=0.05, filter0=True):
        self.filter0 = filter0
        self.threshold = threshold

    def __call__(self, data):
        threshold = self.threshold
        if self.threshold < 1:
            threshold *= data.X.shape[0]

        if self.filter0:
            if sp.issparse(data.X):
                data_csc = sp.csc_matrix(data.X)
                h, w = data_csc.shape
                sparseness = [h - data_csc[:, i].count_nonzero()
                              for i in range(w)]
            else:
                sparseness = data.X.shape[0] - np.count_nonzero(data.X, axis=0)
        else:  # filter by nans
            if sp.issparse(data.X):
                data_csc = sp.csc_matrix(data.X)
                sparseness = [np.sum(np.isnan(data.X[:, i].data))
                              for i in range(data_csc.shape[1])]
            else:
                sparseness = np.sum(np.isnan(data.X), axis=0)
        att = [a for a, s in zip(data.domain.attributes, sparseness)
               if s <= threshold]
        domain = Orange.data.Domain(att, data.domain.class_vars,
                                    data.domain.metas)
        return data.transform(domain)


class AdaptiveNormalize(Preprocess):
    """
    Construct a preprocessors that normalizes or merely scales the data.
    If the input is sparse, data is only scaled, to avoid turning it to
    dense. Parameters are diveded to those passed to Normalize or Scale
    class. Scaling takes only scale parameter.
     If the user wants to have more options with scaling,
    they should use the preprocessing widget.
    For more details, check Scale and Normalize widget.

    Parameters
    ----------

    zero_based : bool (default=True)
        passed to Normalize

    norm_type : NormTypes (default: Normalize.NormalizeBySD)
        passed to Normalize

    transform_class : bool (default=False)
        passed to Normalize

    center : bool(default=True)
        passed to Normalize

    normalize_datetime : bool (default=False)
        passed to Normalize

    scale : ScaleTypes (default: Scale.Span)
        passed to Scale
    """

    def __init__(self,
                 zero_based=True,
                 norm_type=Normalize.NormalizeBySD,
                 transform_class=False,
                 normalize_datetime=False,
                 center=True,
                 scale=Scale.Span):
        self.normalize_pps = Normalize(zero_based,
                                       norm_type,
                                       transform_class,
                                       center,
                                       normalize_datetime)
        self.scale_pps = Scale(center=Scale.NoCentering, scale=scale)

    def __call__(self, data):
        if sp.issparse(data.X):
            return self.scale_pps(data)
        return self.normalize_pps(data)
