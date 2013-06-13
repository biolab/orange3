from Orange.data import Domain
import numpy as np


class Results:
    """
    Class for storing predicted in model testing.

    .. attribute:: data

        Data used for testing (optional; can be `None`). When data is stored,
        this is typically not a copy but a reference.

    .. attribute:: row_indices

        Indices of rows in :obj:`data` that were used in testing, stored as
        a numpy vector of length `nrows`. Values of `actual[i]`, `predicted[i]`
        and `probabilities[i]` refer to the target value of instance
        `data[row_indices[i]]`.

    .. attribute:: nrows

        The number of test instances (including duplicates).

    .. attribute:: models

        A list of induced models (optional; can be `None`).

    .. attribute:: actual

        Actual values of target variable; a numpy vector of length `nrows` and
        of the same type as `data` (or `np.float32` if the type of data cannot
        be determined).

    .. attribute:: predicted

        Predicted values of target variable; a numpy array of shape
        (number-of-methods, `nrows`) and of the same type as `data` (or
        `np.float32` if the type of data cannot be determined).

    .. attribute:: probabilities

        Predicted probabilities (for discrete target variables); a numpy array
        of shape (number-of-methods, `nrows`, number-of-classes) of type
        `np.float32`.
    """

    # noinspection PyBroadException
    def __init__(self, data, nmethods, nrows=None):
        """
        Construct an instance with default values: `None` for :obj:data and
        :obj:models. The number of classes and the data type for :obj:actual
        and :obj:predicted is determined from the data; if the latter cannot
        be find, `np.float32` is used. Attributes :obj:actual, :obj:predicted,
        :obj:probabilities and :obj:row_indices are constructed as empty
        (uninitialized) arrays of the appropriate size.

        :param data: Data or domain; the data is not stored
        :type data: Orange.data.Table or Orange.data.Domain
        :param nmethods: The number of methods that will be tested
        :type nmethods: int
        :param nrows: The number of test instances (including duplicates)
        :type nrows: int
        """
        self.data = None
        self.models = None
        domain = data if isinstance(data, Domain) else data.domain
        nclasses = len(domain.class_var.values)
        if nrows is None:
            nrows = len(data)
        try:
            dtype = data.Y.dtype
        except:
            dtype = np.float32
        self.actual = np.empty(nrows, dtype=dtype)
        self.predicted = np.empty((nmethods, nrows), dtype=dtype)
        self.probabilities = np.empty((nmethods, nrows, nclasses),
                                      dtype=np.float32)
        self.row_indices = np.empty(nrows, dtype=np.int32)


class Testing:
    """
    Abstract base class for varius sampling procedures like cross-validation,
    leave one out or bootstrap. Derived classes define a `__call__` operator
    that executes the testing procedure and returns an instance of
    :obj:Results.

    .. attribute:: store_data

        A flag that tells whether to store the data used for test.

    .. attribute:: store_models

        A flag that tells whether to store the constructed models
    """

    def __new__(cls, data=None, learners=None, **kwargs):
        """
        Construct an instance and store the values of keyword arguments
        `store_data` and `store_classifiers`, if given. If `data` and
        `learners` are given, the constructor also calls the testing
        procedure. For instance, CrossValidation(data, learners) will return
        an instance of `Results` with results of cross validating `learners` on
        `data`.

        :param data: Data instances used for testing procedures
        :type data: Orange.data.Storage
        :param learners: A list of learning algorithms to be tested
        :type learners: list of Orange.classification.Fitter
        :param store_data: A flag that tells whether to store the data (as
            keyword argument)
        :type store_data: bool
        :param store_models: A flag that tells whether to store the models (as
            keyword argument)
        :type store_models: bool
        """
        self = super().__new__(cls)
        self.store_data = kwargs.pop('store_data', False)
        self.store_classifiers = kwargs.pop('store_models', False)

        if data or learners:
            self = self.__init__(**kwargs)
            return self(data, learners)
        return self


class CrossValidation(Testing):
    """
    K-fold cross validation.

    If the constructor is given the data and a list of learning algorithms, it
    runs cross validation and returns an instance of `Results` containing the
    predicted values and probabilities.

    .. attribute:: k

        The number of folds.
    """
    def __init__(self, k=10, **kwargs):
        self.k = kwargs.pop('k', 10)

    def __call__(self, data, fitters):
        from sklearn import cross_validation
        n = len(data)
        indices = cross_validation.KFold(n, self.k, shuffle=True)
        results = Results(data, len(fitters))

        for train, test in indices:
            for i, fitter in enumerate(fitters):
                model = fitter(data[train])
                values, probs = model(data[test], model.ValueProbs)
                results.predicted[i][test, :] = values
                results.probabilities[i][test, :] = probs
        return results
