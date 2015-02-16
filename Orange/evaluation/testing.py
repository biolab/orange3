import numpy as np

import sklearn.cross_validation as skl_cross_validation

import Orange.data
from Orange.data import Domain, Table

__all__ = ["Results", "CrossValidation", "LeaveOneOut", "TestOnTrainingData",
           "Bootstrap", "TestOnTestData", "sample"]


def is_discrete(var):
    return isinstance(var, Orange.data.DiscreteVariable)


def is_continuous(var):
    return isinstance(var, Orange.data.ContinuousVariable)


class Results:
    """
    Class for storing predictions in model testing.

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

    .. attribute:: folds

        A list of indices (or slice objects) corresponding to rows of each
        fold; `None` if not applicable.
    """

    # noinspection PyBroadException
    # noinspection PyNoneFunctionAssignment
    def __init__(self, data=None, nmethods=0, nrows=None, nclasses=None,
                 store_data=False, store_models=False, domain=None,
                 actual=None, row_indices=None,
                 predicted=None, probabilities=None):
        """
        Construct an instance with default values: `None` for :obj:`data` and
        :obj:`models`.

        If the number of rows and/or the number of classes is not given, it is
        inferred from :obj:`data`, if provided. The data type for
        :obj:`actual` and :obj:`predicted` is determined from the data; if the
        latter cannot be find, `np.float32` is used.

        Attribute :obj:`actual` and :obj:`row_indices` are constructed as empty
        (uninitialized) arrays of the appropriate size, if the number of rows
        is known. Attribute :obj:`predicted` is constructed if the number of
        rows and of methods is given; :obj:`probabilities` also requires
        knowing the number of classes.

        :param data: Data or domain
        :type data: Orange.data.Table or Orange.data.Domain
        :param nmethods: The number of methods that will be tested
        :type nmethods: int
        :param nrows: The number of test instances (including duplicates)
        :type nrows: int
        :param nclasses: The number of class values
        :type nclasses: int
        :param store_data: A flag that tells whether to store the data;
            this argument can be given only as keyword argument
        :type store_data: bool
        :param store_models: A flag that tells whether to store the models;
            this argument can be given only as keyword argument
        :type store_models: bool
        """
        self.store_data = store_data
        self.store_models = store_models
        self.data = data if store_data else None
        self.models = None
        self.folds = None
        dtype = np.float32

        def set_or_raise(value, exp_values, msg):
            for exp_value in exp_values:
                if exp_value is False:
                    continue
                if value is None:
                    value = exp_value
                elif value != exp_value:
                    raise ValueError(msg)
            return value

        domain = self.domain = set_or_raise(
            domain, [data is not None and data.domain],
            "mismatching domain")
        nrows = set_or_raise(
            nrows, [data is not None and len(data),
                    actual is not None and len(actual),
                    row_indices is not None and len(row_indices),
                    predicted is not None and predicted.shape[1],
                    probabilities is not None and probabilities.shape[1]],
            "mismatching number of rows")
        nclasses = set_or_raise(
            nclasses, [domain is not None and (len(domain.class_var.values)
                                               if is_discrete(domain.class_var)
                                               else None),
                       probabilities is not None and probabilities.shape[2]],
            "mismatching number of class values")
        if nclasses is not None and probabilities is not None:
            raise ValueError("regression results cannot have 'probabilities'")
        nmethods = set_or_raise(
            nmethods, [predicted is not None and predicted.shape[0],
                       probabilities is not None and probabilities.shape[0]],
            "mismatching number of methods")
        try:
            dtype = data.Y.dtype
        except AttributeError:  # no data or no Y or not numpy
            pass

        if actual is not None:
            self.actual = actual
        elif nrows is not None:
            self.actual = np.empty(nrows, dtype=dtype)

        if row_indices is not None:
            self.row_indices = row_indices
        elif nrows is not None:
                self.row_indices = np.empty(nrows, dtype=np.int32)

        if predicted is not None:
            self.predicted = predicted
        elif nmethods is not None and nrows is not None:
            self.predicted = np.empty((nmethods, nrows), dtype=dtype)

        if probabilities is not None:
            self.probabilities = probabilities
        elif nmethods is not None and nrows is not None and \
                nclasses is not None:
            self.probabilities = \
                np.empty((nmethods, nrows, nclasses), dtype=np.float32)

    def get_fold(self, fold):
        results = Results()
        results.data = self.data

        if self.folds is None:
            raise ValueError("This 'Results' instance does not have folds.")

        if self.models is not None:
            results.models = self.models[fold]

        results.row_indices = self.row_indices[self.folds[fold]]
        results.actual = self.actual[self.folds[fold]]
        results.predicted = self.predicted[:, self.folds[fold]]
        results.domain = self.domain

        if self.probabilities is not None:
            results.probabilities = self.probabilities[:, self.folds[fold]]

        return results


class CrossValidation(Results):
    """
    K-fold cross validation.

    If the constructor is given the data and a list of learning algorithms, it
    runs cross validation and returns an instance of `Results` containing the
    predicted values and probabilities.

    .. attribute:: k

        The number of folds.

    .. attribute:: random_state

    """
    def __init__(self, data, learners, k=10, random_state=0, store_data=False,
                 store_models=False):
        super().__init__(data, len(learners), store_data=store_data,
                         store_models=store_models)
        self.k = k
        self.random_state = random_state
        Y = data.Y.copy().flatten()
        if is_discrete(data.domain.class_var):
            indices = skl_cross_validation.StratifiedKFold(
                Y, self.k, shuffle=True, random_state=self.random_state
            )
        else:
            indices = skl_cross_validation.KFold(
                len(Y), self.k, shuffle=True, random_state=self.random_state
            )


        self.folds = []
        if self.store_models:
            self.models = []
        ptr = 0
        class_var = data.domain.class_var
        for train, test in indices:
            train_data, test_data = data[train], data[test]
            if len(test_data) == 0:
                raise RuntimeError("One of the test folds is empty.")
            fold_slice = slice(ptr, ptr + len(test))
            self.folds.append(fold_slice)
            self.row_indices[fold_slice] = test
            self.actual[fold_slice] = test_data.Y.flatten()
            if self.store_models:
                fold_models = []
                self.models.append(fold_models)
            for i, learner in enumerate(learners):
                model = learner(train_data)
                if self.store_models:
                    fold_models.append(model)

                if is_discrete(class_var):
                    values, probs = model(test_data, model.ValueProbs)
                    self.predicted[i][fold_slice] = values
                    self.probabilities[i][fold_slice, :] = probs
                elif is_continuous(class_var):
                    values = model(test_data, model.Value)
                    self.predicted[i][fold_slice] = values

            ptr += len(test)


class LeaveOneOut(Results):
    """Leave-one-out testing"""

    def __init__(self, data, learners, store_data=False, store_models=False):
        super().__init__(data, len(learners), store_data=store_data,
                         store_models=store_models)

        domain = data.domain
        X = data.X.copy()
        Y = data._Y.copy()
        metas = data.metas.copy()

        teX, trX = X[:1], X[1:]
        teY, trY = Y[:1], Y[1:]
        te_metas, tr_metas = metas[:1], metas[1:]
        if data.has_weights():
            W = data.W.copy()
            teW, trW = W[:1], W[1:]
        else:
            W = teW = trW = None

        self.row_indices = np.arange(len(data))
        if self.store_models:
            self.models = []
        self.actual = Y.flatten()
        class_var = data.domain.class_var
        for test_idx in self.row_indices:
            X[[0, test_idx]] = X[[test_idx, 0]]
            Y[[0, test_idx]] = Y[[test_idx, 0]]
            metas[[0, test_idx]] = metas[[test_idx, 0]]
            if W:
                W[[0, test_idx]] = W[[test_idx, 0]]
            test_data = Table.from_numpy(domain, teX, teY, te_metas, teW)
            train_data = Table.from_numpy(domain, trX, trY, tr_metas, trW)
            if self.store_models:
                fold_models = []
                self.models.append(fold_models)
            for i, learner in enumerate(learners):
                model = learner(train_data)
                if self.store_models:
                    fold_models.append(model)

                if is_discrete(class_var):
                    values, probs = model(test_data, model.ValueProbs)
                    self.predicted[i][test_idx] = values
                    self.probabilities[i][test_idx, :] = probs
                elif is_continuous(class_var):
                    values = model(test_data, model.Value)
                    self.predicted[i][test_idx] = values


class TestOnTrainingData(Results):
    """Trains and test on the same data"""

    def __init__(self, data, learners, store_data=False, store_models=False):
        super().__init__(data, len(learners), store_data=store_data,
                         store_models=store_models)
        self.row_indices = np.arange(len(data))
        if self.store_models:
            models = []
            self.models = [models]
        self.actual = data.Y.flatten()
        class_var = data.domain.class_var
        for i, learner in enumerate(learners):
            model = learner(data)
            if self.store_models:
                models.append(model)

            if is_discrete(class_var):
                values, probs = model(data, model.ValueProbs)
                self.predicted[i] = values
                self.probabilities[i] = probs
            elif is_continuous(class_var):
                values = model(data, model.Value)
                self.predicted[i] = values


class Bootstrap(Results):
    def __init__(self, data, learners, n_resamples=10, p=0.75, random_state=0,
                 store_data=False, store_models=False):
        super().__init__(data, len(learners), store_data=store_data,
                         store_models=store_models)
        self.store_models = store_models
        self.n_resamples = n_resamples
        self.p = p
        self.random_state = random_state

        indices = skl_cross_validation.Bootstrap(
            len(data), n_iter=self.n_resamples, train_size=self.p,
            random_state=self.random_state
        )

        self.folds = []
        if self.store_models:
            self.models = []

        row_indices = []
        actual = []
        predicted = [[] for _ in learners]
        probabilities = [[] for _ in learners]
        fold_start = 0
        class_var = data.domain.class_var
        for train, test in indices:
            train_data, test_data = data[train], data[test]
            self.folds.append(slice(fold_start, fold_start + len(test)))
            row_indices.append(test)
            actual.append(test_data.Y.flatten())
            if self.store_models:
                fold_models = []
                self.models.append(fold_models)

            for i, learner in enumerate(learners):
                model = learner(train_data)
                if self.store_models:
                    fold_models.append(model)

                if is_discrete(class_var):
                    values, probs = model(test_data, model.ValueProbs)
                    predicted[i].append(values)
                    probabilities[i].append(probs)
                elif is_continuous(class_var):
                    values = model(test_data, model.Value)
                    predicted[i].append(values)

            fold_start += len(test)

        row_indices = np.hstack(row_indices)
        actual = np.hstack(actual)
        predicted = np.array([np.hstack(pred) for pred in predicted])
        if is_discrete(class_var):
            probabilities = np.array([np.vstack(prob) for prob in probabilities])
        nrows = len(actual)
        nmodels = len(predicted)

        self.nrows = len(actual)
        self.row_indices = row_indices
        self.actual = actual
        self.predicted = predicted.reshape(nmodels, nrows)
        if is_discrete(class_var):
            self.probabilities = probabilities


class TestOnTestData(Results):
    """
    Test on a separate test data set.
    """
    def __init__(self, train_data, test_data, learners, store_data=False,
                 store_models=False):
        super().__init__(test_data, len(learners), store_data=store_data,
                         store_models=store_models)
        if self.store_models:
            self.models = []

        self.row_indices = np.arange(len(test_data))
        self.actual = test_data.Y.flatten()

        class_var = train_data.domain.class_var
        for i, learner in enumerate(learners):
            model = learner(train_data)
            if is_discrete(class_var):
                values, probs = model(test_data, model.ValueProbs)
                self.predicted[i] = values
                self.probabilities[i][:, :] = probs
            elif is_continuous(class_var):
                values = model(test_data, model.Value)
                self.predicted[i] = values

            if self.store_models:
                self.models.append(model)

        self.nrows = len(test_data)
        self.folds = [slice(0, len(test_data))]


def sample(table, n=0.7, stratified=False, replace=False,
                    random_state=None):
    """
    Samples data instances from a data table. Returns the sample and
    a data set from input data table that are not in the sample. Also
    uses several sampling functions from
    `scikit-learn <http://scikit-learn.org>`_.

    table : data table
        A data table from which to sample.

    n : float, int (default = 0.7)
        If float, should be between 0.0 and 1.0 and represents
        the proportion of data instances in the resulting sample. If
        int, n is the number of data instances in the resulting sample.

    stratified : bool, optional (default = False)
        If true, sampling will try to consider class values and
        match distribution of class values
        in train and test subsets.

    replace : bool, optional (default = False)
        sample with replacement

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.
    """

    if type(n) == float:
        n = int(n * len(table))

    if replace:
        if random_state is None:
            rgen = np.random
        else:
            rgen = np.random.mtrand.RandomState(random_state)
        sample = rgen.random_integers(0, len(table) - 1, n)
        o = np.ones(len(table))
        o[sample] = 0
        others = np.nonzero(o)[0]
        return table[sample], table[others]

    n = len(table) - n
    if stratified and is_discrete(table.domain.class_var):
        test_size = max(len(table.domain.class_var.values), n)
        ind = skl_cross_validation.StratifiedShuffleSplit(
            table.Y.ravel(), n_iter=1,
            test_size=test_size, train_size=len(table) - test_size,
            random_state=random_state)
    else:
        ind = skl_cross_validation.ShuffleSplit(
            len(table), n_iter=1,
            test_size=n, random_state=random_state)
    ind = next(iter(ind))
    return table[ind[0]], table[ind[1]]
