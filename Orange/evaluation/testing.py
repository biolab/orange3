import numpy as np

import sklearn.cross_validation as skl_cross_validation

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable

__all__ = ["Results", "CrossValidation", "LeaveOneOut", "TestOnTrainingData",
           "ShuffleSplit", "TestOnTestData", "sample"]


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
                 predicted=None, probabilities=None,
                 preprocessor=None, callback=None):
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
        :param preprocessor: Preprocessor for training data
        :type preprocessor: Orange.preprocess.Preprocess
        :param callback: Function for reporting back the progress as a value
            between 0 and 1
        :type callback: callable
        """
        self.store_data = store_data
        self.store_models = store_models
        self.data = data if store_data else None
        self.models = None
        self.folds = None
        self.indices = []
        self.learners = []
        self.row_indices = row_indices
        dtype = np.float32
        self.preprocessor = preprocessor
        self.callback = callback

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
            nclasses, [domain and (len(domain.class_var.values)
                                   if domain.has_discrete_class
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

        if nmethods is not None:
            self.failed = [False] * nmethods

        if actual is not None:
            self.actual = actual
        elif nrows is not None:
            self.actual = np.empty(nrows, dtype=dtype)

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

    def train_if_succ(self, learner_index, learner, data):
        if self.failed[learner_index]:
            return False
        try:
            return learner(data)
        except Exception as ex:
            self.failed[learner_index] = ex
            return False

    def call_callback(self, progress):
        if self.callback:
            self.callback(progress)

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

    def get_augmented_data(self, model_names, include_attrs=True, include_predictions=True, include_probabilities=True):
        """
        Return the data, augmented with predictions, probabilities (if the task is classification) and folds info.
        Predictions, probabilities and folds are inserted as meta attributes.

        Args:
            model_names (list): A list of strings containing learners' names.
            include_attrs (bool): Flag that tells whether to include original attributes.
            include_predictions (bool): Flag that tells whether to include predictions.
            include_probabilities (bool): Flag that tells whether to include probabilities.

        Returns:
            Orange.data.Table: Data augmented with predictions, (probabilities) and (fold).

        """
        assert self.predicted.shape[0] == len(model_names)

        data = self.data[self.row_indices]
        class_var = data.domain.class_var
        classification = class_var and class_var.is_discrete

        new_meta_attr = []
        new_meta_vals = np.empty((len(data), 0))

        if classification:
            # predictions
            if include_predictions:
                new_meta_attr.extend(DiscreteVariable(name=name, values=class_var.values)
                                     for name in model_names)
                new_meta_vals = np.hstack((new_meta_vals, self.predicted.T))

            # probabilities
            if include_probabilities:
                for name in model_names:
                    new_meta_attr.extend(ContinuousVariable(name="%s (%s)" % (name, value))
                                         for value in class_var.values)

                for i in self.probabilities:
                    new_meta_vals = np.hstack((new_meta_vals, i))

        elif include_predictions:
            # regression
            new_meta_attr.extend(ContinuousVariable(name=name)
                                 for name in model_names)
            new_meta_vals = np.hstack((new_meta_vals, self.predicted.T))

        # add fold info
        if self.folds is not None:
            new_meta_attr.append(DiscreteVariable(name="Fold", values=[i+1 for i, s in enumerate(self.folds)]))
            fold = np.empty((len(data), 1))
            for i, s in enumerate(self.folds):
                fold[s, 0] = i
            new_meta_vals = np.hstack((new_meta_vals, fold))

        # append new columns to meta attributes
        new_meta_attr = list(data.domain.metas) + new_meta_attr
        new_meta_vals = np.hstack((data.metas, new_meta_vals))

        X = data.X if include_attrs else np.empty((len(data), 0))
        attrs = data.domain.attributes if include_attrs else []

        domain = Domain(attrs, data.domain.class_vars, metas=new_meta_attr)
        predictions = Table.from_numpy(domain, X, data.Y, metas=new_meta_vals)
        predictions.name = data.name
        return predictions

    def fit(self, data):
        n_callbacks = len(self.learners) * len(self.indices)
        if self.store_models:
            self.models = [[None] * len(self.learners)] * len(self.indices)

        ptr = 0
        for idx, (train_indices, test_indices) in enumerate(self.indices):
            train_data = data[train_indices]
            test_data = data[test_indices]

            if self.preprocessor is not None:
                train_data = self.preprocessor(train_data)

            for k, learner in enumerate(self.learners):
                model = self.train_if_succ(k, learner, train_data)
                self.call_callback((len(self.learners) * idx + k) / n_callbacks)
                if not model:
                    continue

                if self.store_models:
                    self.models[idx][k] = model

                result_slice = slice(ptr, ptr + len(test_indices))

                if data.domain.has_discrete_class:
                    values, probs = model(test_data, model.ValueProbs)
                    self.predicted[k][result_slice] = values
                    self.probabilities[k][result_slice, :] = probs
                elif data.domain.has_continuous_class:
                    values = model(test_data, model.Value)
                    self.predicted[k][result_slice] = values
                else:
                    raise Results.DataError("Unknown target type")
            ptr += len(test_indices)

        self.call_callback(1)

    def prepare_arrays(self, data):
        self.folds = []
        row_indices = []

        ptr = 0
        for fold_idx, (train, test) in enumerate(self.indices):
            self.folds.append(slice(ptr, ptr + len(test)))
            row_indices.append(test)
            ptr += len(test)

        self.row_indices = np.concatenate(row_indices, axis=0)
        self.actual = data[self.row_indices].Y.flatten()
        self.predicted = np.empty((len(self.learners), len(self.row_indices)))

        if data.domain.has_discrete_class:
            self.probabilities = np.empty((len(self.learners),
                                           len(self.row_indices),
                                           len(data.domain.class_var.values)))
        else:
            self.probabilities = None

    class DataError(Exception):
        pass


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
    def __init__(self, data, learners, k=10, stratified=True, random_state=0, store_data=False,
                 store_models=False, preprocessor=None, callback=None, warnings=None):
        super().__init__(data, len(learners), store_data=store_data,
                         store_models=store_models, preprocessor=preprocessor,
                         callback=callback)
        self.folds = []
        self.learners = learners
        self.k = k
        self.stratified = stratified
        self.random_state = random_state

        self.indices = None
        if self.stratified and data.domain.has_discrete_class:
            self.indices = skl_cross_validation.StratifiedKFold(
                data.Y, self.k, shuffle=True, random_state=self.random_state
            )
            if any(len(train) == 0 or len(test) == 0 for train, test in self.indices):
                if warnings is not None:
                    warnings.append("Using non-stratified sampling.")
                self.indices = None
        if self.indices is None:
            self.indices = skl_cross_validation.KFold(
                len(data), self.k, shuffle=True, random_state=self.random_state
            )
        self.prepare_arrays(data)
        self.fit(data)


class LeaveOneOut(Results):
    """Leave-one-out testing"""

    def __init__(self, data, learners, store_data=False, store_models=False,
                 preprocessor=None, callback=None):
        super().__init__(data, learners=learners, store_data=store_data,
                         store_models=store_models, preprocessor=preprocessor,
                         callback=callback)
        self.indices = skl_cross_validation.LeaveOneOut(len(data))

        self.prepare_arrays(data)
        self.fit(data)

    def prepare_arrays(self, data):
        self.row_indices = np.arange(len(data))
        self.actual = data.Y.flatten()


class ShuffleSplit(Results):
    def __init__(self, data, learners, n_resamples=10, train_size=None,
                 test_size=0.1, stratified=True, random_state=0, store_data=False,
                 store_models=False, preprocessor=None, callback=None):
        super().__init__(data, len(learners), store_data=store_data,
                         store_models=store_models, preprocessor=preprocessor,
                         callback=callback)
        self.learners = learners
        self.n_resamples = n_resamples
        self.train_size = train_size
        self.test_size = test_size
        self.stratified = stratified
        self.random_state = random_state

        if self.stratified and data.domain.has_discrete_class:
            self.indices = skl_cross_validation.StratifiedShuffleSplit(
                data.Y, n_iter=self.n_resamples, train_size=self.train_size,
                test_size=self.test_size, random_state=self.random_state
            )
        else:
            self.indices = skl_cross_validation.ShuffleSplit(
                len(data), n_iter=self.n_resamples, train_size=self.train_size,
                test_size=self.test_size, random_state=self.random_state
            )

        self.prepare_arrays(data)
        self.fit(data)


class TestOnTestData(Results):
    """
    Test on a separate test data set.
    """
    def __init__(self, train_data, test_data, learners, store_data=False,
                 store_models=False, preprocessor=None, callback=None):
        super().__init__(test_data, len(learners), store_data=store_data,
                         store_models=store_models, preprocessor=preprocessor,
                         callback=callback)
        nmethods = len(learners)
        if self.store_models:
            models = [None] * nmethods
            self.models = [models]

        self.row_indices = np.arange(len(test_data))
        self.actual = test_data.Y.flatten()

        if self.preprocessor is not None:
            train_data = self.preprocessor(train_data)
        for i, learner in enumerate(learners):
            model = self.train_if_succ(i, learner, train_data)
            self.call_callback(i / nmethods)
            if not model:
                continue
            if train_data.domain.has_discrete_class:
                values, probs = model(test_data, model.ValueProbs)
                self.predicted[i] = values
                self.probabilities[i][:, :] = probs
            elif train_data.domain.has_continuous_class:
                values = model(test_data, model.Value)
                self.predicted[i] = values
            if self.store_models:
                models[i] = model

        self.nrows = len(test_data)
        self.folds = [slice(0, len(test_data))]
        self.call_callback(1)


class TestOnTrainingData(TestOnTestData):
    """
    Trains and test on the same data
    """

    def __init__(self, data, learners, store_data=False, store_models=False,
                 preprocessor=None, callback=None):

        if preprocessor is not None:
            data = preprocessor(data)

        super().__init__(train_data=data, test_data=data, learners=learners,
                         store_data=store_data, store_models=store_models,
                         preprocessor=None, callback=callback)


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
    if stratified and table.domain.has_discrete_class:
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
