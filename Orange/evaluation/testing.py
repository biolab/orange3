# __init__ methods have all kinds of unused arguments to match signatures
# pylint: disable=unused-argument
# __new__ methods have different arguments
# pylint: disable=arguments-differ

from collections import namedtuple

import numpy as np

import sklearn.model_selection as skl

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable

__all__ = ["Results", "CrossValidation", "LeaveOneOut", "TestOnTrainingData",
           "ShuffleSplit", "TestOnTestData", "sample", "CrossValidationFeature"]

_MpResults = namedtuple('_MpResults', ('fold_i', 'learner_i', 'model',
                                       'failed', 'n_values', 'values', 'probs'))


def _identity(x):
    return x


def _mp_worker(fold_i, train_data, test_data, learner_i, learner,
               store_models):
    predicted, probs, model, failed = None, None, None, False
    try:
        if len(train_data) == 0 or len(test_data) == 0:
            raise RuntimeError('Test fold is empty')
        model = learner(train_data)
        if train_data.domain.has_discrete_class:
            predicted, probs = model(test_data, model.ValueProbs)
        elif train_data.domain.has_continuous_class:
            predicted = model(test_data, model.Value)
    # Different models can fail at any time raising any exception
    except Exception as ex:  # pylint: disable=broad-except
        failed = ex
    return _MpResults(fold_i, learner_i, store_models and model,
                      failed, len(test_data), predicted, probs)


class Results:
    """
    Class for storing predictions in model testing.

    Attributes:
        data (Optional[Table]):  Data used for testing.

        models (Optional[List[Model]]): A list of induced models.

        row_indices (np.ndarray): Indices of rows in `data` that were used in
            testing, stored as a numpy vector of length `nrows`.
            Values of `actual[i]`, `predicted[i]` and `probabilities[i]` refer
            to the target value of instance `data[row_indices[i]]`.

        nrows (int): The number of test instances (including duplicates).

        actual (np.ndarray): Actual values of target variable;
            a numpy vector of length `nrows` and of the same type as `data`
            (or `np.float32` if the type of data cannot be determined).

        predicted (np.ndarray): Predicted values of target variable;
            a numpy array of shape (number-of-methods, `nrows`) and
            of the same type as `data` (or `np.float32` if the type of data
            cannot be determined).

        probabilities (Optional[np.ndarray]): Predicted probabilities
            (for discrete target variables);
            a numpy array of shape (number-of-methods, `nrows`, number-of-classes)
            of type `np.float32`.

        folds (List[Slice or List[int]]): A list of indices (or slice objects)
            corresponding to rows of each fold.
    """
    def __init__(self, data=None, *,
                 nmethods=None, nrows=None, nclasses=None,
                 domain=None,
                 row_indices=None, folds=None, score_by_folds=True,
                 learners=None, models=None, failed=None,
                 actual=None, predicted=None, probabilities=None,
                 store_data=None, store_models=None):
        """
        Construct an instance.

        The constructor is a bit too smart to ensure backward compabitility.
        It consists of three steps.

        - Set any attributes specified directly through arguments.
        - Infer the domain, nmethods, nrows and nclasses from other data and/or
          check their overall consistency.
        - Prepare empty arrays `actual`, `predicted`, `probabilities` and
          `failed`. If not enough data is available, the corresponding arrays
          are `None`.

        Args:
            data (Orange.data.Table): stored data from which test was sampled
            nmethods (int): number of methods; can be inferred (or must match)
                the size of `learners`, `models`, `failed`, `predicted` and
                `probabilities`
            nrows (int): number of data instances; can be inferred (or must
                match) `data`, `row_indices`, `actual`, `predicted` and
                `probabilities`
            nclasses (int): number of class values (`None` if continuous); can
                be inferred (or must match) from `domain.class_var` or
                `probabilities`
            domain (Orange.data.Domain): data domain; can be inferred (or must)
                match `data.domain`
            row_indices (np.ndarray): see class documentation
            folds (np.ndarray): see class documentation
            score_by_folds (np.ndarray): see class documentation
            learners (np.ndarray): see class documentation
            models (np.ndarray): see class documentation
            failed (list of str): see class documentation
            actual (np.ndarray): see class documentation
            predicted (np.ndarray): see class documentation
            probabilities (np.ndarray): see class documentation
            store_data (bool): ignored; kept for backward compabitility
            store_models (bool): ignored; kept for backward compabitility
        """

        # Set given data directly from arguments
        self.data = data
        self.domain = domain

        self.row_indices = row_indices
        self.folds = folds
        self.score_by_folds = score_by_folds

        self.learners = learners
        self.models = models

        self.actual = actual
        self.predicted = predicted
        self.probabilities = probabilities
        self.failed = failed

        # Guess the rest -- or check for ambguities
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
        self.nrows = nrows = set_or_raise(
            nrows, [data is not None and len(data),
                    actual is not None and len(actual),
                    row_indices is not None and len(row_indices),
                    predicted is not None and predicted.shape[1],
                    probabilities is not None and probabilities.shape[1]],
            "mismatching number of rows")
        nclasses = set_or_raise(
            nclasses, [len(domain.class_var.values)
                       if domain is not None and domain.has_discrete_class
                       else None,
                       probabilities is not None and probabilities.shape[2]],
            "mismatching number of class values")
        if nclasses is None and probabilities is not None:
            raise ValueError("regression results cannot have 'probabilities'")
        nmethods = set_or_raise(
            nmethods, [learners is not None and len(learners),
                       models is not None and len(models),
                       failed is not None and len(failed),
                       predicted is not None and predicted.shape[0],
                       probabilities is not None and probabilities.shape[0]],
            "mismatching number of methods")

        # Prepare empty arrays
        if actual is None \
                and nrows is not None:
            self.actual = np.empty(nrows)

        if predicted is None \
                and nmethods is not None and nrows is not None:
            self.predicted = np.empty((nmethods, nrows))

        if probabilities is None \
                and nmethods is not None and nrows is not None \
                and nclasses is not None:
            self.probabilities = \
                np.empty((nmethods, nrows, nclasses))

        if failed is None \
                and nmethods is not None:
            self.failed = [False] * nmethods

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

    def get_augmented_data(self, model_names, include_attrs=True, include_predictions=True,
                           include_probabilities=True):
        """
        Return the data, augmented with predictions, probabilities (if the task is classification)
        and folds info. Predictions, probabilities and folds are inserted as meta attributes.

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
            new_meta_attr.append(
                DiscreteVariable(name="Fold",
                                 values=[str(i+1) for i, _ in enumerate(self.folds)]))
            fold = np.empty((len(data), 1))
            for i, s in enumerate(self.folds):
                fold[s, 0] = i
            new_meta_vals = np.hstack((new_meta_vals, fold))

        # append new columns to meta attributes
        new_meta_attr = list(data.domain.metas) + new_meta_attr
        new_meta_vals = np.hstack((data.metas, new_meta_vals))

        attrs = data.domain.attributes if include_attrs else []
        domain = Domain(attrs, data.domain.class_vars, metas=new_meta_attr)
        predictions = data.transform(domain)
        predictions.metas = new_meta_vals
        predictions.name = data.name
        return predictions

    def split_by_model(self):
        """Split evaluation results by models
        """
        data = self.data
        nmethods = len(self.predicted)
        for i in range(nmethods):
            res = Results()
            res.data = data
            res.domain = self.domain
            res.learners = [self.learners[i]]
            res.row_indices = self.row_indices
            res.actual = self.actual
            res.folds = self.folds
            res.score_by_folds = self.score_by_folds

            res.predicted = self.predicted[(i,), :]
            if getattr(self, "probabilities", None) is not None:
                res.probabilities = self.probabilities[(i,), :, :]

            if self.models is not None:
                res.models = self.models[:, i]

            res.failed = [self.failed[i]]
            yield res


class Validation:
    score_by_folds = False

    def __new__(cls,
                data=None, learners=None, preprocessor=None, test_data=None,
                *, callback=None, store_data=False, store_models=False,
                **kwargs):
        """
        Base class for different testing schemata such as cross validation and
        testing on separate data set.

        If the constructor is given data and learning algorithms, it
        automagically calls `fit` and returns `Results` instead of an
        instance of `Validation`.

        Args:
            data (Orange.data.Table): data to be used (usually split) into
                training and testing
            learners (list of Orange.Learner): a list of learning algorithms
            preprocessor (Orange.preprocess.Preprocess): preprocessor applied
                on training data
            test_data (Orange.data.Table): separate test data, if supported
                by the method; must be `None` otherwise
            callback (Callable): a function called to notify about the progress
            store_data (bool): a flag defining whether the data is stored
            store_models (bool): a flag defining whether the models are stored
        """
        self = super().__new__(cls)
        self.store_data = store_data
        self.store_models = store_models
        self.__dict__.update(kwargs)

        if learners is None != data is None:
            raise ValueError(
                "learners and train_data must both be present or not")
        if learners is None:
            if preprocessor is not None:
                raise ValueError("preprocessor cannot be given if learners "
                                 "and train_data are omitted")
            if test_data is not None:
                raise ValueError("test_data cannot be given if learners "
                                 "and train_data are omitted")
            if callback is not None:
                raise ValueError("callback cannot be given if learners "
                                 "and train_data are omitted")
            return self

        return self.fit(learners, preprocessor, data, test_data,
                        callback=callback)

    def fit(self, learners, preprocessor, data, test_data=None,
            *, callback=None):
        if preprocessor is None:
            preprocessor = _identity
        if callback is None:
            callback = _identity
        indices = self.get_indices(data, test_data)
        folds, row_indices, actual = \
            self.prepare_arrays(data, test_data, indices)
        stored_data = test_data or data
        results = Results(
            stored_data if self.store_data else None,
            domain=stored_data and stored_data.domain,
            nrows=len(row_indices), learners=learners,
            row_indices=row_indices, folds=folds, actual=actual,
            score_by_folds=self.score_by_folds)

        if test_data is None:
            test_data = data
        data_splits = (
            (fold_i, preprocessor(data[train_i]), test_data[test_i])
            for fold_i, (train_i, test_i) in enumerate(indices))
        args_iter = (
            (fold_i, data, test_data, learner_i, learner, self.store_models)
            for (fold_i, data, test_data) in data_splits
            for (learner_i, learner) in enumerate(learners))

        if self.store_models:
            results.models = np.tile(None, (len(indices), len(learners)))

        part_results = []
        parts = np.linspace(.0, .99, len(learners) * len(indices) + 1)[1:]
        for progress, part in zip(parts, args_iter):
            part_results.append(_mp_worker(*(part + ())))
            callback(progress)

        part_results = sorted(part_results)

        ptr, prev_fold_i, prev_n_values = 0, 0, 0
        for res in part_results:
            if res.fold_i != prev_fold_i:
                ptr += prev_n_values
                prev_fold_i = res.fold_i
            result_slice = slice(ptr, ptr + res.n_values)
            prev_n_values = res.n_values

            if res.failed:
                results.failed[res.learner_i] = res.failed
                continue

            if self.store_models:
                results.models[res.fold_i][res.learner_i] = res.model

            results.predicted[res.learner_i][result_slice] = res.values
            if data.domain.has_discrete_class:
                results.probabilities[res.learner_i][result_slice, :] = res.probs

        callback(1)
        return results

    @classmethod
    def prepare_arrays(cls, data, test_data, indices):
        """Prepare data for `fit` method."""
        if test_data is not None:
            raise ValueError(f"{cls.__name__} can't use separate test_data")

        folds = []
        row_indices = []

        ptr = 0
        for _, test in indices:
            folds.append(slice(ptr, ptr + len(test)))
            row_indices.append(test)
            ptr += len(test)

        row_indices = np.concatenate(row_indices, axis=0)
        actual = data[row_indices].Y.ravel()
        return folds, row_indices, actual

    @staticmethod
    def get_indices(data, test_data):
        """Initializes `self.indices` with iterable objects with slices
        (or indices) for each fold.

        Args:
            data (Table): data table
            test_data (Table): separate test table
        """
        raise NotImplementedError()


class CrossValidation(Validation):
    """
    K-fold cross validation.

    If the constructor is given the data and a list of learning algorithms, it
    runs cross validation and returns an instance of `Results` containing the
    predicted values and probabilities.

    .. attribute:: k

        The number of folds.

    .. attribute:: random_state

    """
    def __new__(cls, data, learners, k=10, stratified=True, random_state=0,
                store_data=False, store_models=False, preprocessor=None,
                callback=None, warnings=None, n_jobs=1):
        return super().__new__(
            cls,
            data, learners, preprocessor=preprocessor, callback=callback,
            k=k, stratified=stratified, random_state=random_state,
            store_data=store_data, store_models=store_models,
            warnings=[] if warnings is None else warnings)

    # __init__ will be called only if __new__ doesn't have data and learners
    # Also, attributes are already set in __new__; __init__ is here only so that
    # IDE's and pylint know about attributes
    def __init__(self, data, learners, k=10, stratified=True, random_state=0,
                 store_data=False, store_models=False, preprocessor=None,
                 callback=None, warnings=None, n_jobs=1):
        super().__init__(store_data=store_data, store_models=store_models)
        self.k = k
        self.stratified = stratified
        self.random_state = random_state

    def get_indices(self, data, test_data):
        if self.stratified and data.domain.has_discrete_class:
            try:
                splitter = skl.StratifiedKFold(
                    self.k, shuffle=True, random_state=self.random_state
                )
                splitter.get_n_splits(data.X, data.Y)
                return list(splitter.split(data.X, data.Y))
            except ValueError:
                self.warnings.append("Using non-stratified sampling.")

        splitter = skl.KFold(
            self.k, shuffle=True, random_state=self.random_state)
        splitter.get_n_splits(data)
        return list(splitter.split(data))


class CrossValidationFeature(Validation):
    """
    Cross validation with folds according to values of a feature.

    .. attribute:: feature

        The feature defining the folds.

    """
    def __new__(cls, data, learners, feature,
                store_data=False, store_models=False, preprocessor=None,
                callback=None, warnings=None, n_jobs=1):
        return super().__new__(
            cls,
            data, learners, preprocessor=preprocessor, callback=callback,
            feature=feature,
            store_data=store_data, store_models=store_models,
            warnings=warnings or [])

    # __init__ will be called only if __new__ doesn't have data and learners
    # Also, attributes are already set in __new__; __init__ is here only so that
    # IDE's and pylint know about attributes
    def __init__(self, data, learners, feature,
                 store_data=False, store_models=False, preprocessor=None,
                 callback=None, warnings=None, n_jobs=1):
        super().__init__(store_data=store_data, store_models=store_models)
        self.feature = feature

    def get_indices(self, data, _test_data):
        data = Table(Domain([self.feature], None), data)
        values = data[:, self.feature].X
        indices = []
        for v in range(len(self.feature.values)):
            test_index = np.where(values == v)[0]
            train_index = np.where((values != v) & (~np.isnan(values)))[0]
            if len(test_index) and len(train_index):
                indices.append((train_index, test_index))
        if not indices:
            raise ValueError("No folds could be created from the given feature.")
        return indices


class LeaveOneOut(Validation):
    """Leave-one-out testing"""
    score_by_folds = False

    def get_indices(self, data, test_data):
        if test_data is not None:
            raise ValueError("Leave one out can't use separate test_data")

        splitter = skl.LeaveOneOut()
        splitter.get_n_splits(data)
        return list(splitter.split(data))

    def prepare_arrays(self, data, test_data, indices):
        # sped up version of super().prepare_arrays(data)
        if test_data is not None:
            raise ValueError("Leave one out can't use separate test_data")
        row_indices = np.arange(len(data))
        return row_indices, row_indices, data.Y.flatten()


class ShuffleSplit(Validation):
    def __new__(cls, data, learners,
                n_resamples=10, train_size=None, test_size=0.1,
                stratified=True, random_state=0,
                store_data=False, store_models=False, preprocessor=None,
                callback=None, n_jobs=1):
        return super().__new__(
            cls,
            data, learners, preprocessor=preprocessor, callback=callback,
            n_resamples=n_resamples, train_size=train_size, test_size=test_size,
            stratified=stratified, random_state=random_state,
            store_data=store_data, store_models=store_models)

    # __init__ will be called only if __new__ doesn't have data and learners
    # Also, attributes are already set in __new__; __init__ is here only so that
    # IDE's and pylint know about attributes
    def __init__(self, data, learners,
                 n_resamples=10, train_size=None, test_size=0.1,
                 stratified=True, random_state=0,
                 store_data=False, store_models=False, preprocessor=None,
                 callback=None, n_jobs=1):
        super().__init__(store_data=store_data, store_models=store_models)
        self.n_resamples = n_resamples
        self.train_size = train_size
        self.test_size = test_size
        self.stratified = stratified
        self.random_state = random_state

    def get_indices(self, data, test_data):
        if test_data is not None:
            raise ValueError("Shuffle split can't use separate test_data")

        if self.stratified and data.domain.has_discrete_class:
            splitter = skl.StratifiedShuffleSplit(
                n_splits=self.n_resamples, train_size=self.train_size,
                test_size=self.test_size, random_state=self.random_state
            )
            splitter.get_n_splits(data.X, data.Y)
            return list(splitter.split(data.X, data.Y))

        splitter = skl.ShuffleSplit(
            n_splits=self.n_resamples, train_size=self.train_size,
            test_size=self.test_size, random_state=self.random_state
        )
        splitter.get_n_splits(data)
        return list(splitter.split(data))


class TestOnTestData(Validation):
    def __new__(cls, data=None, *args, **kwargs):
        # Backward compatibility. Don't ask.
        # Old signature didn't match that of the super class
        if args and isinstance(args[0], Table):
            test_data, learners, preprocessor, *_ = args + (None, ) * 3
        else:
            learners, preprocessor, test_data, *_ = args + (None, ) * 3
        if "train_data" in kwargs:
            if data is None:
                data = kwargs.pop("train_data")
            else:
                raise TypeError("argument 'data' is given twice")
        elif data is None:
            raise TypeError("missing required argument 'data'")
        if "test_data" in kwargs:
            test_data = kwargs.pop("test_data")
        if "preprocessor" in kwargs:
            preprocessor = kwargs.pop("preprocessor")
        if "learners" in kwargs:
            learners = kwargs.pop("learners")
        return super().__new__(
            cls,
            data, learners, preprocessor=preprocessor, test_data=test_data,
            **kwargs)

    def get_indices(self, _train_data, _test_data):
        return ((Ellipsis, Ellipsis),)

    def prepare_arrays(self, data, test_data, indices):
        return (Ellipsis, ), np.arange(len(test_data)), test_data.Y.ravel()


class TestOnTrainingData(TestOnTestData):
    def __new__(cls,
                data=None, learners=None, preprocessor=None, **kwargs):
        if preprocessor is not None:
            data = preprocessor(data)
        return super().__new__(
            cls, data, learners, preprocessor=None, test_data=data, **kwargs)


def sample(table, n=0.7, stratified=False, replace=False,
           random_state=None):
    """
    Samples data instances from a data table. Returns the sample and
    a dataset from input data table that are not in the sample. Also
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

    if isinstance(n, float):
        n = int(n * len(table))

    if replace:
        if random_state is None:
            rgen = np.random
        else:
            rgen = np.random.mtrand.RandomState(random_state)
        sample = rgen.randint(0, len(table), n)
        o = np.ones(len(table))
        o[sample] = 0
        others = np.nonzero(o)[0]
        return table[sample], table[others]

    n = len(table) - n
    if stratified and table.domain.has_discrete_class:
        test_size = max(len(table.domain.class_var.values), n)
        splitter = skl.StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, train_size=len(table) - test_size,
            random_state=random_state)
        splitter.get_n_splits(table.X, table.Y)
        ind = splitter.split(table.X, table.Y)
    else:
        splitter = skl.ShuffleSplit(
            n_splits=1, test_size=n, random_state=random_state)
        splitter.get_n_splits(table)
        ind = splitter.split(table)
    ind = next(ind)
    return table[ind[0]], table[ind[1]]
