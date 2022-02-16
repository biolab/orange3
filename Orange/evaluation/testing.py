# __new__ methods have different arguments
# pylint: disable=arguments-differ
from warnings import warn
from collections import namedtuple
from itertools import chain
from time import time

import numpy as np

import sklearn.model_selection as skl

from Orange.data import Domain, ContinuousVariable, DiscreteVariable
from Orange.data.util import get_unique_names

__all__ = ["Results", "CrossValidation", "LeaveOneOut", "TestOnTrainingData",
           "ShuffleSplit", "TestOnTestData", "sample", "CrossValidationFeature"]

_MpResults = namedtuple('_MpResults', ('fold_i', 'learner_i', 'model',
                                       'failed', 'n_values', 'values',
                                       'probs', 'train_time', 'test_time'))


def _identity(x):
    return x


def _mp_worker(fold_i, train_data, test_data, learner_i, learner,
               store_models):
    predicted, probs, model, failed = None, None, None, False
    train_time, test_time = None, None
    try:
        if not train_data or not test_data:
            raise RuntimeError('Test fold is empty')
        # training
        t0 = time()
        model = learner(train_data)
        train_time = time() - t0
        t0 = time()
        # testing
        class_var = train_data.domain.class_var
        if class_var and class_var.is_discrete:
            predicted, probs = model(test_data, model.ValueProbs)
        else:
            predicted = model(test_data, model.Value)
        test_time = time() - t0
    # Different models can fail at any time raising any exception
    except Exception as ex:  # pylint: disable=broad-except
        failed = ex
    return _MpResults(fold_i, learner_i, store_models and model,
                      failed, len(test_data), predicted, probs,
                      train_time, test_time)


class Results:
    """
    Class for storing predictions in model testing.

    Attributes:
        data (Optional[Table]):  Data used for testing.

        models (Optional[List[Model]]): A list of induced models.

        row_indices (np.ndarray): Indices of rows in `data` that were used in
            testing, stored as a numpy vector of length `nrows`.
            Values of `actual[i]`, `predicted[i]` and `probabilities[i]` refer
            to the target value of instance, that is, the i-th test instance
            is `data[row_indices[i]]`, its actual class is `actual[i]`, and
            the prediction by m-th method is `predicted[m, i]`.

        nrows (int): The number of test instances (including duplicates);
            `nrows` equals the length of `row_indices` and `actual`, and the
            second dimension of `predicted` and `probabilities`.

        actual (np.ndarray): true values of target variable in a vector of
            length `nrows`.

        predicted (np.ndarray): predicted values of target variable in an array
            of shape (number-of-methods, `nrows`)

        probabilities (Optional[np.ndarray]): predicted probabilities
            (for discrete target variables) in an array of shape
            (number-of-methods, `nrows`, number-of-classes)

        folds (List[Slice or List[int]]): a list of indices (or slice objects)
            corresponding to testing data subsets, that is,
            `row_indices[folds[i]]` contains row indices used in fold i, so
            `data[row_indices[folds[i]]]` is the corresponding testing data

        train_time (np.ndarray): training times of batches

        test_time (np.ndarray): testing times of batches
    """
    def __init__(self, data=None, *,
                 nmethods=None, nrows=None, nclasses=None,
                 domain=None,
                 row_indices=None, folds=None, score_by_folds=True,
                 learners=None, models=None, failed=None,
                 actual=None, predicted=None, probabilities=None,
                 store_data=None, store_models=None,
                 train_time=None, test_time=None):
        """
        Construct an instance.

        The constructor stores the given data, and creates empty arrays
        `actual`, `predicted` and `probabilities` if ther are not given but
        sufficient data is provided to deduct their shapes.

        The function

        - set any attributes specified directly through arguments.
        - infers the number of methods, rows and classes from other data
           and/or check their overall consistency.
        - Prepare empty arrays `actual`, `predicted`, `probabilities` and
          `failed` if the are not given. If not enough data is available,
          the corresponding arrays are `None`.

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
            store_data (bool): ignored; kept for backward compatibility
            store_models (bool): ignored; kept for backward compatibility
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

        self.train_time = train_time
        self.test_time = test_time

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
            nrows, [actual is not None and len(actual),
                    row_indices is not None and len(row_indices),
                    predicted is not None and predicted.shape[1],
                    probabilities is not None and probabilities.shape[1]],
            "mismatching number of rows")
        if domain is not None and domain.has_continuous_class:
            if nclasses is not None:
                raise ValueError(
                    "regression results cannot have non-None 'nclasses'")
            if probabilities is not None:
                raise ValueError(
                    "regression results cannot have 'probabilities'")
        nclasses = set_or_raise(
            nclasses, [domain is not None and domain.has_discrete_class and
                       len(domain.class_var.values),
                       probabilities is not None and probabilities.shape[2]],
            "mismatching number of class values")
        nmethods = set_or_raise(
            nmethods, [learners is not None and len(learners),
                       models is not None and models.shape[1],
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

    def get_augmented_data(self, model_names,
                           include_attrs=True, include_predictions=True,
                           include_probabilities=True):
        """
        Return the test data table augmented with meta attributes containing
        predictions, probabilities (if the task is classification) and fold
        indices.

        Args:
            model_names (list of str): names of models
            include_attrs (bool):
                if set to `False`, original attributes are removed
            include_predictions (bool):
                if set to `False`, predictions are not added
            include_probabilities (bool):
                if set to `False`, probabilities are not added

        Returns:
            augmented_data (Orange.data.Table):
                data augmented with predictions, probabilities and fold indices

        """
        assert self.predicted.shape[0] == len(model_names)

        data = self.data[self.row_indices]
        domain = data.domain
        class_var = domain.class_var
        classification = class_var and class_var.is_discrete

        new_meta_attr = []
        new_meta_vals = np.empty((len(data), 0))
        names = [var.name for var in chain(domain.attributes,
                                           domain.metas,
                                           domain.class_vars)]

        if classification:
            # predictions
            if include_predictions:
                uniq_new, names = self.create_unique_vars(names, model_names, class_var.values)
                new_meta_attr += uniq_new
                new_meta_vals = np.hstack((new_meta_vals, self.predicted.T))

            # probabilities
            if include_probabilities:
                proposed = [f"{name} ({value})" for name in model_names for value in class_var.values]

                uniq_new, names = self.create_unique_vars(names, proposed)
                new_meta_attr += uniq_new

                for i in self.probabilities:
                    new_meta_vals = np.hstack((new_meta_vals, i))

        elif include_predictions:
            # regression
            uniq_new, names = self.create_unique_vars(names, model_names)
            new_meta_attr += uniq_new
            new_meta_vals = np.hstack((new_meta_vals, self.predicted.T))

        # add fold info
        if self.folds is not None:
            values = [str(i + 1) for i in range(len(self.folds))]
            uniq_new, names = self.create_unique_vars(names, ["Fold"], values)
            new_meta_attr += uniq_new
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
        with predictions.unlocked(predictions.metas):
            predictions.metas = new_meta_vals
        predictions.name = data.name
        return predictions

    def create_unique_vars(self, names, proposed_names, values=()):
        unique_vars = []
        for proposed in proposed_names:
            uniq = get_unique_names(names, proposed)
            if values:
                unique_vars.append(DiscreteVariable(uniq, values))
            else:
                unique_vars.append(ContinuousVariable(uniq))
            names.append(uniq)
        return unique_vars, names

    def split_by_model(self):
        """
        Split evaluation results by models.

        The method generates instances of `Results` containing data for single
        models
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
            res.test_time = self.test_time[i]
            res.train_time = self.train_time[i]

            res.predicted = self.predicted[(i,), :]
            if getattr(self, "probabilities", None) is not None:
                res.probabilities = self.probabilities[(i,), :, :]

            if self.models is not None:
                res.models = self.models[:, i:i + 1]

            res.failed = [self.failed[i]]
            yield res


class Validation:
    """
    Base class for different testing schemata such as cross validation and
    testing on separate data set.

    If `data` is some data table and `learners` is a list of learning
    algorithms. This will run 5-fold cross validation and store the results
    in `res`.

        cv = CrossValidation(k=5)
        res = cv(data, learners)

    If constructor was given data and learning algorithms (as in
    `res = CrossValidation(data, learners, k=5)`, it used to automagically
    call the instance after constructing it and return `Results` instead
    of an instance of `Validation`. This functionality
    is deprecated and will be removed in the future.

    Attributes:
        store_data (bool): a flag defining whether the data is stored
        store_models (bool): a flag defining whether the models are stored
    """
    score_by_folds = False

    def __new__(cls,
                data=None, learners=None, preprocessor=None, test_data=None,
                *, callback=None, store_data=False, store_models=False,
                n_jobs=None, **kwargs):
        self = super().__new__(cls)

        if (learners is None) != (data is None):
            raise ValueError(
                "learners and train_data must both be present or not")
        if learners is None:
            if preprocessor is not None:
                raise ValueError("preprocessor cannot be given if learners "
                                 "and train_data are omitted")
            if callback is not None:
                raise ValueError("callback cannot be given if learners "
                                 "and train_data are omitted")
            return self

        warn("calling Validation's constructor with data and learners "
             "is deprecated;\nconstruct an instance and call it",
             DeprecationWarning, stacklevel=2)

        # Explicitly call __init__ because Python won't
        self.__init__(store_data=store_data, store_models=store_models,
                      **kwargs)
        if test_data is not None:
            test_data_kwargs = {"test_data": test_data}
        else:
            test_data_kwargs = {}
        return self(data, learners=learners, preprocessor=preprocessor,
                    callback=callback, **test_data_kwargs)

    # Note: this will be called only if __new__ doesn't have data and learners
    def __init__(self, *, store_data=False, store_models=False):
        self.store_data = store_data
        self.store_models = store_models

    def fit(self, *args, **kwargs):
        warn("Validation.fit is deprecated; use the call operator",
             DeprecationWarning)
        return self(*args, **kwargs)

    def __call__(self, data, learners, preprocessor=None, *, callback=None):
        """
        Args:
            data (Orange.data.Table): data to be used (usually split) into
                training and testing
            learners (list of Orange.Learner): a list of learning algorithms
            preprocessor (Orange.preprocess.Preprocess): preprocessor applied
                on training data
            callback (Callable): a function called to notify about the progress

        Returns:
            results (Result): results of testing
        """
        if preprocessor is None:
            preprocessor = _identity
        if callback is None:
            callback = _identity
        indices = self.get_indices(data)
        folds, row_indices, actual = self.prepare_arrays(data, indices)

        data_splits = (
            (fold_i, preprocessor(data[train_i]), data[test_i])
            for fold_i, (train_i, test_i) in enumerate(indices))
        args_iter = (
            (fold_i, data, test_data, learner_i, learner, self.store_models)
            for (fold_i, data, test_data) in data_splits
            for (learner_i, learner) in enumerate(learners))

        part_results = []
        parts = np.linspace(.0, .99, len(learners) * len(indices) + 1)[1:]
        for progress, part in zip(parts, args_iter):
            part_results.append(_mp_worker(*(part + ())))
            callback(progress)
        callback(1)

        results = Results(
            data=data if self.store_data else None,
            domain=data.domain,
            nrows=len(row_indices), learners=learners,
            row_indices=row_indices, folds=folds, actual=actual,
            score_by_folds=self.score_by_folds,
            train_time=np.zeros((len(learners),)),
            test_time=np.zeros((len(learners),)))

        if self.store_models:
            results.models = np.tile(None, (len(indices), len(learners)))
        self._collect_part_results(results, part_results)
        return results

    @classmethod
    def prepare_arrays(cls, data, indices):
        """Prepare `folds`, `row_indices` and `actual`.

        The method is used by `__call__`. While functional, it may be
        overriden in subclasses for speed-ups.

        Args:
            data (Orange.data.Table): data use for testing
            indices (list of vectors):
                indices of data instances in each test sample

        Returns:
            folds: (np.ndarray): see class documentation
            row_indices: (np.ndarray): see class documentation
            actual: (np.ndarray): see class documentation
        """
        folds = []
        row_indices = []

        ptr = 0
        for _, test in indices:
            folds.append(slice(ptr, ptr + len(test)))
            row_indices.append(test)
            ptr += len(test)

        row_indices = np.concatenate(row_indices, axis=0)
        return folds, row_indices, data[row_indices].Y

    @staticmethod
    def get_indices(data):
        """
        Return a list of arrays of indices of test data instance

        For example, in k-fold CV, the result is a list with `k` elements,
        each containing approximately `len(data) / k` nonoverlapping indices
        into `data`.

        This method is abstract and must be implemented in derived classes
        unless they provide their own implementation of the `__call__`
        method.

        Args:
            data (Orange.data.Table): test data

        Returns:
            indices (list of np.ndarray):
                a list of arrays of indices into `data`
        """
        raise NotImplementedError()

    def _collect_part_results(self, results, part_results):
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
            results.train_time[res.learner_i] += res.train_time
            results.test_time[res.learner_i] += res.test_time
            if res.probs is not None:
                results.probabilities[res.learner_i][result_slice, :] = \
                    res.probs


class CrossValidation(Validation):
    """
    K-fold cross validation

    Attributes:
        k (int): number of folds (default: 10)
        random_state (int):
            seed for random number generator (default: 0). If set to `None`,
            a different seed is used each time
        stratified (bool):
            flag deciding whether to perform stratified cross-validation.
            If `True` but the class sizes don't allow it, it uses non-stratified
            validataion and adds a list `warning` with a warning message(s) to
            the `Result`.
    """
    # TODO: list `warning` contains just repetitions of the same message
    #       replace with a flag in `Results`?
    def __init__(self, k=10, stratified=True, random_state=0,
                 store_data=False, store_models=False, warnings=None):
        super().__init__(store_data=store_data, store_models=store_models)
        self.k = k
        self.stratified = stratified
        self.random_state = random_state
        self.warnings = [] if warnings is None else warnings

    def get_indices(self, data):
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

    Attributes:
        feature (Orange.data.Variable): the feature defining the folds
    """
    def __init__(self, feature=None,
                 store_data=False, store_models=False, warnings=None):
        super().__init__(store_data=store_data, store_models=store_models)
        self.feature = feature

    def get_indices(self, data):
        data = data.transform(Domain([self.feature], None))
        values = data[:, self.feature].X
        indices = []
        for v in range(len(self.feature.values)):
            test_index = np.where(values == v)[0]
            train_index = np.where((values != v) & (~np.isnan(values)))[0]
            if test_index.size and train_index.size:
                indices.append((train_index, test_index))
        if not indices:
            raise ValueError(
                f"'{self.feature.name}' does not have at least two distinct "
                "values on the data")
        return indices


class LeaveOneOut(Validation):
    """Leave-one-out testing"""
    score_by_folds = False

    def get_indices(self, data):
        splitter = skl.LeaveOneOut()
        splitter.get_n_splits(data)
        return list(splitter.split(data))

    @staticmethod
    def prepare_arrays(data, indices):
        # sped up version of super().prepare_arrays(data)
        row_indices = np.arange(len(data))
        return row_indices, row_indices, data.Y.flatten()


class ShuffleSplit(Validation):
    """
    Test by repeated random sampling

    Attributes:
        n_resamples (int): number of repetitions
        test_size (float, int, None):
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size. By default, the value is set to 0.1.
            The default will change in version 0.21. It will remain 0.1 only
            if ``train_size`` is unspecified, otherwise it will complement
            the specified ``train_size``.
            (from documentation of scipy.sklearn.StratifiedShuffleSplit)

        train_size : float, int, or None, default is None
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. If
            int, represents the absolute number of train samples. If None,
            the value is automatically set to the complement of the test size.
            (from documentation of scipy.sklearn.StratifiedShuffleSplit)

        stratified (bool):
            flag deciding whether to perform stratified cross-validation.

        random_state (int):
            seed for random number generator (default: 0). If set to `None`,
            a different seed is used each time

    """
    def __init__(self, n_resamples=10, train_size=None, test_size=0.1,
                 stratified=True, random_state=0,
                 store_data=False, store_models=False):
        super().__init__(store_data=store_data, store_models=store_models)
        self.n_resamples = n_resamples
        self.train_size = train_size
        self.test_size = test_size
        self.stratified = stratified
        self.random_state = random_state

    def get_indices(self, data):
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
    """
    Test on separately provided test data

    Note that the class has a different signature for `__call__`.
    """
    # get_indices is not needed in this class, pylint: disable=abstract-method

    def __new__(cls, data=None, test_data=None, learners=None,
                preprocessor=None, **kwargs):
        if "train_data" in kwargs:
            if data is None:
                data = kwargs.pop("train_data")
            else:
                raise ValueError(
                    "argument 'data' is given twice (once as 'train_data')")
        return super().__new__(
            cls,
            data=data, learners=learners, preprocessor=preprocessor,
            test_data=test_data, **kwargs)

    def __call__(self, data, test_data, learners, preprocessor=None,
                 *, callback=None):
        """
        Args:
            data (Orange.data.Table): training data
            test_data (Orange.data.Table): test_data
            learners (list of Orange.Learner): a list of learning algorithms
            preprocessor (Orange.preprocess.Preprocess): preprocessor applied
                on training data
            callback (Callable): a function called to notify about the progress

        Returns:
            results (Result): results of testing
        """
        if preprocessor is None:
            preprocessor = _identity
        if callback is None:
            callback = _identity

        train_data = preprocessor(data)
        part_results = []
        for (learner_i, learner) in enumerate(learners):
            part_results.append(
                _mp_worker(0, train_data, test_data, learner_i, learner,
                           self.store_models))
            callback((learner_i + 1) / len(learners))
        callback(1)

        results = Results(
            data=test_data if self.store_data else None,
            domain=test_data.domain,
            nrows=len(test_data), learners=learners,
            row_indices=np.arange(len(test_data)),
            folds=(Ellipsis, ),
            actual=test_data.Y,
            score_by_folds=self.score_by_folds,
            train_time=np.zeros((len(learners),)),
            test_time=np.zeros((len(learners),)))

        if self.store_models:
            results.models = np.tile(None, (1, len(learners)))
        self._collect_part_results(results, part_results)
        return results


class TestOnTrainingData(TestOnTestData):
    """Test on training data"""
    # get_indices is not needed in this class, pylint: disable=abstract-method
    # signature is such as on the base class, pylint: disable=signature-differs
    def __new__(cls, data=None, learners=None, preprocessor=None, **kwargs):
        return super().__new__(
            cls,
            data, test_data=data, learners=learners, preprocessor=preprocessor,
            **kwargs)

    def __call__(self, data, learners, preprocessor=None, *, callback=None,
                 **kwargs):
        kwargs.setdefault("test_data", data)
        # if kwargs contains anything besides test_data, this will be detected
        # (and complained about) by super().__call__
        return super().__call__(
            data=data, learners=learners, preprocessor=preprocessor,
            callback=callback, **kwargs)


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
        a_sample = rgen.randint(0, len(table), n)
        o = np.ones(len(table))
        o[a_sample] = 0
        others = np.nonzero(o)[0]
        return table[a_sample], table[others]

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
