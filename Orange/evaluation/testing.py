import sys
import multiprocessing as mp
from threading import Thread
from collections import namedtuple
import pickle
import warnings

import numpy as np

import joblib
import sklearn.cross_validation as skl_cross_validation

from Orange.util import OrangeWarning
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable

__all__ = ["Results", "CrossValidation", "LeaveOneOut", "TestOnTrainingData",
           "ShuffleSplit", "TestOnTestData", "sample"]

_MpResults = namedtuple('_MpResults', ('fold_i', 'learner_i', 'model',
                                       'failed', 'n_values', 'values', 'probs'))


def _identity(x):
    return x


def _mp_worker(fold_i, train_data, test_data, learner_i, learner,
               store_models, mp_queue):
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
    mp_queue.put("dummy text; use for printing when debugging")
    return _MpResults(fold_i, learner_i, store_models and model,
                      failed, len(test_data), predicted, probs)


class Results:
    """
    Class for storing predictions in model testing.

    Attributes:
        data (Optional[Table]):  Data used for testing. When data is stored,
            this is typically not a copy but a reference.

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
    score_by_folds = True
    # noinspection PyBroadException
    # noinspection PyNoneFunctionAssignment
    def __init__(self, data=None, nmethods=0, *, learners=None, train_data=None,
                 nrows=None, nclasses=None,
                 store_data=False, store_models=False,
                 domain=None, actual=None, row_indices=None,
                 predicted=None, probabilities=None,
                 preprocessor=None, callback=None, n_jobs=1):
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
        :param n_jobs: The number of processes to parallelize the evaluation
            on. -1 to parallelize on all but one CPUs. 1 for no
            parallelization.
        :type n_jobs: int
        """
        self.store_data = store_data
        self.store_models = store_models
        self.dtype = np.float32
        self.n_jobs = max(1, joblib.cpu_count() - 1 if n_jobs < 0 else n_jobs)

        self.models = None
        self.folds = None
        self.indices = None

        self.row_indices = row_indices
        self.preprocessor = preprocessor or _identity
        self._callback = callback or _identity
        self.learners = learners
        if learners:
            nmethods = len(learners)

        if nmethods is not None:
            self.failed = [False] * nmethods

        if data:
            self.data = data if self.store_data else None
            self.domain = data.domain
            self.dtype = getattr(data.Y, 'dtype', self.dtype)

        if learners:
            train_data = train_data or data
            self.fit(train_data, data)
            return

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

        if actual is not None:
            self.actual = actual
        elif nrows is not None:
            self.actual = np.empty(nrows, dtype=self.dtype)

        if predicted is not None:
            self.predicted = predicted
        elif nmethods is not None and nrows is not None:
            self.predicted = np.empty((nmethods, nrows), dtype=self.dtype)

        if probabilities is not None:
            self.probabilities = probabilities
        elif nmethods is not None and nrows is not None and \
                nclasses is not None:
            self.probabilities = \
                np.empty((nmethods, nrows, nclasses), dtype=np.float32)

    def _prepare_arrays(self, data):
        """Initialize some mandatory arrays for results"""
        nmethods = len(self.learners)
        self.nrows = len(self.row_indices)
        if self.store_models:
            self.models = np.tile(None, (len(self.indices), nmethods))
        # Initialize `predicted` and `probabilities` (only for discrete classes)
        self.predicted = np.empty((nmethods, self.nrows), dtype=self.dtype)
        if data.domain.has_discrete_class:
            nclasses = len(data.domain.class_var.values)
            self.probabilities = np.empty((nmethods, self.nrows, nclasses),
                                          dtype=np.float32)

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

    def fit(self, train_data, test_data=None):
        """Fits `self.learners` using folds sampled from the provided data.

        Args:
            train_data (Table): table to sample train folds
            test_data (Optional[Table]): tap to sample test folds
                of None then `train_data` will be used

        """
        test_data = test_data or train_data
        self.setup_indices(train_data, test_data)
        self.prepare_arrays(test_data)
        self._prepare_arrays(test_data)

        n_callbacks = len(self.learners) * len(self.indices)
        n_jobs = max(1, min(self.n_jobs, n_callbacks))

        def _is_picklable(obj):
            try:
                return bool(pickle.dumps(obj))
            except (AttributeError, TypeError, pickle.PicklingError):
                return False

        if n_jobs > 1 and not all(_is_picklable(learner) for learner in self.learners):
            n_jobs = 1
            warnings.warn("Not all arguments (learners) are picklable. "
                          "Setting n_jobs=1", OrangeWarning)

        if n_jobs > 1 and mp.current_process().daemon:
            n_jobs = 1
            warnings.warn("Worker subprocesses cannot spawn new worker "
                          "subprocesses (e.g. parameter tuning with internal "
                          "cross-validation). Setting n_jobs=1", OrangeWarning)

        # Workaround for NumPy locking on Macintosh.
        # https://pythonhosted.org/joblib/parallel.html#bad-interaction-of-multiprocessing-and-third-party-libraries
        mp_ctx = mp.get_context(
            'forkserver' if sys.platform == 'darwin' and n_jobs > 1 else None)

        try:
            # Use context-adapted Queue or just the regular Queue if no
            # multiprocessing (otherwise it shits itself at least on Windos)
            mp_queue = mp_ctx.Manager().Queue() if n_jobs > 1 else mp.Queue()
        except (EOFError, RuntimeError):
            mp_queue = mp.Queue()
            n_jobs = 1
            warnings.warn('''

        Can't run multiprocessing code without a __main__ guard.

        Multiprocessing strategies 'forkserver' (used by Orange's evaluation
        methods by default on Mac OS X) and 'spawn' (default on Windos)
        require the main code entry point be guarded with:

            if __name__ == '__main__':
                import multiprocessing as mp
                mp.freeze_support()  # Needed only on Windos
                ...  # Rest of your code
                ...  # See: https://docs.python.org/3/library/__main__.html

        Otherwise, as the module is re-imported in another process, infinite
        recursion ensues.

        Guard your executed code with above Python idiom, or pass n_jobs=1
        to evaluation methods, i.e. {}(..., n_jobs=1). Setting n_jobs to 1.
            '''.format(self.__class__.__name__), OrangeWarning)

        data_splits = (
            (fold_i, self.preprocessor(train_data[train_i]), test_data[test_i])
            for fold_i, (train_i, test_i) in enumerate(self.indices))

        args_iter = (
            (fold_i, train_data, test_data, learner_i, learner,
             self.store_models, mp_queue)
            # NOTE: If this nested for loop doesn't work, try
            # itertools.product
            for (fold_i, train_data, test_data) in data_splits
            for (learner_i, learner) in enumerate(self.learners))

        def _callback_percent(n_steps, queue):
            """Block until one of the subprocesses completes, before
            signalling callback with percent"""
            for percent in np.linspace(.0, .99, n_steps + 1)[1:]:
                queue.get()
                try:
                    self._callback(percent)
                except Exception:
                    # Callback may error for whatever reason (e.g. PEBKAC)
                    # In that case, rather gracefully continue computation
                    # instead of failing
                    pass

        results = []
        with joblib.Parallel(n_jobs=n_jobs, backend=mp_ctx) as parallel:
            tasks = (joblib.delayed(_mp_worker)(*args) for args in args_iter)
            # Start the tasks from another thread ...
            thread = Thread(target=lambda: results.append(parallel(tasks)))
            thread.start()
            # ... so that we can update the GUI (callback) from the main thread
            _callback_percent(n_callbacks, mp_queue)
            thread.join()

        results = sorted(results[0])

        ptr, prev_fold_i, prev_n_values = 0, 0, 0
        for res in results:
            if res.fold_i != prev_fold_i:
                ptr += prev_n_values
                prev_fold_i = res.fold_i
            result_slice = slice(ptr, ptr + res.n_values)
            prev_n_values = res.n_values

            if res.failed:
                self.failed[res.learner_i] = res.failed
                continue

            if self.store_models:
                self.models[res.fold_i][res.learner_i] = res.model

            self.predicted[res.learner_i][result_slice] = res.values
            if train_data.domain.has_discrete_class:
                self.probabilities[res.learner_i][result_slice, :] = res.probs

        self._callback(1)
        return self

    def prepare_arrays(self, test_data):
        """Initialize arrays that will be used by `fit` method.
        """
        self.folds = []
        row_indices = []

        ptr = 0
        for train, test in self.indices:
            self.folds.append(slice(ptr, ptr + len(test)))
            row_indices.append(test)
            ptr += len(test)

        self.row_indices = np.concatenate(row_indices, axis=0)
        self.actual = test_data[self.row_indices].Y.ravel()

    def setup_indices(self, train_data, test_data):
        """Initializes `self.indices` with iterable objects with slices
        (or indices) for each fold.

        Args:
            train_data (Table): train table
            test_data (Table): test table
        """
        raise NotImplementedError()

    def split_by_model(self):
        """Split evaluation results by models
        """
        data = self.data
        nmethods = len(self.predicted)
        for i in range(nmethods):
            res = Results()
            res.data = data
            res.domain = self.domain
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
                 store_models=False, preprocessor=None, callback=None, warnings=None,
                 n_jobs=1):
        self.k = k
        self.stratified = stratified
        self.random_state = random_state
        if warnings is None:
            self.warnings = []
        else:
            self.warnings = warnings

        super().__init__(data, learners=learners, store_data=store_data,
                         store_models=store_models, preprocessor=preprocessor,
                         callback=callback, n_jobs=n_jobs)

    def setup_indices(self, train_data, test_data):
        self.indices = None
        if self.stratified and test_data.domain.has_discrete_class:
            self.indices = skl_cross_validation.StratifiedKFold(
                test_data.Y, self.k, shuffle=True, random_state=self.random_state
            )
            if any(len(train) == 0 or len(test) == 0 for train, test in self.indices):
                self.warnings.append("Using non-stratified sampling.")
                self.indices = None
        if self.indices is None:
            self.indices = skl_cross_validation.KFold(
                len(test_data), self.k, shuffle=True, random_state=self.random_state
            )


class LeaveOneOut(Results):
    """Leave-one-out testing"""
    score_by_folds = False

    def __init__(self, data, learners, store_data=False, store_models=False,
                 preprocessor=None, callback=None, n_jobs=1):
        super().__init__(data, learners=learners, store_data=store_data,
                         store_models=store_models, preprocessor=preprocessor,
                         callback=callback, n_jobs=n_jobs)

    def setup_indices(self, train_data, test_data):
        self.indices = skl_cross_validation.LeaveOneOut(len(test_data))

    def prepare_arrays(self, test_data):
        # sped up version of super().prepare_arrays(data)
        self.row_indices = np.arange(len(test_data))
        self.folds = self.row_indices
        self.actual = test_data.Y.flatten()


class ShuffleSplit(Results):
    def __init__(self, data, learners, n_resamples=10, train_size=None,
                 test_size=0.1, stratified=True, random_state=0, store_data=False,
                 store_models=False, preprocessor=None, callback=None, n_jobs=1):
        self.n_resamples = n_resamples
        self.train_size = train_size
        self.test_size = test_size
        self.stratified = stratified
        self.random_state = random_state

        super().__init__(data, learners=learners, store_data=store_data,
                         store_models=store_models, preprocessor=preprocessor,
                         callback=callback, n_jobs=n_jobs)

    def setup_indices(self, train_data, test_data):
        if self.stratified and test_data.domain.has_discrete_class:
            self.indices = skl_cross_validation.StratifiedShuffleSplit(
                test_data.Y, n_iter=self.n_resamples, train_size=self.train_size,
                test_size=self.test_size, random_state=self.random_state
            )
        else:
            self.indices = skl_cross_validation.ShuffleSplit(
                len(test_data), n_iter=self.n_resamples, train_size=self.train_size,
                test_size=self.test_size, random_state=self.random_state
            )


class TestOnTestData(Results):
    """
    Test on a separate test data set.
    """
    def __init__(self, train_data, test_data, learners, store_data=False,
                 store_models=False, preprocessor=None, callback=None, n_jobs=1):
        super().__init__(test_data, train_data=train_data, learners=learners,
                         store_data=store_data,
                         store_models=store_models, preprocessor=preprocessor,
                         callback=callback, n_jobs=n_jobs)

    def setup_indices(self, train_data, test_data):
        self.indices = ((Ellipsis, Ellipsis),)

    def prepare_arrays(self, test_data):
        self.row_indices = np.arange(len(test_data))
        self.folds = (Ellipsis, )
        self.actual = test_data.Y.ravel()


class TestOnTrainingData(TestOnTestData):
    """
    Trains and test on the same data
    """

    def __init__(self, data, learners, store_data=False, store_models=False,
                 preprocessor=None, callback=None, n_jobs=1):

        if preprocessor is not None:
            data = preprocessor(data)

        super().__init__(train_data=data, test_data=data, learners=learners,
                         store_data=store_data, store_models=store_models,
                         preprocessor=None, callback=callback, n_jobs=n_jobs)
        self.preprocessor = preprocessor


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
