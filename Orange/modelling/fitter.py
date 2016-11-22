from collections import namedtuple

from Orange.base import Learner, Model

LearnerTypes = namedtuple('LearnerTypes', ['classification', 'regression'])


class Fitter(Learner):
    """Handle multiple types of target variable with one learner.

    Subclasses of this class serve as a sort of dispatcher. When subclassing,
    we provide a `LearnerTypes` instance which contain actual learner classes
    that handle appropriate data types. The fitter can then be used on any
    data and will delegate the work to the appropriate learner.

    If the learners that handle each data type require different parameters,
    you should pass in all the possible parameters to the fitter. The fitter
    will then determine which parameters have to passed to individual learners.

    """
    __fits__ = None
    __returns__ = Model

    # Constants to indicate what kind of problem we're dealing with
    CLASSIFICATION, REGRESSION = range(2)

    def __init__(self, *args, **kwargs):
        super().__init__(preprocessors=kwargs.get('preprocessors', None))
        self.args = args
        self.kwargs = kwargs
        self.problem_type = self.CLASSIFICATION
        self.__regression_learner = self.__classification_learner = None

    def __call__(self, data):
        # Set the appropriate problem type from the data
        self.problem_type = self.CLASSIFICATION if \
            data.domain.has_discrete_class else self.REGRESSION

        return self.learner(data)

    def __get_kwargs(self, kwargs, problem_type):
        if problem_type == self.CLASSIFICATION:
            params = self._get_learner_kwargs(self.__fits__.classification)
            kwarg_keys = params & set(kwargs.keys())
            kwargs = {k: kwargs[k] for k in kwarg_keys}
        elif problem_type == self.REGRESSION:
            params = self._get_learner_kwargs(self.__fits__.regression)
            kwarg_keys = params & set(kwargs.keys())
            kwargs = {k: kwargs[k] for k in kwarg_keys}
        # In case the preprocessors were changed since instantionation, we must
        # make sure to build the model with the latest preprocessors
        kwargs['preprocessors'] = self.preprocessors
        return kwargs

    @property
    def classification_learner(self):
        self.__check_dispatches(self.__fits__)
        if self.__classification_learner is None:
            self.__classification_learner = self.__fits__.classification(
                *self.args,
                **self.__get_kwargs(self.kwargs, self.CLASSIFICATION))
        return self.__classification_learner

    @property
    def regression_learner(self):
        self.__check_dispatches(self.__fits__)
        if self.__regression_learner is None:
            self.__regression_learner = self.__fits__.regression(
                *self.args, **self.__get_kwargs(self.kwargs, self.REGRESSION))
        return self.__regression_learner

    @staticmethod
    def _get_learner_kwargs(learner):
        """Get a `set` of kwarg names that belong to the given learner."""
        # Get function params except `self`
        params = learner.__init__.__code__.co_varnames[1:]
        return set(params)

    @staticmethod
    def __check_dispatches(dispatches):
        if not isinstance(dispatches, LearnerTypes):
            raise AssertionError(
                'The `__fits__` property must be an instance of '
                '`Orange.base.LearnerTypes`.')

    @property
    def learner(self):
        self._get_learner_kwargs(self.classification_learner)
        return self.classification_learner if \
            self.problem_type == self.CLASSIFICATION else \
            self.regression_learner

    @property
    def default_preprocessors(self):
        """Since the fitter is a proxy, it inherently has no preprocessors.
        Instead, it will pass the active learner preprocessors as the default
        ones."""
        return self.learner.default_preprocessors

    def __getattr__(self, item):
        return getattr(self.learner, item)
