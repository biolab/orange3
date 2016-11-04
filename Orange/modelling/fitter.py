from collections import namedtuple

from Orange.base import Learner, Model

LearnerTypes = namedtuple('LearnerTypes', ['classification', 'regression'])


class Fitter(Learner):
    __fits__ = None
    __returns__ = Model

    # Constants to indicate what kind of problem we're dealing with
    CLASSIFICATION, REGRESSION = range(2)

    def __init__(self, *args, **kwargs):
        super().__init__(preprocessors=kwargs.get('preprocessors', None))
        self.args = args
        self.kwargs = kwargs
        self.problem_type = None
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

    def __getattr__(self, item):
        return getattr(self.learner, item)
