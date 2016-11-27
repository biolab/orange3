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
    will then determine which parameters have to be passed to individual
    learners.

    """
    __fits__ = None
    __returns__ = Model

    # Constants to indicate what kind of problem we're dealing with
    CLASSIFICATION, REGRESSION = range(2)

    def __init__(self, preprocessors=None, **kwargs):
        super().__init__(preprocessors=preprocessors)
        self.kwargs = kwargs
        # Make sure to pass preprocessor params to individual learners
        self.kwargs['preprocessors'] = preprocessors
        self.problem_type = None
        self.__learners = {self.CLASSIFICATION: None, self.REGRESSION: None}

    def __call__(self, data):
        # Set the appropriate problem type from the data
        self.problem_type = self.CLASSIFICATION if \
            data.domain.has_discrete_class else self.REGRESSION
        return self.get_learner(self.problem_type)(data)

    def get_learner(self, problem_type):
        if problem_type not in self.__learners:
            return None
        if self.__learners[problem_type] is None:
            learner = self.__fits__[problem_type](**self.__get_kwargs(
                self.kwargs, problem_type))
            learner.use_default_preprocessors = self.use_default_preprocessors
            self.__learners[problem_type] = learner
        return self.__learners[problem_type]

    def __get_kwargs(self, kwargs, problem_type):
        params = self._get_learner_kwargs(self.__fits__[problem_type])
        return {k: kwargs[k] for k in params & set(kwargs.keys())}

    @staticmethod
    def _get_learner_kwargs(learner):
        """Get a `set` of kwarg names that belong to the given learner."""
        # Get function params except `self`
        return set(learner.__init__.__code__.co_varnames[1:])

    def __getattr__(self, item):
        return getattr(self.get_learner(self.problem_type), item)
