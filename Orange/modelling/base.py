from Orange.base import Learner, Model


class FitterMeta(type):
    """Ensure that each subclass of the `Fitter` class overrides the `__fits__`
    attribute with a valid value."""
    def __new__(mcs, name, bases, attrs):
        # Check that a fitter implementation defines a valid `__fits__`
        if any(cls.__name__ == 'Fitter' for cls in bases):
            fits = attrs.get('__fits__')
            assert isinstance(fits, dict), '__fits__ must be dict instance'
            assert fits.get('classification') and fits.get('regression'), \
                ('`__fits__` property does not define classification '
                 'or regression learner. Use a simple learner if you don\'t '
                 'need the functionality provided by Fitter.')
        return super().__new__(mcs, name, bases, attrs)


class Fitter(Learner, metaclass=FitterMeta):
    """Handle multiple types of target variable with one learner.

    Subclasses of this class serve as a sort of dispatcher. When subclassing,
    we provide a `dict` which contain actual learner classes that handle
    appropriate data types. The fitter can then be used on any data and will
    delegate the work to the appropriate learner.

    If the learners that handle each data type require different parameters,
    you should pass in all the possible parameters to the fitter. The fitter
    will then determine which parameters have to be passed to individual
    learners.

    """
    __fits__ = None
    __returns__ = Model

    # Constants to indicate what kind of problem we're dealing with
    CLASSIFICATION, REGRESSION = 'classification', 'regression'

    def __init__(self, preprocessors=None, **kwargs):
        super().__init__(preprocessors=preprocessors)
        self.kwargs = kwargs
        # Make sure to pass preprocessor params to individual learners
        self.kwargs['preprocessors'] = preprocessors
        self.__learners = {self.CLASSIFICATION: None, self.REGRESSION: None}

    def _fit_model(self, data):
        if data.domain.has_discrete_class:
            learner = self.get_learner(self.CLASSIFICATION)
        else:
            learner = self.get_learner(self.REGRESSION)

        if type(self).fit is Learner.fit:
            return learner.fit_storage(data)
        else:
            X, Y, W = data.X, data.Y, data.W if data.has_weights() else None
            return learner.fit(X, Y, W)

    def get_learner(self, problem_type):
        """Get the learner for a given problem type.

        Returns
        -------
        Learner
            The appropriate learner for the given problem type.

        """
        # Prevent trying to access the learner when problem type is None
        if problem_type not in self.__fits__:
            raise TypeError("No learner to handle '{}'".format(problem_type))
        if self.__learners[problem_type] is None:
            learner = self.__fits__[problem_type](**self.__kwargs(problem_type))
            learner.use_default_preprocessors = self.use_default_preprocessors
            self.__learners[problem_type] = learner
        return self.__learners[problem_type]

    def __kwargs(self, problem_type):
        learner_kwargs = set(
            self.__fits__[problem_type].__init__.__code__.co_varnames[1:])
        changed_kwargs = self._change_kwargs(self.kwargs, problem_type)
        return {k: v for k, v in changed_kwargs.items() if k in learner_kwargs}

    def _change_kwargs(self, kwargs, problem_type):
        """Handle the kwargs to be passed to the learner before they are used.

        In some cases we need to manipulate the kwargs that will be passed to
        the learner, e.g. SGD takes a `loss` parameter in both the regression
        and classification learners, but the learner widget cannot
        differentiate between these two, so it passes classification and
        regression loss parameters individually. The appropriate one must be
        renamed into `loss` before passed to the actual learner instance. This
        is done here.

        """
        return kwargs

    @property
    def supports_weights(self):
        """The fitter supports weights if both the classification and
        regression learners support weights."""
        return (
            hasattr(self.get_learner(self.CLASSIFICATION), 'supports_weights')
            and self.get_learner(self.CLASSIFICATION).supports_weights) and (
            hasattr(self.get_learner(self.REGRESSION), 'supports_weights')
            and self.get_learner(self.REGRESSION).supports_weights)

    @property
    def params(self):
        raise TypeError(
            'A fitter does not have its own params. If you need to access '
            'learner params, please use the `get_params` method.')

    def get_params(self, problem_type):
        """Access the specific learner params of a given learner."""
        return self.get_learner(problem_type).params
