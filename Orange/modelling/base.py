import numpy as np

from Orange.base import Learner, Model, SklLearner
from Orange.data import Table, Domain


class Fitter(Learner):
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
    __fits__ = {}
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
        learner = self.get_learner(data)

        if type(self).fit is Learner.fit:
            return learner.fit_storage(data)
        else:
            X, Y, W = data.X, data.Y, data.W if data.has_weights() else None
            return learner.fit(X, Y, W)

    def preprocess(self, data):
        return self.get_learner(data).preprocess(data)

    def get_learner(self, problem_type):
        """Get the learner for a given problem type.

        Parameters
        ----------
        problem_type: str or Table or Domain
            If str, one of ``'classification'`` or ``'regression'``. If Table
            or Domain, the type is inferred from Domain's first class variable.

        Returns
        -------
        Learner
            The appropriate learner for the given problem type.

        Raises
        ------
        TypeError
            When (inferred) problem type not one of ``'classification'``
            or ``'regression'``.
        """
        if isinstance(problem_type, Table):
            problem_type = problem_type.domain
        if isinstance(problem_type, Domain):
            problem_type = (self.CLASSIFICATION if problem_type.has_discrete_class else
                            self.REGRESSION if problem_type.has_continuous_class else
                            None)
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
        # Make sure to remove any params that are set to None and use defaults
        filtered_kwargs = {k: v for k, v in changed_kwargs.items() if v is not None}
        return {k: v for k, v in filtered_kwargs.items() if k in learner_kwargs}

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
            getattr(self.get_learner(self.CLASSIFICATION), 'supports_weights', False) and
            getattr(self.get_learner(self.REGRESSION), 'supports_weights', False)
        )

    @property
    def params(self):
        raise TypeError(
            'A fitter does not have its own params. If you need to access '
            'learner params, please use the `get_params` method.')

    def get_params(self, problem_type):
        """Access the specific learner params of a given learner."""
        return self.get_learner(problem_type).params


class SklFitter(Fitter):
    def _fit_model(self, data):
        model = super()._fit_model(data)
        model.used_vals = [np.unique(y) for y in data.Y[:, None].T]
        model.params = self.get_params(data)
        return model
