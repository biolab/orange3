import unittest
from unittest.mock import Mock

from Orange.classification.base_classification import LearnerClassification
from Orange.data import Table, ContinuousVariable
from Orange.modelling import Fitter
from Orange.preprocess import Randomize, Discretize
from Orange.regression.base_regression import LearnerRegression


class DummyClassificationLearner(LearnerClassification):
    pass


class DummyRegressionLearner(LearnerRegression):
    pass


class DummyFitter(Fitter):
    name = 'dummy'
    __fits__ = {'classification': DummyClassificationLearner,
                'regression': DummyRegressionLearner}


class FitterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.heart_disease = Table('heart_disease')
        cls.housing = Table('housing')

    def test_dispatches_to_correct_learner(self):
        """Based on the input data, it should dispatch the fitting process to
        the appropriate learner"""
        DummyClassificationLearner.fit = Mock()
        DummyRegressionLearner.fit = Mock()
        fitter = DummyFitter()

        fitter(self.heart_disease)
        self.assertEqual(
            DummyClassificationLearner.fit.call_count, 1,
            'Classification learner was never called for classification'
            'problem')
        self.assertEqual(
            DummyRegressionLearner.fit.call_count, 0,
            'Regression learner was called for classification problem')

        DummyClassificationLearner.fit.reset_mock()
        DummyRegressionLearner.fit.reset_mock()

        fitter(self.housing)
        self.assertEqual(
            DummyRegressionLearner.fit.call_count, 1,
            'Regression learner was never called for regression problem')
        self.assertEqual(
            DummyClassificationLearner.fit.call_count, 0,
            'Classification learner was called for regression problem')

    def test_constructs_learners_with_appropriate_parameters(self):
        """In case the classification and regression learners require different
        parameters, the fitter should be able to determine which ones have to
        be passed where"""

        class DummyClassificationLearner(LearnerClassification):
            def __init__(self, classification_param=1, **_):
                super().__init__()
                self.param = classification_param

        class DummyRegressionLearner(LearnerRegression):
            def __init__(self, regression_param=2, **_):
                super().__init__()
                self.param = regression_param

        class DummyFitter(Fitter):
            __fits__ = {'classification': DummyClassificationLearner,
                        'regression': DummyRegressionLearner}

        # Prevent fitting error from being thrown
        DummyClassificationLearner.fit = Mock()
        DummyRegressionLearner.fit = Mock()

        # Test without passing any params
        fitter = DummyFitter()
        self.assertEqual(fitter.get_learner(Fitter.CLASSIFICATION).param, 1)
        self.assertEqual(fitter.get_learner(Fitter.REGRESSION).param, 2)

        # Pass specific params
        try:
            fitter = DummyFitter(classification_param=10, regression_param=20)
            self.assertEqual(fitter.get_learner(Fitter.CLASSIFICATION).param, 10)
            self.assertEqual(fitter.get_learner(Fitter.REGRESSION).param, 20)
        except TypeError:
            self.fail('Fitter did not properly distribute params to learners')

    def test_correctly_sets_preprocessors_on_learner(self):
        """Fitters have to be able to pass the `use_default_preprocessors` and
        preprocessors down to individual learners"""
        pp = Randomize()
        fitter = DummyFitter(preprocessors=pp)
        fitter.use_default_preprocessors = True
        learner = fitter.get_learner(Fitter.CLASSIFICATION)

        self.assertEqual(
            learner.use_default_preprocessors, True,
            'Fitter did not properly pass the `use_default_preprocessors`'
            'attribute to its learners')
        self.assertEqual(
            tuple(learner.active_preprocessors), (pp,),
            'Fitter did not properly pass its preprocessors to its learners')

    def test_properly_delegates_preprocessing(self):
        class DummyClassificationLearner(LearnerClassification):
            preprocessors = [Discretize()]

            def __init__(self, classification_param=1, **_):
                super().__init__()
                self.param = classification_param

        class DummyFitter(Fitter):
            __fits__ = {'classification': DummyClassificationLearner,
                        'regression': DummyRegressionLearner}

        data = self.heart_disease
        fitter = DummyFitter()
        # Sanity check
        self.assertTrue(any(
            isinstance(v, ContinuousVariable) for v in data.domain.variables))
        # Preprocess the data and check that the discretization was applied
        pp_data = fitter.preprocess(self.heart_disease)
        self.assertTrue(not any(
            isinstance(v, ContinuousVariable) for v in pp_data.domain.variables))

    def test_default_kwargs_with_change_kwargs(self):
        """Fallback to default args in case specialized params not specified.
        """
        class DummyClassificationLearner(LearnerClassification):
            def __init__(self, param='classification_default', **_):
                super().__init__()
                self.param = param

            def fit_storage(self, data):
                return DummyModel(self.param)

        class DummyRegressionLearner(LearnerRegression):
            def __init__(self, param='regression_default', **_):
                super().__init__()
                self.param = param

            def fit_storage(self, data):
                return DummyModel(self.param)

        class DummyModel:
            def __init__(self, param):
                self.param = param

        class DummyFitter(Fitter):
            __fits__ = {'classification': DummyClassificationLearner,
                        'regression': DummyRegressionLearner}

            def _change_kwargs(self, kwargs, problem_type):
                if problem_type == self.CLASSIFICATION:
                    kwargs['param'] = kwargs.get('classification_param')
                else:
                    kwargs['param'] = kwargs.get('regression_param')
                return kwargs

        learner = DummyFitter()
        iris, housing = Table('iris')[:5], Table('housing')[:5]
        self.assertEqual(learner(iris).param, 'classification_default')
        self.assertEqual(learner(housing).param, 'regression_default')
