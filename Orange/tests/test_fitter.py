import unittest
from unittest.mock import Mock

from Orange.classification.base_classification import LearnerClassification
from Orange.data import Table
from Orange.modelling import Fitter, LearnerTypes
from Orange.preprocess import Discretize
from Orange.preprocess import Randomize
from Orange.regression.base_regression import LearnerRegression


class DummyClassificationLearner(LearnerClassification):
    pass


class DummyRegressionLearner(LearnerRegression):
    pass


class DummyFitter(Fitter):
    __fits__ = LearnerTypes(
        classification=DummyClassificationLearner,
        regression=DummyRegressionLearner)


class DummyClassificationLearnerPPs(LearnerClassification):
    preprocessors = (Randomize(),)


class DummyRegressionLearnerPPs(LearnerRegression):
    preprocessors = (Randomize(),)


class DummyFitterPPs(Fitter):
    __fits__ = LearnerTypes(
        classification=DummyClassificationLearnerPPs,
        regression=DummyRegressionLearnerPPs)


class FitterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.heart_disease = Table('heart_disease')
        cls.housing = Table('housing')

    def test_throws_if_fits_property_is_invalid(self):
        """The `__fits__` attribute must be an instance of `LearnerTypes`"""
        with self.assertRaises(Exception):
            class DummyFitter(Fitter):
                __fits__ = (DummyClassificationLearner, DummyRegressionLearner)

            fitter = DummyFitter()
            fitter(self.heart_disease)

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
            __fits__ = LearnerTypes(
                classification=DummyClassificationLearner,
                regression=DummyRegressionLearner)

        # Prevent fitting error from being thrown
        DummyClassificationLearner.fit = Mock()
        DummyRegressionLearner.fit = Mock()

        # Test without passing any params
        fitter = DummyFitter()
        self.assertEqual(fitter.classification_learner.param, 1)
        self.assertEqual(fitter.regression_learner.param, 2)

        # Pass specific params
        try:
            fitter = DummyFitter(classification_param=10, regression_param=20)
            self.assertEqual(fitter.classification_learner.param, 10)
            self.assertEqual(fitter.regression_learner.param, 20)
        except AttributeError:
            self.fail('Fitter did not properly distribute params to learners.')

    def test_uses_default_preprocessors_unless_custom_pps_specified(self):
        """Learners should use their default preprocessors unless custom
        preprocessors were passed in to the constructor"""
        fitter = DummyFitterPPs()
        self.assertEqual(
            type(fitter.classification_learner).preprocessors,
            fitter.classification_learner.active_preprocessors,
            'Classification learner should use default preprocessors, unless '
            'preprocessors were specified in init')
        self.assertEqual(
            type(fitter.regression_learner).preprocessors,
            fitter.regression_learner.active_preprocessors,
            'Regression learner should use default preprocessors, unless '
            'preprocessors were specified in init')

    def test_overrides_custom_preprocessors(self):
        pp = Discretize()
        fitter = DummyFitterPPs(preprocessors=(pp,))
        self.assertEqual(
            fitter.classification_learner.active_preprocessors, (pp,),
            'Classification learner should override default preprocessors '
            'when specified in constructor')
        self.assertEqual(
            fitter.regression_learner.active_preprocessors, (pp,),
            'Regression learner should override default preprocessors '
            'when specified in constructor')

    def test_use_default_preprocessors_property(self):
        fitter = DummyFitterPPs(preprocessors=(Discretize(),))
        fitter.use_default_preprocessors = True
        self.assertTrue(
            fitter.classification_learner.use_default_preprocessors,
            'Use default preprocessors property was not passed down to actual '
            'learner')
        self.assertEqual(
            len(fitter.classification_learner.active_preprocessors), 2,
            'Learner did not properly insert custom preprocessor into '
            'preprocessor list')
        self.assertIsInstance(
            fitter.classification_learner.active_preprocessors[0], Discretize,
            'Custom preprocessor was inserted in incorrect order')
        self.assertIsInstance(
            fitter.classification_learner.active_preprocessors[1], Randomize)

    def test_preprocessors_can_be_passed_in_as_non_iterable(self):
        pp = Discretize()
        fitter = DummyFitterPPs(preprocessors=pp)
        self.assertEqual(
            fitter.learner.active_preprocessors, (pp,),
            'Preprocessors should be able to be passed in as single object '
            'as well as an iterable object')
