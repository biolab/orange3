# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import pickle
import unittest

from Orange.base import SklLearner, Learner, Model
from Orange.data import Domain, Table
from Orange.preprocess import Discretize, Randomize, Continuize
from Orange.regression import LinearRegressionLearner


class DummyLearner(Learner):
    def fit(self, *_, **__):
        return unittest.mock.Mock()


class DummySklLearner(SklLearner):
    def fit(self, *_, **__):
        return unittest.mock.Mock()


class DummyLearnerPP(Learner):
    preprocessors = (Randomize(),)


class TestLearner(unittest.TestCase):
    def test_uses_default_preprocessors_unless_custom_pps_specified(self):
        """Learners should use their default preprocessors unless custom
        preprocessors were passed in to the constructor"""
        learner = DummyLearner()
        self.assertEqual(
            type(learner).preprocessors, tuple(learner.active_preprocessors),
            'Learner should use default preprocessors, unless preprocessors '
            'were specified in init')

    def test_overrides_custom_preprocessors(self):
        """Passing preprocessors to the learner constructor should override the
        default preprocessors defined on the learner"""
        pp = Discretize()
        learner = DummyLearnerPP(preprocessors=(pp,))
        self.assertEqual(
            tuple(learner.active_preprocessors), (pp,),
            'Learner should override default preprocessors when specified in '
            'constructor')

    def test_use_default_preprocessors_property(self):
        """We can specify that we want to use default preprocessors despite
        passing our own ones in the constructor"""
        learner = DummyLearnerPP(preprocessors=(Discretize(),))
        learner.use_default_preprocessors = True

        preprocessors = list(learner.active_preprocessors)
        self.assertEqual(
            len(preprocessors), 2,
            'Learner did not properly insert custom preprocessor into '
            'preprocessor list')
        self.assertIsInstance(
            preprocessors[0], Discretize,
            'Custom preprocessor was inserted in incorrect order')
        self.assertIsInstance(preprocessors[1], Randomize)

    def test_preprocessors_can_be_passed_in_as_non_iterable(self):
        """For convenience, we can pass a single preprocessor instance"""
        pp = Discretize()
        learner = DummyLearnerPP(preprocessors=pp)
        self.assertEqual(
            tuple(learner.active_preprocessors), (pp,),
            'Preprocessors should be able to be passed in as single object '
            'as well as an iterable object')

    def test_preprocessors_can_be_passed_in_as_generator(self):
        """Since we support iterables, we should support generators as well"""
        pp = (Discretize(),)
        learner = DummyLearnerPP(p for p in pp)
        self.assertEqual(
            tuple(learner.active_preprocessors), pp,
            'Preprocessors should be able to be passed in as single object '
            'as well as an iterable object')

    def test_callback(self):
        callback = unittest.mock.Mock()
        learner = DummyLearner(preprocessors=[Discretize(), Randomize()])
        learner(Table("iris"), callback)
        args = [x[0][0] for x in callback.call_args_list]
        self.assertEqual(min(args), 0)
        self.assertEqual(max(args), 1)
        self.assertListEqual(args, sorted(args))


class TestSklLearner(unittest.TestCase):
    def test_sklearn_supports_weights(self):
        """Check that the SklLearner correctly infers whether or not the
        learner supports weights"""

        class DummySklLearner:
            def fit(self, X, y, sample_weight=None):
                pass

        class DummyLearner(SklLearner):
            __wraps__ = DummySklLearner

        self.assertTrue(DummyLearner().supports_weights)

        class DummySklLearner:
            def fit(self, X, y):
                pass

        class DummyLearner(SklLearner):
            __wraps__ = DummySklLearner

        self.assertFalse(DummyLearner().supports_weights)

    def test_linreg(self):
        self.assertTrue(
            LinearRegressionLearner().supports_weights,
            "Either LinearRegression no longer supports weighted tables or "
            "SklLearner.supports_weights is out-of-date.")

    def test_callback(self):
        callback = unittest.mock.Mock()
        learner = DummySklLearner(preprocessors=[Continuize(), Randomize()])
        learner(Table("iris"), callback)
        args = [x[0][0] for x in callback.call_args_list]
        self.assertEqual(min(args), 0)
        self.assertEqual(max(args), 1)
        self.assertListEqual(args, sorted(args))


class TestModel(unittest.TestCase):
    def test_pickle(self):
        """Make sure data is not saved when pickling a model."""
        model = Model(Domain([]))
        model.original_data = [1, 2, 3]
        model2 = pickle.loads(pickle.dumps(model))
        self.assertEqual(model.domain, model2.domain)
        self.assertEqual(model.original_data, [1, 2, 3])
        self.assertEqual(model2.original_data, None)


if __name__ == "__main__":
    unittest.main()
