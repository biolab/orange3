import unittest
import inspect
import pkgutil
import traceback

import Orange
from Orange.data import Table
from Orange.regression import Learner
from Orange.regression import LinearRegressionLearner

class RegressionLearnersTest(unittest.TestCase):
    def all_learners(self):
        regression_modules = pkgutil.walk_packages(
            path=Orange.regression.__path__,
            prefix="Orange.regression.",
            onerror=lambda x: None)
        for importer, modname, ispkg in regression_modules:
            try:
                module = pkgutil.importlib.import_module(modname)
            except ImportError:
                continue

            for name, class_ in inspect.getmembers(module, inspect.isclass):
                if issubclass(class_, Learner) and 'base' not in class_.__module__:
                    yield class_

    def test_adequacy_all_learners(self):
        for learner in self.all_learners():
            try:
                learner = learner()
                table = Table("iris")
                self.assertRaises(ValueError, learner, table)
            except TypeError as err:
                traceback.print_exc()
                continue

    def test_coefficients(self):
        data = Table([[11], [12], [13]], [0, 1, 2])
        model = LinearRegressionLearner()(data)
        self.assertAlmostEqual(float(model.intercept), -11)
        self.assertEqual(len(model.coefficients), 1)
        self.assertAlmostEqual(float(model.coefficients[0]), 1)

        for learner in self.all_learners():
            if isinstance(learner, LinearRegressionLearner):
                data = Table([[1, 2, 3], [1, 2, 3]], [0, 1.1])
                model = learner()(data)
                self.assertEqual(len(model.coefficients), 3)
