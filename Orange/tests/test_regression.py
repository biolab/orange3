import unittest
import inspect
import pkgutil
import traceback

import Orange
from Orange.data import Table
from Orange.regression import Learner


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
