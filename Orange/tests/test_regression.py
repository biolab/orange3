# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import inspect
import pkgutil
import traceback

import Orange
from Orange.data import Table
from Orange.regression import Learner, CurveFitLearner
from Orange.tests import test_filename


def all_learners():
    regression_modules = pkgutil.walk_packages(
        path=Orange.regression.__path__,
        prefix="Orange.regression.",
        onerror=lambda x: None)
    for _, modname, _ in regression_modules:
        try:
            module = pkgutil.importlib.import_module(modname)
        except ImportError:
            continue

        for _, class_ in inspect.getmembers(module, inspect.isclass):
            if issubclass(class_, Learner) and 'base' not in class_.__module__:
                yield class_


def init_learner(learner, table):
    if learner == CurveFitLearner:
        return CurveFitLearner(
            lambda x, a: x[:, -1] * a, [],
            [table.domain.attributes[-1].name]
        )
    return learner()


class TestRegression(unittest.TestCase):
    def test_adequacy_all_learners(self):
        table = Table("iris")
        for learner in all_learners():
            learner = init_learner(learner, table)
            with self.assertRaises(ValueError):
                learner(table)

    def test_adequacy_all_learners_multiclass(self):
        table = Table(test_filename("datasets/test8.tab"))
        for learner in all_learners():
            learner = init_learner(learner, table)
            with self.assertRaises(ValueError):
                learner(table)

    def test_missing_class(self):
        table = Table(test_filename("datasets/imports-85.tab"))
        for learner in all_learners():
            learner = init_learner(learner, table)
            model = learner(table)
            model(table)


if __name__ == "__main__":
    unittest.main()
