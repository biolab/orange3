# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import numpy as np

from Orange.classification import (CN2Learner, CN2UnorderedLearner,
                                   CN2SDLearner, CN2SDUnorderedLearner)
from Orange.classification.rules import (_RuleLearner, _RuleClassifier,
                                         RuleHunter, Rule, EntropyEvaluator,
                                         LaplaceAccuracyEvaluator,
                                         WeightedRelativeAccuracyEvaluator,
                                         argmaxrnd, hash_dist)
from Orange.data import Table
from Orange.data.filter import HasClass
from Orange.preprocess import Impute


class TestRuleInduction(unittest.TestCase):

    def setUp(self):
        self.titanic = Table('titanic')
        self.iris = Table('iris')

    def test_base_RuleLearner(self):
        """
        Base rule induction learner test. To pass the test, all base
        components are checked, including preprocessors, top-level
        control procedure elements (covering algorithm, rule stopping,
        data stopping), and bottom-level search procedure controller
        (rule finder).

        Every learner that extends _RuleLearner should override the fit
        method. It should at this point not yet be available (exception
        raised).
        """
        base_rule_learner = _RuleLearner()

        # test the number of default preprocessors
        self.assertEqual(len(list(base_rule_learner.active_preprocessors)), 3)
        # preprocessor types
        preprocessor_types = [type(x) for x in base_rule_learner.active_preprocessors]
        self.assertIn(HasClass, preprocessor_types)
        self.assertIn(Impute, preprocessor_types)

        # test find_rules
        base_rule_learner.domain = self.iris.domain
        base_rule_learner.find_rules(self.iris.X, self.iris.Y.astype(int),
                                     None, None, [], self.iris.domain)

        # test top-level control procedure elements
        self.assertTrue(hasattr(base_rule_learner, "data_stopping"))
        self.assertTrue(hasattr(base_rule_learner, "cover_and_remove"))
        self.assertTrue(hasattr(base_rule_learner, "rule_stopping"))

        # test exclusive covering algorithm
        new_rule = Rule()
        new_rule.covered_examples = np.array([True, False, True], dtype=bool)
        new_rule.target_class = None

        X, Y, W = base_rule_learner.exclusive_cover_and_remove(
            self.iris.X[:3], self.iris.Y[:3], None, new_rule)
        self.assertTrue(len(X) == len(Y) == 1)

        # test rule finder
        self.assertTrue(hasattr(base_rule_learner, "rule_finder"))
        rule_finder = base_rule_learner.rule_finder
        self.assertIsInstance(rule_finder, RuleHunter)
        self.assertTrue(hasattr(rule_finder, "search_algorithm"))
        self.assertTrue(hasattr(rule_finder, "search_strategy"))
        self.assertTrue(hasattr(rule_finder, "quality_evaluator"))
        self.assertTrue(hasattr(rule_finder, "complexity_evaluator"))
        self.assertTrue(hasattr(rule_finder, "general_validator"))
        self.assertTrue(hasattr(rule_finder, "significance_validator"))

    def testBaseRuleClassifier(self):
        """
        Every classifier that extends _RuleClassifier should override
        the predict method. It should at this point not yet be available
        (exception raised).
        """
        base_rule_classifier = _RuleClassifier(domain=self.iris.domain)
        self.assertRaises(NotImplementedError, base_rule_classifier.predict,
                          self.iris.X)

    def testCN2Learner(self):
        learner = CN2Learner()

        # classic CN2 removes covered learning instances
        self.assertTrue(learner.cover_and_remove ==
                        _RuleLearner.exclusive_cover_and_remove)

        # Entropy measure is used to evaluate found hypotheses
        self.assertTrue(type(learner.rule_finder.quality_evaluator) ==
                        EntropyEvaluator)

        # test that the learning requirements are relaxed by default
        self.assertTrue(learner.rule_finder.general_validator.max_rule_length >= 5)
        self.assertTrue(learner.rule_finder.general_validator.min_covered_examples == 1)

        classifier = learner(self.titanic)
        self.assertEqual(classifier.original_domain, self.titanic.domain)

        # all learning instances are covered when limitations do not
        # impose rule length or minimum number of covered examples
        num_covered = np.sum([rule.curr_class_dist
                              for rule in classifier.rule_list[:-1]])
        self.assertEqual(num_covered, self.titanic.X.shape[0])

        # prediction (matrix-wise, all testing instances at once)
        # test that returned result is of correct size
        predictions = classifier.predict(self.titanic.X)
        self.assertEqual(len(predictions), self.titanic.X.shape[0])

    def testUnorderedCN2Learner(self):
        learner = CN2UnorderedLearner()

        # unordered CN2 removes covered learning instances
        self.assertTrue(learner.cover_and_remove ==
                        _RuleLearner.exclusive_cover_and_remove)

        # Laplace accuracy measure is used to evaluate found hypotheses
        self.assertTrue(type(learner.rule_finder.quality_evaluator) ==
                        LaplaceAccuracyEvaluator)

        # test that the learning requirements are relaxed by default
        self.assertTrue(learner.rule_finder.general_validator.max_rule_length >= 5)
        self.assertTrue(learner.rule_finder.general_validator.min_covered_examples == 1)

        # by default, continuous variables are
        # constrained by the learning algorithm
        self.assertTrue(learner.rule_finder.search_strategy.constrain_continuous)

        classifier = learner(self.iris)
        self.assertEqual(classifier.original_domain, self.iris.domain)

        # all learning instances should be covered given the parameters
        for curr_class in range(len(self.iris.domain.class_var.values)):
            target_covered = (np.sum([rule.curr_class_dist[rule.target_class]
                                      for rule in classifier.rule_list
                                      if rule.target_class == curr_class]))
            self.assertEqual(target_covered, np.sum(self.iris.Y == curr_class))

        # a custom example, test setting several parameters
        learner = CN2UnorderedLearner()
        learner.rule_finder.search_algorithm.beam_width = 5
        learner.rule_finder.search_strategy.constrain_continuous = True
        learner.rule_finder.general_validator.min_covered_examples = 10
        learner.rule_finder.general_validator.max_rule_length = 2
        learner.rule_finder.significance_validator.parent_alpha = 0.9
        learner.rule_finder.significance_validator.default_alpha = 0.8

        classifier = learner(self.iris)

        # only the TRUE rule may exceed imposed limitations
        for rule in classifier.rule_list[:-1]:
            self.assertLessEqual(len(rule.selectors), 2)
            self.assertGreaterEqual(np.max(rule.curr_class_dist), 10)

        # prediction (matrix-wise, all testing instances at once)
        # test that returned result is of correct size
        predictions = classifier.predict(self.iris.X)
        self.assertEqual(len(predictions), self.iris.X.shape[0])

    def testOrderedCN2SDLearner(self):
        learner = CN2SDLearner()

        # Weighted relative accuracy measure is
        # used to evaluate found hypotheses
        self.assertTrue(type(learner.rule_finder.quality_evaluator) ==
                        WeightedRelativeAccuracyEvaluator)

        # gamma parameter must be initialized and defined
        self.assertTrue(hasattr(learner, "gamma"))

        classifier = learner(self.titanic)
        self.assertEqual(classifier.original_domain, self.titanic.domain)

        # prediction (matrix-wise, all testing instances at once)
        # test that returned result is of correct size
        predictions = classifier.predict(self.titanic.X)
        self.assertEqual(len(predictions), self.titanic.X.shape[0])

    def testUnorderedCN2SDLearner(self):
        learner = CN2SDUnorderedLearner()
        learner.rule_finder.significance_validator.parent_alpha = 0.2
        learner.rule_finder.significance_validator.default_alpha = 0.8

        self.assertTrue(type(learner.rule_finder.quality_evaluator) ==
                        WeightedRelativeAccuracyEvaluator)

        # gamma parameter must be initialized and defined
        self.assertTrue(hasattr(learner, "gamma"))

        classifier = learner(self.titanic)
        self.assertEqual(classifier.original_domain, self.titanic.domain)

        # prediction (matrix-wise, all testing instances at once)
        # test that returned result is of correct size
        predictions = classifier.predict(self.titanic.X)
        self.assertEqual(len(predictions), self.titanic.X.shape[0])

    def testArgMaxRnd(self):
        temp = np.array([np.nan, 1, 2.3, 37, 37, 37, 1])
        self.assertEqual(argmaxrnd(temp, hash_dist(np.array([3, 4]))), 5)
        self.assertRaises(ValueError, argmaxrnd, np.ones((1, 1, 1)))

if __name__ == '__main__':
    unittest.main()
