# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import numpy as np
from Orange.data import Table
from Orange.classification.rules import argmaxrnd, hash_dist
from Orange.classification.rules import main as rules_main
from Orange.classification import CN2Learner, CN2UnorderedLearner


class TestRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Table('iris')

    def test_ClassicCN2(self):
        learner = CN2Learner()
        classifier = learner(self.data)

        # all learning instances are covered when limitations
        # don't impose rule length or minimum number of covered
        # examples
        num_covered = np.sum([rule.curr_class_dist
                              for rule in classifier.rule_list])
        to_compare = self.data.X.shape[0]
        self.assertEqual(num_covered, to_compare)

        # prediction (matrix-wise, all testing instances at once)
        # returned result is of correct size
        predictions = classifier.predict(self.data.X)
        self.assertEqual(len(predictions), self.data.X.shape[0])

    def test_UnorderedCN2(self):
        learner = CN2UnorderedLearner()
        classifier = learner(self.data)

        # for each target class, all learning instances are
        # covered when limitations don't impose rule length
        # or minimum number of covered examples
        for curr_class in range(len(self.data.domain.class_var.values)):
            target_covered = (np.sum([rule.curr_class_dist[rule.target_class]
                                      for rule in classifier.rule_list
                                      if rule.target_class == curr_class]))
            to_compare = np.sum(self.data.Y == curr_class)
            self.assertEqual(target_covered, to_compare)

        learner = CN2UnorderedLearner()
        learner.rule_finder.general_validator.minimum_covered_examples = 10
        learner.rule_finder.general_validator.max_rule_length = 2
        learner.rule_finder.significance_validator.alpha = 0.9
        learner.rule_finder.search_algorithm.beam_width = 5
        learner.rule_finder.search_strategy.discretise_continuous = True
        classifier = learner(self.data)

        # no rules should exceed imposed limitations (except the TRUE rule)
        for rule in classifier.rule_list[:-1]:
            self.assertLessEqual(len(rule.selectors), 2)
            self.assertGreaterEqual(np.sum(rule.curr_class_dist), 10)

        # prediction (matrix-wise, all testing instances at once)
        # returned result is of correct size
        predictions = classifier.predict(self.data.X)
        self.assertEqual(len(predictions), self.data.X.shape[0])

    def test_argmaxrnd(self):
        temp = np.array([np.nan, 1, 2.3, 37, 37, 37, 1])
        self.assertEqual(
            argmaxrnd(temp, hash_dist(np.array([3, 4]))), 5)
        self.assertRaises(ValueError, argmaxrnd, np.ones((1, 1, 1)))

    def test_rules_main(self):
        # make sure that the examples in rules.main raise no errors
        rules_main()

if __name__ == '__main__':
    unittest.main()
