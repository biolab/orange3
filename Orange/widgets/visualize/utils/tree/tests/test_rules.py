"""Test rules for classification and regression trees."""
import unittest

from Orange.widgets.visualize.utils.tree.rules import (
    ContinuousRule,
    IntervalRule,
)


class TestRules(unittest.TestCase):
    """Rules for classification and regression trees.

    See Also
    --------
    Orange.widgets.visualize.widgetutils.tree.rules

    """
    # CONTINUOUS RULES
    def test_merging_two_gt_continuous_rules(self):
        """Merging `x > 1` and `x > 2` should produce `x > 2`."""
        rule1 = ContinuousRule('Rule', True, 1)
        rule2 = ContinuousRule('Rule', True, 2)
        new_rule = rule1.merge_with(rule2)
        self.assertEqual(new_rule.value, 2)

    def test_merging_gt_with_gte_continuous_rule(self):
        """Merging `x > 1` and `x ≥ 1` should produce `x > 1`."""
        rule1 = ContinuousRule('Rule', True, 1, inclusive=True)
        rule2 = ContinuousRule('Rule', True, 1, inclusive=False)
        new_rule = rule1.merge_with(rule2)
        self.assertEqual(new_rule.inclusive, False)

    def test_merging_two_lt_continuous_rules(self):
        """Merging `x < 1` and `x < 2` should produce `x < 1`."""
        rule1 = ContinuousRule('Rule', False, 1)
        rule2 = ContinuousRule('Rule', False, 2)
        new_rule = rule1.merge_with(rule2)
        self.assertEqual(new_rule.value, 1)

    def test_merging_lt_with_lte_rule(self):
        """Merging `x < 1` and `x ≤ 1` should produce `x < 1`."""
        rule1 = ContinuousRule('Rule', False, 1, inclusive=True)
        rule2 = ContinuousRule('Rule', False, 1, inclusive=False)
        new_rule = rule1.merge_with(rule2)
        self.assertEqual(new_rule.inclusive, False)

    def test_merging_lt_with_gt_continuous_rules(self):
        """Merging `x > 1` and `x < 2` should produce `1 < x < 2`."""
        rule1 = ContinuousRule('Rule', True, 1)
        rule2 = ContinuousRule('Rule', False, 2)
        new_rule = rule1.merge_with(rule2)
        self.assertIsInstance(new_rule, IntervalRule)
        self.assertEqual(new_rule.left_rule, rule1)
        self.assertEqual(new_rule.right_rule, rule2)

    # INTERVAL RULES
    def test_merging_interval_rule_with_smaller_continuous_rule(self):
        """Merging `1 < x < 2` and `x < 3` should produce `1 < x < 2`."""
        rule1 = IntervalRule('Rule',
                             ContinuousRule('Rule', True, 1),
                             ContinuousRule('Rule', False, 2))
        rule2 = ContinuousRule('Rule', False, 2)
        new_rule = rule1.merge_with(rule2)
        self.assertIsInstance(new_rule, IntervalRule)
        self.assertEqual(new_rule.right_rule.value, 2)

    def test_merging_interval_rule_with_larger_continuous_rule(self):
        """Merging `1 < x < 2` and `x < 3` should produce `1 < x < 2`."""
        rule1 = IntervalRule('Rule',
                             ContinuousRule('Rule', True, 1),
                             ContinuousRule('Rule', False, 2))
        rule2 = ContinuousRule('Rule', False, 3)
        new_rule = rule1.merge_with(rule2)
        self.assertIsInstance(new_rule, IntervalRule)
        self.assertEqual(new_rule.left_rule.value, 1)

    def test_merging_interval_rule_with_larger_lt_continuous_rule(self):
        """Merging `0 < x < 3` and `x > 1` should produce `1 < x < 3`."""
        rule1 = IntervalRule('Rule',
                             ContinuousRule('Rule', True, 0),
                             ContinuousRule('Rule', False, 3))
        rule2 = ContinuousRule('Rule', True, 1)
        new_rule = rule1.merge_with(rule2)
        self.assertIsInstance(new_rule, IntervalRule)
        self.assertEqual(new_rule.left_rule.value, 1)

    def test_merging_interval_rule_with_smaller_gt_continuous_rule(self):
        """Merging `0 < x < 3` and `x < 2` should produce `0 < x < 2`."""
        rule1 = IntervalRule('Rule',
                             ContinuousRule('Rule', True, 0),
                             ContinuousRule('Rule', False, 3))
        rule2 = ContinuousRule('Rule', False, 2)
        new_rule = rule1.merge_with(rule2)
        self.assertIsInstance(new_rule, IntervalRule)
        self.assertEqual(new_rule.right_rule.value, 2)

    def test_merging_interval_rules_with_smaller_lt_component(self):
        """Merging `1 < x < 2` and `0 < x < 2` should produce `1 < x < 2`."""
        rule1 = IntervalRule('Rule',
                             ContinuousRule('Rule', True, 1),
                             ContinuousRule('Rule', False, 2))
        rule2 = IntervalRule('Rule',
                             ContinuousRule('Rule', True, 0),
                             ContinuousRule('Rule', False, 2))
        new_rule = rule1.merge_with(rule2)
        self.assertEqual(new_rule.left_rule.value, 1)
        self.assertEqual(new_rule.right_rule.value, 2)

    def test_merging_interval_rules_with_larger_lt_component(self):
        """Merging `0 < x < 4` and `1 < x < 4` should produce `1 < x < 4`."""
        rule1 = IntervalRule('Rule',
                             ContinuousRule('Rule', True, 0),
                             ContinuousRule('Rule', False, 4))
        rule2 = IntervalRule('Rule',
                             ContinuousRule('Rule', True, 1),
                             ContinuousRule('Rule', False, 4))
        new_rule = rule1.merge_with(rule2)
        self.assertEqual(new_rule.left_rule.value, 1)
        self.assertEqual(new_rule.right_rule.value, 4)

    def test_merging_interval_rules_generally(self):
        """Merging `0 < x < 4` and `2 < x < 6` should produce `2 < x < 4`."""
        rule1 = IntervalRule('Rule',
                             ContinuousRule('Rule', True, 0),
                             ContinuousRule('Rule', False, 4))
        rule2 = IntervalRule('Rule',
                             ContinuousRule('Rule', True, 2),
                             ContinuousRule('Rule', False, 6))
        new_rule = rule1.merge_with(rule2)
        self.assertEqual(new_rule.left_rule.value, 2)
        self.assertEqual(new_rule.right_rule.value, 4)

    # ALL RULES
    def test_merge_commutativity_on_continuous_rules(self):
        """Continuous rule merging should be commutative."""
        rule1 = ContinuousRule('Rule1', True, 1)
        rule2 = ContinuousRule('Rule1', True, 2)
        new_rule1 = rule1.merge_with(rule2)
        new_rule2 = rule2.merge_with(rule1)
        self.assertEqual(new_rule1.value, new_rule2.value)

    def test_merge_commutativity_on_interval_rules(self):
        """Interval rule merging should be commutative."""
        rule1 = IntervalRule('Rule',
                             ContinuousRule('Rule', True, 0),
                             ContinuousRule('Rule', False, 4))
        rule2 = IntervalRule('Rule',
                             ContinuousRule('Rule', True, 2),
                             ContinuousRule('Rule', False, 6))
        new_rule1 = rule1.merge_with(rule2)
        new_rule2 = rule2.merge_with(rule1)
        self.assertEqual(new_rule1.left_rule.value,
                         new_rule2.left_rule.value)
        self.assertEqual(new_rule1.right_rule.value,
                         new_rule2.right_rule.value)

    def test_merge_keeps_gt_on_continuous_rules(self):
        """Merging ccontinuous rules should keep GT property."""
        rule1 = ContinuousRule('Rule1', True, 1)
        rule2 = ContinuousRule('Rule1', True, 2)
        new_rule = rule1.merge_with(rule2)
        self.assertEqual(new_rule.greater, True)

    def test_merge_keeps_attr_name_on_continuous_rules(self):
        """Merging continuous rules should keep the name of the rule."""
        rule1 = ContinuousRule('Rule1', True, 1)
        rule2 = ContinuousRule('Rule1', True, 2)
        new_rule = rule1.merge_with(rule2)
        self.assertEqual(new_rule.attr_name, 'Rule1')
