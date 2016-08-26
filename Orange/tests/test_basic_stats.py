# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from unittest import TestCase

import numpy as np

from Orange.data import Table
from Orange.statistics.basic_stats import DomainBasicStats, BasicStats


class TestBasicStats(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zoo = Table("zoo")

    def test_domain_basic_stats(self):
        domain = self.zoo.domain
        attr_stats = [BasicStats(self.zoo, a) for a in domain.attributes]
        class_var_stats = [BasicStats(self.zoo, a) for a in domain.class_vars]
        meta_stats = [BasicStats(self.zoo, a) for a in domain.metas]

        domain_stats = DomainBasicStats(self.zoo)
        self.assertStatsEqual(domain_stats.stats,
                              attr_stats + class_var_stats)

        domain_stats = DomainBasicStats(self.zoo, include_metas=True)
        self.assertStatsEqual(domain_stats.stats,
                              attr_stats + class_var_stats + meta_stats)

    def test_basic_stats_empty(self):
        stats = BasicStats()
        self.assertEqual(stats.min, float("inf"))
        self.assertEqual(stats.max, float("-inf"))
        self.assertEqual(stats.mean, 0)
        self.assertEqual(stats.var, 0)
        self.assertEqual(stats.nans, 0)
        self.assertEqual(stats.non_nans, 0)

    def assertStatsEqual(self, stats1, stats2):
        self.assertEqual(len(stats1), len(stats2))
        for stat1, stat2 in zip(stats1, stats2):
            self.assert_almost_equal_string_safe(stat1.min, stat2.min)
            self.assert_almost_equal_string_safe(stat1.max, stat2.max)
            self.assert_almost_equal_string_safe(stat1.mean, stat2.mean)
            self.assert_almost_equal_string_safe(stat1.var, stat2.var)
            self.assert_almost_equal_string_safe(stat1.nans, stat2.nans)
            self.assert_almost_equal_string_safe(stat1.non_nans, stat2.non_nans)

    def assert_almost_equal_string_safe(self, one, two):
        try:
            self.assertEqual(one, two)
        except:
            np.testing.assert_almost_equal(one, two)
