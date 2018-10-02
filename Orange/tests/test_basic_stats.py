# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from unittest import TestCase

import time

import numpy as np

from Orange.data import Table
from Orange.statistics.basic_stats import DomainBasicStats, BasicStats


class TestDomainBasicStats(TestCase):
    def setUp(self):
        self.zoo = Table("zoo")

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

    def test_speed(self):
        n, m = 10, 10000
        data = Table.from_numpy(None, np.random.rand(n, m))
        start = time.time()
        for i in range(m):
            BasicStats(data, i)
        elapsed = time.time() - start
        self.assertLess(elapsed, 10.0)

    def assertStatsEqual(self, stats1, stats2):
        self.assertEqual(len(stats1), len(stats2))
        for stat1, stat2 in zip(stats1, stats2):
            self.assertAlmostEqual(stat1.min, stat2.min)
            self.assertAlmostEqual(stat1.max, stat2.max)
            self.assertAlmostEqual(stat1.mean, stat2.mean)
            self.assertAlmostEqual(stat1.var, stat2.var)
            self.assertAlmostEqual(stat1.nans, stat2.nans)
            self.assertAlmostEqual(stat1.non_nans, stat2.non_nans)
