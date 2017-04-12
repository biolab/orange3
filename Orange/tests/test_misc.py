import unittest

from Orange.misc.cache import memoize_method, single_cache


class Calculator:
    @memoize_method()
    def my_sum(self, *nums):
        return sum(nums)


@single_cache
def my_sum(*nums):
    return sum(nums)


class TestCache(unittest.TestCase):

    def test_single_cache(self):
        self.assertEqual(my_sum(1, 2, 3, 4, 5), 15)
        self.assertEqual(my_sum(1, 2, 3, 4, 5), 15)
        # Make sure different args produce different results
        self.assertEqual(my_sum(1, 2, 3, 4), 10)

    def test_memoize_method(self):
        calc = Calculator()
        self.assertEqual(calc.my_sum(1, 2, 3, 4, 5), 15)
        self.assertEqual(calc.my_sum.cache_info().currsize, 1)
        self.assertEqual(calc.my_sum(1, 2, 3, 4, 5), 15)
        self.assertEqual(calc.my_sum.cache_info().hits, 1)
        # Make sure different args produce different results
        self.assertEqual(calc.my_sum(1, 2, 3, 4), 10)
        self.assertEqual(calc.my_sum.cache_info().currsize, 2)
        # Clear cache
        calc.my_sum.cache_clear()
        self.assertEqual(calc.my_sum.cache_info().currsize, 0)
