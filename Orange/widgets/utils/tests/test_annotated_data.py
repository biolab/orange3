import unittest

from Orange.widgets.utils.annotated_data import get_next_name

class AnnotatedDataTest(unittest.TestCase):
    def test_get_var_name(self):
        self.assertEqual(get_next_name(["a"], "XX"), "XX")
        self.assertEqual(get_next_name(["a", "XX"], "XX"), "XX (1)")
        self.assertEqual(get_next_name(["a", "XX (4)"], "XX"), "XX (5)")
        self.assertEqual(get_next_name(["a", "XX", "XX (4)"], "XX"), "XX (5)")
