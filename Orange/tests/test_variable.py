import math
import unittest

from Orange.testing import create_pickling_tests
from Orange import data

class DiscreteVariableTest(unittest.TestCase):
    def test_to_val(self):
        values = ["F", "M"]
        var = data.DiscreteVariable(name="Feature 0", values=values)

        self.assertEqual(var.to_val(0), 0)
        self.assertEqual(var.to_val("F"), 0)
        self.assertEqual(var.to_val(0.), 0)
        self.assertTrue(math.isnan(var.to_val("?")))

        # TODO: with self.assertRaises(ValueError): var.to_val(2)
        with self.assertRaises(ValueError): var.to_val("G")




PickleContinuousVariable = create_pickling_tests("PickleContinuousVariable",
    ("variable",  lambda: data.ContinuousVariable()),
    ("with_name", lambda: data.ContinuousVariable(name="Feature 0")),
)

PickleDiscreteVariable = create_pickling_tests("PickleDiscreteVariable",
    ("variable",        lambda: data.DiscreteVariable()),
    ("with_name",       lambda: data.DiscreteVariable(name="Feature 0")),
    ("with_int_values", lambda: data.DiscreteVariable(name="Feature 0", values=[1,2,3])),
    ("with_str_value",  lambda: data.DiscreteVariable(name="Feature 0", values=["F", "M"])),
    ("ordered",         lambda: data.DiscreteVariable(name="Feature 0", values=["F", "M"], ordered=True)),
    ("with_base_value", lambda: data.DiscreteVariable(name="Feature 0", values=["F", "M"], base_value=0)),
)

PickleStringVariable = create_pickling_tests("PickleStringVariable",
    ("variable",  lambda: data.StringVariable()),
    ("with_name", lambda: data.StringVariable(name="Feature 0")),
)


if __name__ == "__main__":
    unittest.main()