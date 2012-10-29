import unittest

from Orange import testing
from Orange import data


class PickleContinuousVariable(testing.PickleTest):
    def generate_objects(self):
        yield data.ContinuousVariable()
        yield data.ContinuousVariable(name="Feature 0")

    def assertEqual(self, original, unpickled, msg=None):
        super().assertEqual(original.var_type, unpickled.var_type)
        super().assertEqual(original.name, unpickled.name)


class PickleDiscreteVariable(testing.PickleTest):
    def generate_objects(self):
        yield data.DiscreteVariable()
        yield data.DiscreteVariable(name="Feature 0")
        yield data.DiscreteVariable(name="Feature 0", values=[1,2,3])
        yield data.DiscreteVariable(name="Feature 0", values=["F", "M"])
        yield data.DiscreteVariable(name="Feature 0", values=["F", "M"], ordered=True)
        yield data.DiscreteVariable(name="Feature 0", values=["F", "M"], base_value=0)


    def assertEqual(self, original, unpickled, msg=None):
        super().assertEqual(original.var_type, unpickled.var_type)
        super().assertEqual(original.name, unpickled.name)
        super().assertEqual(original.values, unpickled.values)
        super().assertEqual(original.ordered, unpickled.ordered)
        super().assertEqual(original.base_value, unpickled.base_value)

class PickleStringVariable(testing.PickleTest):
    def generate_objects(self):
        yield data.StringVariable()
        yield data.StringVariable(name="Feature 0")

    def assertEqual(self, original, unpickled, msg=None):
        super().assertEqual(original.var_type, unpickled.var_type)
        super().assertEqual(original.name, unpickled.name)

if __name__ == "__main__":
    unittest.main()