import unittest

from Orange.testing import create_pickling_tests
from Orange import data

age = data.ContinuousVariable(name="AGE")
gender = data.DiscreteVariable(name="Gender", values=["M", "F"])
income = data.ContinuousVariable(name="AGE")
education = data.DiscreteVariable(name="education", values=["GS","HS", "C"])
ssn = data.StringVariable(name="SSN")

PickleDomain = create_pickling_tests("PickleDomain",
    ("empty_domain",             lambda: data.Domain([])),
    ("with_continuous_variable", lambda: data.Domain([age])),
    ("with_discrete_variable",   lambda: data.Domain([gender])),
    ("with_mixed_variables",     lambda: data.Domain([age, gender])),
    ("with_continuous_class",    lambda: data.Domain([age, gender], [income])),
    ("with_discrete_class",      lambda: data.Domain([age, gender], [education])),
    ("with_multiple_classes",    lambda: data.Domain([age, gender], [income, education])),
    ("with_metas",               lambda: data.Domain([age, gender], metas=[ssn])),
    ("with_class_and_metas",     lambda: data.Domain([age, gender], [income, education], [ssn])),
)

if __name__ == "__main__":
    unittest.main()
