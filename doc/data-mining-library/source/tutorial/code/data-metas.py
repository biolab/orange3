from Orange.data import Table, Domain
from Orange.data import ContinuousVariable, DiscreteVariable, StringVariable
import numpy as np

X = np.array([[2.2, 1625], [0.3, 163]])
Y = np.array([0, 1])
M = np.array([["houston", 10], ["ljubljana", -1]])

domain = Domain(
    [ContinuousVariable("population"), ContinuousVariable("area")],
    [DiscreteVariable("snow", ("no", "yes"))],
    [StringVariable("city"), StringVariable("temperature")],
)
data = Table(domain, X, Y, M)
print(data)
