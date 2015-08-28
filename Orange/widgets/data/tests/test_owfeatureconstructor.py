import unittest

from Orange.data import (Table, Domain, StringVariable,
                         ContinuousVariable, DiscreteVariable)
from Orange.widgets.utils.itemmodels import PyListModel
from Orange.widgets.data.owfeatureconstructor import (DiscreteDescriptor,
                                                      ContinuousDescriptor,
                                                      StringDescriptor,
                                                      construct_variables)


class FeatureConstructorTest(unittest.TestCase):
    def test_construct_variables_discrete(self):
        data = Table("iris")
        name = 'Discrete Variable'
        expression = "iris_one if iris == 'Iris-setosa' else iris_two " \
                     "if iris == 'Iris-versicolor' else iris_three"
        values = ('iris one', 'iris two', 'iris three')
        desc = PyListModel(
            [DiscreteDescriptor(name=name, expression=expression,
                                values=values, base_value=-1, ordered=False)]
        )
        data = Table(Domain(list(data.domain.attributes) +
                            construct_variables(desc, data.domain),
                            data.domain.class_vars,
                            data.domain.metas), data)
        self.assertTrue(isinstance(data.domain[name], DiscreteVariable))
        self.assertEqual(data.domain[name].values, list(values))
        for i in range(3):
            self.assertEqual(data[i * 50, name], values[i])

    def test_construct_variables_continuous(self):
        data = Table("iris")
        name = 'Continuous Variable'
        expression = "pow(sepal_length + sepal_width, 2)"
        featuremodel = PyListModel(
            [ContinuousDescriptor(name=name, expression=expression,
                                  number_of_decimals=2)]
        )
        data = Table(Domain(list(data.domain.attributes) +
                            construct_variables(featuremodel, data.domain),
                            data.domain.class_vars,
                            data.domain.metas), data)
        self.assertTrue(isinstance(data.domain[name], ContinuousVariable))
        for i in range(3):
            self.assertEqual(data[i * 50, name],
                             pow(data[i * 50, 0] + data[i * 50, 1], 2))

    def test_construct_variables_string(self):
        data = Table("iris")
        name = 'String Variable'
        expression = "str(iris) + '_name'"
        desc = PyListModel(
            [StringDescriptor(name=name, expression=expression)]
        )
        data = Table(Domain(data.domain.attributes,
                            data.domain.class_vars,
                            list(data.domain.metas) +
                            construct_variables(desc, data.domain)),
                     data)
        self.assertTrue(isinstance(data.domain[name], StringVariable))
        for i in range(3):
            self.assertEqual(data[i * 50, name],
                             str(data[i * 50, "iris"]) + "_name")
