# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import numpy as np

from Orange.data import Table
from Orange.widgets.data.owpreprocess import OWPreprocess
from Orange.widgets.tests.base import WidgetTest


class TestOWPreprocess(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)
        self.zoo = Table("zoo")

    def test_randomize(self):
        saved = {"preprocessors": [("orange.preprocess.randomize",
                                    {"rand_type": 0, "rand_seed": 1})]}
        model = self.widget.load(saved)
        self.widget.set_model(model)
        self.send_signal("Data", self.zoo)
        output = self.get_output("Preprocessed Data")
        np.random.seed(1)
        np.random.shuffle(self.zoo.Y)
        np.testing.assert_array_equal(self.zoo.X, output.X)
        np.testing.assert_array_equal(self.zoo.Y, output.Y)
        np.testing.assert_array_equal(self.zoo.metas, output.metas)
