# pylint: disable=protected-access
import numpy as np

from Orange.base import Model
from Orange.data import Table
from Orange.widgets.model.owtree import OWTreeLearner
from Orange.widgets.tests.base import (
    DefaultParameterMapping,
    ParameterMapping,
    WidgetLearnerTestMixin,
    WidgetTest,
)


class TestOWClassificationTree(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(
            OWTreeLearner, stored_settings={"auto_apply": False})
        self.init()
        self.model_class = Model

        self.parameters = [
            ParameterMapping.from_attribute(self.widget, 'max_depth'),
            ParameterMapping.from_attribute(
                self.widget, 'min_internal', 'min_samples_split'),
            ParameterMapping.from_attribute(
                self.widget, 'min_leaf', 'min_samples_leaf')]
        # NB. sufficient_majority is divided by 100, so it cannot be tested
        # like this

        self.checks = [sb.gui_element.cbox for sb in self.parameters]

    def test_parameters_unchecked(self):
        """Check learner and model for various values of all parameters
        when pruning parameters are not checked
        """
        for cb in self.checks:
            cb.setCheckState(False)
        self.parameters = [DefaultParameterMapping(par.name, val)
                           for par, val in zip(self.parameters, (None, 2, 1))]
        self.test_parameters()

    def test_sparse_data_classification(self):
        """
        Classification Tree can handle sparse data.
        GH-2430
        """
        table1 = Table("iris")
        self.send_signal("Data", table1)
        model_dense = self.get_output("Model")
        table2 = Table("iris").to_sparse()
        self.send_signal("Data", table2)
        model_sparse = self.get_output("Model")
        self.assertTrue(np.array_equal(model_dense._code, model_sparse._code))
        self.assertTrue(np.array_equal(model_dense._values, model_sparse._values))

    def test_sparse_data_regression(self):
        """
        Regression Tree can handle sparse data.
        GH-2497
        """
        table1 = Table("housing")
        self.send_signal("Data", table1)
        model_dense = self.get_output("Model")
        table2 = Table("housing").to_sparse()
        self.send_signal("Data", table2)
        model_sparse = self.get_output("Model")
        self.assertTrue(np.array_equal(model_dense._code, model_sparse._code))
        self.assertTrue(np.array_equal(model_dense._values, model_sparse._values))
