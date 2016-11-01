# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.base import Model
from Orange.widgets.regression.owregressiontree import OWRegressionTree
from Orange.widgets.tests.base import (WidgetTest, DefaultParameterMapping,
                                       ParameterMapping, WidgetLearnerTestMixin)


class TestOWRegressionTree(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWRegressionTree,
                                         stored_settings={"auto_apply": False})
        self.init()
        self.model_class = Model
        self.parameters = [
            ParameterMapping.from_attribute(self.widget, 'max_depth'),
            ParameterMapping.from_attribute(
                self.widget, 'min_internal', 'min_samples_split'),
            ParameterMapping.from_attribute(
                self.widget, 'min_leaf', 'min_samples_leaf')]
        # NB. sufficient_majority is divided by 100, so it cannot be tested like
        # this

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
