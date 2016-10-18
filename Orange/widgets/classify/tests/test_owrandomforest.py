# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.widgets.classify.owrandomforest import OWRandomForest
from Orange.widgets.tests.base import (WidgetTest, DefaultParameterMapping,
                                       ParameterMapping, WidgetLearnerTestMixin)


class TestOWRandomForest(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWRandomForest,
                                         stored_settings={"auto_apply": False})
        self.init()
        nest_spin = self.widget.n_estimators_spin
        nest_min_max = [nest_spin.minimum() * 10, nest_spin.minimum()]
        self.parameters = [
            ParameterMapping("n_estimators", nest_spin, nest_min_max),
            ParameterMapping("min_samples_split",
                             self.widget.min_samples_split_spin[1])]

    def test_parameters_checked(self):
        """Check learner and model for various values of all parameters
        when all properties are checked
        """
        self.widget.max_features_spin[0].setCheckState(True)
        self.widget.random_state_spin[0].setCheckState(True)
        self.widget.max_depth_spin[0].setCheckState(True)
        self.parameters.extend([
            ParameterMapping("max_features", self.widget.max_features_spin[1]),
            ParameterMapping("random_state", self.widget.random_state_spin[1]),
            ParameterMapping("max_depth", self.widget.max_depth_spin[1])])
        self.test_parameters()

    def test_parameters_unchecked(self):
        """Check learner and model for various values of all parameters
        when properties are not checked
        """
        self.widget.min_samples_split_spin[0].setCheckState(False)
        self.parameters = self.parameters[:1]
        self.parameters.extend([
            DefaultParameterMapping("max_features", "auto"),
            DefaultParameterMapping("random_state", None),
            DefaultParameterMapping("max_depth", None),
            DefaultParameterMapping("min_samples_split", 2)])
        self.test_parameters()
