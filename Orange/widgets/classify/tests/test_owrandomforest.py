# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.widgets.classify.owrandomforest import OWRandomForest
from Orange.widgets.tests.base import (WidgetTest, WidgetLearnerTestMixin,
                                       GuiToParam)


class TestOWRandomForest(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWRandomForest,
                                         stored_settings={"auto_apply": False})
        self.init()
        n_est_spin = self.widget.n_estimators_spin
        max_f_spin = self.widget.max_features_spin[1]
        rs_spin = self.widget.random_state_spin[1]
        max_d_spin = self.widget.max_depth_spin[1]
        min_s_spin = self.widget.min_samples_split_spin[1]
        n_est_min_max = [n_est_spin.minimum() * 10, n_est_spin.minimum()]
        min_s_min_max = [min_s_spin.minimum(), min_s_spin.maximum()]
        self.gui_to_params = [
            GuiToParam("n_estimators", n_est_spin, lambda x: x.value(),
                       lambda i, x: x.setValue(i), n_est_min_max, n_est_min_max),
            GuiToParam("max_features", max_f_spin, lambda x: "auto",
                       lambda i, x: x.setValue(i), ["auto"], [0]),
            GuiToParam("random_state", rs_spin, lambda x: None,
                       lambda i, x: x.setValue(i), [None], [0]),
            GuiToParam("max_depth", max_d_spin, lambda x: None,
                       lambda i, x: x.setValue(i), [None], [0]),
            GuiToParam("min_samples_split", min_s_spin, lambda x: x.value(),
                       lambda i, x: x.setValue(i), min_s_min_max, min_s_min_max)]

    def test_parameters_checked(self):
        """Check learner and model for various values of all parameters
        when all properties are checked
        """
        self.widget.max_features_spin[0].click()
        self.widget.random_state_spin[0].click()
        self.widget.max_depth_spin[0].click()
        for j in range(1, 4):
            el = self.gui_to_params[j]
            el_min_max = [el.gui_el.minimum(), el.gui_el.maximum()]
            self.gui_to_params[j] = GuiToParam(
                el.name, el.gui_el, lambda x: x.value(),
                lambda i, x: x.setValue(i), el_min_max, el_min_max)
        self.test_parameters()

    def test_parameters_unchecked(self):
        """Check learner and model for various values of all parameters
        when properties are not checked
        """
        self.widget.min_samples_split_spin[0].click()
        el = self.gui_to_params[4]
        self.gui_to_params[4] = GuiToParam(el.name, el.gui_el, lambda x: 2,
                                           lambda i, x: x.setValue(i), [2], [0])
        self.test_parameters()
