# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.widgets.classify.owclassificationtree import OWClassificationTree
from Orange.widgets.tests.base import (WidgetTest, WidgetLearnerTestMixin,
                                       GuiToParam)


class TestOWClassificationTree(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWClassificationTree,
                                         stored_settings={"auto_apply": False})
        self.init()

        def combo_set_value(i, x):
            x.activated.emit(i)
            x.setCurrentIndex(i)

        scores = [score[1] for score in self.widget.scores]
        md_spin = self.widget.max_depth_spin[1]
        mi_spin = self.widget.min_internal_spin[1]
        ml_spin = self.widget.min_leaf_spin[1]
        md_min_max = [md_spin.minimum(), md_spin.maximum()]
        mi_min_max = [mi_spin.minimum(), mi_spin.maximum()]
        ml_min_max = [ml_spin.minimum(), ml_spin.maximum()]
        self.gui_to_params = [
            GuiToParam('criterion', self.widget.score_combo,
                       lambda x: scores[x.currentIndex()],
                       combo_set_value, scores, list(range(len(scores)))),
            GuiToParam('max_depth', md_spin, lambda x: x.value(),
                       lambda i, x: x.setValue(i), md_min_max, md_min_max),
            GuiToParam('min_samples_split', mi_spin, lambda x: x.value(),
                       lambda i, x: x.setValue(i), mi_min_max, mi_min_max),
            GuiToParam('min_samples_leaf', ml_spin, lambda x: x.value(),
                       lambda i, x: x.setValue(i), ml_min_max, ml_min_max)]

    def test_parameters_unchecked(self):
        """Check learner and model for various values of all parameters
        when pruning parameters are not checked
        """
        self.widget.max_depth_spin[0].setCheckState(False)
        self.widget.min_internal_spin[0].setCheckState(False)
        self.widget.min_leaf_spin[0].setCheckState(False)
        for i, val in ((1, None), (2, 2), (3, 1)):
            el = self.gui_to_params[i]
            self.gui_to_params[i] = GuiToParam(
                el.name, el.gui_el, lambda x, value=val: value,
                el.set, [val], el.set_values[0:1])
        self.test_parameters()
