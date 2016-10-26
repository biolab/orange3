# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.widgets.classify.owclassificationtree import OWClassificationTree
from Orange.widgets.tests.base import (WidgetTest, DefaultParameterMapping,
                                       ParameterMapping, WidgetLearnerTestMixin)


class TestOWClassificationTree(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget = self.create_widget(OWClassificationTree,
                                         stored_settings={"auto_apply": False})
        self.init()
        scores = [score[1] for score in self.widget.scores]
        self.parameters = [
            ParameterMapping('criterion', self.widget.score_combo, scores),
            ParameterMapping('max_depth', self.widget.max_depth_spin[1]),
            ParameterMapping('min_samples_split',
                             self.widget.min_internal_spin[1]),
            ParameterMapping('min_samples_leaf', self.widget.min_leaf_spin[1])]

    def test_parameters_unchecked(self):
        """Check learner and model for various values of all parameters
        when pruning parameters are not checked
        """
        self.widget.max_depth_spin[0].setCheckState(False)
        self.widget.min_internal_spin[0].setCheckState(False)
        self.widget.min_leaf_spin[0].setCheckState(False)
        for i, val in ((1, None), (2, 2), (3, 1)):
            self.parameters[i] = DefaultParameterMapping(
                self.parameters[i].name, val)
        self.test_parameters()
