# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import Table, Domain, DiscreteVariable
from Orange.classification.tree import TreeLearner as ClassificationTreeLearner
from Orange.base import Model
from Orange.widgets.classify.owclassificationtree import OWClassificationTree
from Orange.widgets.tests.base import (WidgetTest, DefaultParameterMapping,
                                       ParameterMapping, WidgetLearnerTestMixin)


class TestOWClassificationTree(WidgetTest, WidgetLearnerTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris")

    def setUp(self):
        self.widget = self.create_widget(OWClassificationTree,
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

    def test_cannot_binarize(self):
        widget = self.widget
        error_shown = widget.Error.cannot_binarize.is_shown
        self.assertFalse(error_shown())
        self.send_signal("Data", self.iris)

        # The widget outputs ClassificationTreeLearner.
        # If not, below tests may not make sense
        learner = self.get_output("Learner")
        self.assertTrue(learner, ClassificationTreeLearner)

        # No error on Iris
        max_binarization = learner.MAX_BINARIZATION
        self.assertFalse(error_shown())

        # Error when too many values
        domain = Domain([
            DiscreteVariable(
                values=[str(x) for x in range(max_binarization + 1)])],
            DiscreteVariable(values="01"))
        self.send_signal("Data", Table(domain, [[0, 0], [1, 1]]))
        self.assertTrue(error_shown())
        # No more error on Iris
        self.send_signal("Data", self.iris)
        self.assertFalse(error_shown())

        # Checking and unchecking binarization works
        widget.controls.binary_trees.click()
        self.assertFalse(widget.binary_trees)
        widget.unconditional_apply()
        self.send_signal("Data", Table(domain, [[0, 0], [1, 1]]))
        self.assertFalse(error_shown())
        widget.controls.binary_trees.click()
        widget.unconditional_apply()
        self.assertTrue(error_shown())
        widget.controls.binary_trees.click()
        widget.unconditional_apply()
        self.assertFalse(error_shown())

        # If something is wrong with the data, no error appears
        domain = Domain([
            DiscreteVariable(
                values=[str(x) for x in range(max_binarization + 1)])],
            DiscreteVariable(values="01"))
        self.send_signal("Data", Table(domain))
        self.assertFalse(error_shown())
