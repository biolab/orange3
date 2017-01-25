# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import Table
from Orange.widgets.data.owdatasampler import OWDataSampler
from Orange.widgets.tests.base import WidgetTest


class TestOWDataSampler(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris")

    def setUp(self):
        self.widget = self.create_widget(OWDataSampler)

    def test_error_message(self):
        """ Check if error message appears and then disappears when
        data is removed from input"""
        self.widget.controls.sampling_type.buttons[2].click()
        self.send_signal("Data", self.iris)
        self.assertFalse(self.widget.Error.too_many_folds.is_shown())
        self.send_signal("Data", self.iris[:5])
        self.assertTrue(self.widget.Error.too_many_folds.is_shown())
        self.send_signal("Data", None)
        self.assertFalse(self.widget.Error.too_many_folds.is_shown())
        self.send_signal("Data", Table(self.iris.domain))
        self.assertTrue(self.widget.Error.no_data.is_shown())

    def test_stratified_on_unbalanced_data(self):
        unbalanced_data = self.iris[:51]

        self.widget.controls.stratify.setChecked(True)
        self.send_signal("Data", unbalanced_data)
        self.assertTrue(self.widget.Warning.could_not_stratify.is_shown())
