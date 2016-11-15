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
