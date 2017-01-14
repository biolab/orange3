# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owcorrespondence \
    import OWCorrespondenceAnalysis


class TestOWCorrespondence(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWCorrespondenceAnalysis)

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal("Data", Table(Table("iris").domain))
        self.assertTrue(self.widget.Error.empty_data.is_shown())
        self.assertIsNone(self.widget.data)
