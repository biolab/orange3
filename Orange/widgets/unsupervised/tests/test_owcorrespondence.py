# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
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

    def test_data_values_in_column(self):
        """
        Check that the widget does not crash when:
        1) Domain has a two or more discrete variables but less than in a table
        2) There is at least one NaN value in a column.
        GH-2066
        """
        table = Table(
            Domain(
                [ContinuousVariable("a"),
                 DiscreteVariable("b", values=["t", "f"]),
                 DiscreteVariable("c", values=["y", "n"]),
                 DiscreteVariable("d", values=["k", "l", "z"])]
            ),
            list(zip(
                [42.48, 16.84, 15.23, 23.8],
                ["t", "t", "", "f"],
                "yyyy",
                "klkk"
            )))
        self.send_signal("Data", table)

    def test_data_one_value_zero(self):
        """
        Check that the widget does not crash on discrete attributes with only
        one value.
        GH-2149
        """
        table = Table(
            Domain(
                [DiscreteVariable("a", values=["0"])]
            ),
            [(0,), (0,), (0,)]
        )
        self.send_signal("Data", table)
