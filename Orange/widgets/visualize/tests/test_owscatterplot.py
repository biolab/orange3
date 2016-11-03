# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import numpy as np

from AnyQt.QtCore import QRectF

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.visualize.owscatterplot import \
    OWScatterPlot, ScatterPlotVizRank


class TestOWScatterPlot(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_name = "Data"
        cls.signal_data = cls.data

    def setUp(self):
        self.widget = self.create_widget(OWScatterPlot)

    def test_set_data(self):
        # Connect iris to scatter plot
        self.send_signal("Data", self.data)

        # First two attribute should be selected as x an y
        self.assertEqual(self.widget.attr_x, self.data.domain[0])
        self.assertEqual(self.widget.attr_y, self.data.domain[1])

        # Class var should be selected as color
        self.assertIs(self.widget.graph.attr_color, self.data.domain.class_var)

        # Change which attributes are displayed
        self.widget.attr_x = self.data.domain[2]
        self.widget.attr_y = self.data.domain[3]

        # Disconnect the data
        self.send_signal("Data", None)

        # removing data should have cleared attributes
        self.assertEqual(self.widget.attr_x, None)
        self.assertEqual(self.widget.attr_y, None)
        self.assertEqual(self.widget.graph.attr_color, None)

        # and remove the legend
        self.assertEqual(self.widget.graph.legend, None)

        # Connect iris again
        # same attributes that were used last time should be selected
        self.send_signal("Data", self.data)

        self.assertIs(self.widget.attr_x, self.data.domain[2])
        self.assertIs(self.widget.attr_y, self.data.domain[3])

    def test_score_heuristics(self):
        domain = Domain([ContinuousVariable(c) for c in "abcd"],
                        DiscreteVariable("c", values="ab"))
        a = np.arange(10).reshape((10, 1))
        data = Table(domain, np.hstack([a, a, a, a]), a >= 5)
        self.send_signal("Data", data)
        vizrank = ScatterPlotVizRank(self.widget)
        self.assertEqual([x.name for x in vizrank.score_heuristic()],
                         list("abcd"))

    def test_optional_combos(self):
        domain = self.data.domain
        d1 = Domain(domain.attributes[:2], domain.class_var,
                   [domain.attributes[2]])
        t1 = Table(d1, self.data)
        self.send_signal("Data", t1)
        self.widget.graph.attr_size = domain.attributes[2]

        d2 = Domain(domain.attributes[:2], domain.class_var,
                    [domain.attributes[3]])
        t2 = Table(d2, self.data)
        self.send_signal("Data", t2)

    def _select_data(self):
        self.widget.graph.select_by_rectangle(QRectF(4, 3, 3, 1))
        return self.widget.graph.get_selection()

    def test_error_message(self):
        """Check if error message appears and then disappears when
        data is removed from input"""
        data = self.data.copy()
        data.X[:, 0] = np.nan
        self.send_signal("Data", data)
        self.assertTrue(self.widget.Warning.missing_coords.is_shown())
        self.send_signal("Data", None)
        self.assertFalse(self.widget.Warning.missing_coords.is_shown())
