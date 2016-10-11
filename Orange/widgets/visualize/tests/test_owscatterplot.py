# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import numpy as np

from PyQt4.QtCore import QRectF

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.visualize.owscatterplot import \
    OWScatterPlot, ScatterPlotVizRank


class TestOWScatterPlot(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWScatterPlot)
        self.iris = Table("iris")

    def test_set_data(self):
        # Connect iris to scatter plot
        self.send_signal("Data", self.iris)

        # First two attribute should be selected as x an y
        self.assertEqual(self.widget.attr_x, self.iris.domain[0])
        self.assertEqual(self.widget.attr_y, self.iris.domain[1])

        # Class var should be selected as color
        self.assertIs(self.widget.graph.attr_color, self.iris.domain.class_var)

        # Change which attributes are displayed
        self.widget.attr_x = self.iris.domain[2]
        self.widget.attr_y = self.iris.domain[3]

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
        self.send_signal("Data", self.iris)

        self.assertIs(self.widget.attr_x, self.iris.domain[2])
        self.assertIs(self.widget.attr_y, self.iris.domain[3])

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
        domain = self.iris.domain
        d1 = Domain(domain.attributes[:2], domain.class_var,
                   [domain.attributes[2]])
        t1 = Table(d1, self.iris)
        self.send_signal("Data", t1)
        self.widget.graph.attr_size = domain.attributes[2]

        d2 = Domain(domain.attributes[:2], domain.class_var,
                    [domain.attributes[3]])
        t2 = Table(d2, self.iris)
        self.send_signal("Data", t2)

    def test_outputs(self):
        self.send_signal("Data", self.iris)

        # check selected data output
        self.assertIsNone(self.get_output("Selected Data"))

        # check flagged data output
        flagged = self.get_output("Flagged Data")
        self.assertEqual(0, np.sum([i["Flag"] for i in flagged]))

        # select data points
        self.widget.graph.select_by_rectangle(QRectF(4, 3, 3, 1))

        # check selected data output
        selected = self.get_output("Selected Data")
        self.assertGreater(len(selected), 0)
        self.assertEqual(selected.domain, self.iris.domain)

        # check flagged data output
        flagged = self.get_output("Flagged Data")
        self.assertEqual(len(selected), np.sum([i["Flag"] for i in flagged]))

        # check output when data is removed
        self.send_signal("Data", None)
        self.assertIsNone(self.get_output("Selected Data"))
        self.assertIsNone(self.get_output("Flagged Data"))
