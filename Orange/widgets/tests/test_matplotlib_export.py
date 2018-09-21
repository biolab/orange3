import pyqtgraph as pg

from AnyQt.QtCore import QRectF

import Orange

from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.matplotlib_export import scatterplot_code
from Orange.widgets.visualize.owscatterplot import OWScatterPlot


def add_intro(a):
    r = "import matplotlib.pyplot as plt\n" + \
        "from numpy import array\n" + \
        "plt.clf()"
    return r + a


class TestScatterPlot(WidgetTest):

    def test_owscatterplot_ignore_empty(self):
        iris = Orange.data.Table("iris")
        self.widget = self.create_widget(OWScatterPlot)
        self.send_signal(OWScatterPlot.Inputs.data, iris[::10])
        code = scatterplot_code(self.widget.graph.scatterplot_item)
        self.assertIn("plt.scatter", code)
        # tbe selected graph has to generate nothing
        code = scatterplot_code(self.widget.graph.scatterplot_item_sel)
        self.assertEqual(code, "")
        # for a selection the selected graph has to be non-empty
        self.widget.graph.select_by_rectangle(QRectF(4, 3, 3, 1))
        code = scatterplot_code(self.widget.graph.scatterplot_item_sel)
        self.assertIn("plt.scatter", code)
        exec(add_intro(code), {})

    def test_scatterplot_simple(self):
        plotWidget = pg.PlotWidget(background="w")
        scatterplot = pg.ScatterPlotItem()
        scatterplot.setData(x=[1, 2, 3], y=[3, 2, 1])
        plotWidget.addItem(scatterplot)
        code = scatterplot_code(scatterplot)
        self.assertIn("plt.scatter", code)
        exec(add_intro(code), {})
