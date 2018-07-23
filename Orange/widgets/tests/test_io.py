import os
import tempfile
import unittest
from unittest.mock import patch

from AnyQt.QtWidgets import QGraphicsScene, QGraphicsRectItem

import Orange
from Orange.tests import named_file
from Orange.widgets.tests.base import GuiTest, WidgetTest

from Orange.widgets import io as imgio
from Orange.widgets.io import MatplotlibFormat, MatplotlibPDFFormat
from Orange.widgets.visualize.owscatterplot import OWScatterPlot


@unittest.skipUnless(hasattr(imgio, "PdfFormat"), "QPdfWriter not available")
class TestIO(GuiTest):
    def test_pdf(self):
        sc = QGraphicsScene()
        sc.addItem(QGraphicsRectItem(0, 0, 20, 20))
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        try:
            imgio.PdfFormat.write_image(fname, sc)
        finally:
            os.unlink(fname)


class TestMatplotlib(WidgetTest):

    def test_python(self):
        iris = Orange.data.Table("iris")
        self.widget = self.create_widget(OWScatterPlot)
        self.send_signal(OWScatterPlot.Inputs.data, iris[::10])
        with named_file("", suffix=".py") as fname:
            with patch("Orange.widgets.utils.filedialogs.open_filename_dialog_save",
                       lambda *x: (fname, MatplotlibFormat, None)):
                self.widget.save_graph()
                with open(fname, "rt") as f:
                    code = f.read()
                self.assertIn("plt.show()", code)
                self.assertIn("plt.scatter", code)
                # test if the runs
                exec(code.replace("plt.show()", ""), {})

    def test_pdf(self):
        iris = Orange.data.Table("iris")
        self.widget = self.create_widget(OWScatterPlot)
        self.send_signal(OWScatterPlot.Inputs.data, iris[::10])
        with named_file("", suffix=".pdf") as fname:
            with patch("Orange.widgets.utils.filedialogs.open_filename_dialog_save",
                       lambda *x: (fname, MatplotlibPDFFormat, None)):
                self.widget.save_graph()
                with open(fname, "rb") as f:
                    code = f.read()
                self.assertTrue(code.startswith(b"%PDF"))
