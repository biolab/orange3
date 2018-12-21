import os
import tempfile
import unittest
from unittest.mock import patch, Mock

import pyqtgraph
import pyqtgraph.exporters
from AnyQt.QtWidgets import QGraphicsScene, QGraphicsRectItem
from AnyQt.QtGui import QImage

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


class TestImgFormat(GuiTest):

    def test_pyqtgraph_exporter(self):
        graph = pyqtgraph.PlotWidget()
        with patch("Orange.widgets.io.ImgFormat._get_exporter",
                   Mock()) as mfn:
            with self.assertRaises(Exception):
                imgio.ImgFormat.write("", graph)
            self.assertEqual(1, mfn.call_count)  # run pyqtgraph exporter

    def test_other_exporter(self):
        sc = QGraphicsScene()
        sc.addItem(QGraphicsRectItem(0, 0, 3, 3))
        with patch("Orange.widgets.io.ImgFormat._get_exporter",
                   Mock()) as mfn:
            with self.assertRaises(Exception):
                imgio.ImgFormat.write("", sc)
            self.assertEqual(0, mfn.call_count)


class TestPng(GuiTest):

    def test_pyqtgraph(self):
        fd, fname = tempfile.mkstemp('.png')
        os.close(fd)
        graph = pyqtgraph.PlotWidget()
        try:
            imgio.PngFormat.write(fname, graph)
            im = QImage(fname)
            self.assertLess((200, 200), (im.width(), im.height()))
        finally:
            os.unlink(fname)

    def test_other(self):
        fd, fname = tempfile.mkstemp('.png')
        os.close(fd)
        sc = QGraphicsScene()
        sc.addItem(QGraphicsRectItem(0, 0, 3, 3))
        try:
            imgio.PngFormat.write(fname, sc)
            im = QImage(fname)
            # writer adds 15*2 of empty space
            self.assertEqual((30+4, 30+4), (im.width(), im.height()))
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
