import os
import tempfile
import unittest

from AnyQt.QtWidgets import QGraphicsScene, QGraphicsRectItem
from Orange.widgets.tests.base import GuiTest

from Orange.widgets import io as imgio

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
