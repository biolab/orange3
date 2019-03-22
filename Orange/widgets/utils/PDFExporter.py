from pyqtgraph.exporters.Exporter import Exporter

from AnyQt.QtGui import QPainter, QGraphicsItem, QDesktopWidget
from AnyQt.QtCore import QMarginsF, Qt, QSizeF, QRectF


class PDFExporter(Exporter):
    """A pdf exporter for pyqtgraph graphs. Based on pyqtgraph's
     ImageExporter.

     There is a bug in Qt<5.12 that makes Qt wrongly use a cosmetic pen
     (QTBUG-68537). Workaround: do not use completely opaque colors.

     There is also a bug in Qt<5.12 with bold fonts that then remain bold.
     To see it, save the OWNomogram output."""

    def __init__(self, item):
        Exporter.__init__(self, item)
        if isinstance(item, QGraphicsItem):
            scene = item.scene()
        else:
            scene = item
        bgbrush = scene.views()[0].backgroundBrush()
        bg = bgbrush.color()
        if bgbrush.style() == Qt.NoBrush:
            bg.setAlpha(0)
        self.background = bg

    def export(self, filename=None):
        from AnyQt.QtGui import QPdfWriter

        pw = QPdfWriter(filename)
        dpi = QDesktopWidget().logicalDpiX()
        pw.setResolution(dpi)
        pw.setPageMargins(QMarginsF(0, 0, 0, 0))
        pw.setPageSizeMM(QSizeF(self.getTargetRect().size()) / dpi * 25.4)

        painter = QPainter(pw)
        try:
            self.setExportMode(True, {'antialias': True,
                                      'background': self.background,
                                      'painter': painter})
            painter.setRenderHint(QPainter.Antialiasing, True)
            self.getScene().render(painter,
                                   QRectF(self.getTargetRect()),
                                   QRectF(self.getSourceRect()))
        finally:
            self.setExportMode(False)
        painter.end()
