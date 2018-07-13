import os
import tempfile
from warnings import warn

from AnyQt import QtGui, QtCore, QtSvg
from AnyQt.QtCore import QMimeData
from AnyQt.QtWidgets import (
    QGraphicsScene, QGraphicsView, QWidget, QApplication
)

from Orange.data.io import FileFormat

# Importing WebviewWidget can fail if neither QWebKit (old, deprecated) nor
# QWebEngine (bleeding-edge, hard to install) are available
try:
    from Orange.widgets.utils.webview import WebviewWidget
except ImportError:
    warn('WebView from QWebKit or QWebEngine is not available. Orange '
         'widgets that depend on it will fail to work.')
    WebviewWidget = None


class ImgFormat(FileFormat):
    @staticmethod
    def _get_buffer(size, filename):
        raise NotImplementedError

    @staticmethod
    def _get_target(scene, painter, buffer):
        raise NotImplementedError

    @staticmethod
    def _save_buffer(buffer, filename):
        raise NotImplementedError

    @staticmethod
    def _get_exporter():
        raise NotImplementedError

    @staticmethod
    def _export(self, exporter, filename):
        raise NotImplementedError

    @classmethod
    def write_image(cls, filename, scene):
        try:
            scene = scene.scene()
            scenerect = scene.sceneRect()   #preserve scene bounding rectangle
            viewrect = scene.views()[0].sceneRect()
            scene.setSceneRect(viewrect)
            backgroundbrush = scene.backgroundBrush()  #preserve scene background brush
            scene.setBackgroundBrush(QtCore.Qt.white)
            exporter = cls._get_exporter()
            cls._export(exporter(scene), filename)
            scene.setBackgroundBrush(backgroundbrush)  # reset scene background brush
            scene.setSceneRect(scenerect)   # reset scene bounding rectangle
        except Exception:
            if isinstance(scene, (QGraphicsScene, QGraphicsView)):
                rect = scene.sceneRect()
            elif isinstance(scene, QWidget):
                rect = scene.rect()
            rect = rect.adjusted(-15, -15, 15, 15)
            buffer = cls._get_buffer(rect.size(), filename)

            painter = QtGui.QPainter()
            painter.begin(buffer)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)

            target = cls._get_target(scene, painter, buffer, rect)
            try:
                scene.render(painter, target, rect)
            except TypeError:
                scene.render(painter)  # QWidget.render() takes different params
            painter.end()
            cls._save_buffer(buffer, filename)

    @classmethod
    def write(cls, filename, scene):
        if type(scene) == dict:
            scene = scene['scene']
        cls.write_image(filename, scene)


class PngFormat(ImgFormat):
    EXTENSIONS = ('.png',)
    DESCRIPTION = 'Portable Network Graphics'
    PRIORITY = 50

    @staticmethod
    def _get_buffer(size, filename):
        return QtGui.QPixmap(int(size.width()), int(size.height()))

    @staticmethod
    def _get_target(scene, painter, buffer, source):
        try:
            brush = scene.backgroundBrush()
            if brush.style() == QtCore.Qt.NoBrush:
                brush = QtGui.QBrush(scene.palette().color(QtGui.QPalette.Base))
        except AttributeError:  # not a QGraphicsView/Scene
            brush = QtGui.QBrush(QtCore.Qt.white)
        painter.fillRect(buffer.rect(), brush)
        return QtCore.QRectF(0, 0, source.width(), source.height())

    @staticmethod
    def _save_buffer(buffer, filename):
        buffer.save(filename, "png")

    @staticmethod
    def _get_exporter():
        from pyqtgraph.exporters.ImageExporter import ImageExporter
        return ImageExporter

    @staticmethod
    def _export(exporter, filename):
        buffer = exporter.export(toBytes=True)
        buffer.save(filename, "png")


class ClipboardFormat(PngFormat):
    EXTENSIONS = ()
    DESCRIPTION = 'System Clipboard'
    PRIORITY = 50

    @staticmethod
    def _save_buffer(buffer, _):
        QApplication.clipboard().setPixmap(buffer)

    @staticmethod
    def _export(exporter, _):
        buffer = exporter.export(toBytes=True)
        mimedata = QMimeData()
        mimedata.setData("image/png", buffer)
        QApplication.clipboard().setMimeData(mimedata)


class SvgFormat(ImgFormat):
    EXTENSIONS = ('.svg',)
    DESCRIPTION = 'Scalable Vector Graphics'
    PRIORITY = 100

    @staticmethod
    def _get_buffer(size, filename):
        buffer = QtSvg.QSvgGenerator()
        buffer.setFileName(filename)
        buffer.setSize(QtCore.QSize(int(size.width()), int(size.height())))
        return buffer

    @staticmethod
    def _get_target(scene, painter, buffer, source):
        return QtCore.QRectF(0, 0, source.width(), source.height())

    @staticmethod
    def _save_buffer(buffer, filename):
        dev = buffer.outputDevice()
        if dev is not None:
            dev.flush()
        pass

    @staticmethod
    def _get_exporter():
        from Orange.widgets.utils.SVGExporter import SVGExporter
        return SVGExporter

    @staticmethod
    def _export(exporter, filename):
        exporter.export(filename)

    @classmethod
    def write_image(cls, filename, scene):
        # WebviewWidget exposes its SVG contents more directly;
        # no need to go via QPainter if we can avoid it
        if isinstance(scene, WebviewWidget):
            try:
                svg = scene.svg()
                with open(filename, 'w') as f:
                    f.write(svg)
                return
            except (ValueError, IOError):
                pass

        super().write_image(filename, scene)


def matplotlib_output(scene):
    # for now just showing a graph

    from pyqtgraph.graphicsItems.ScatterPlotItem import ScatterPlotItem
    from itertools import chain

    import matplotlib.pyplot as plt

    plt.clf()

    for item in scene.items:
        if isinstance(item, ScatterPlotItem):
            draw_scatterplot(plt, item)

    # draw axes

    # FIXME code does not work for graphs without axes and for multiple axes!
    for position, set_ticks, set_label in [("bottom", plt.xticks, plt.xlabel), ("left", plt.yticks, plt.ylabel)]:
        axis = scene.getAxis(position)
        set_label(str(axis.labelText))

        # textual tick labels
        if axis._tickLevels is not None:
            major_minor = list(chain(*axis._tickLevels))
            locs = [a[0] for a in major_minor]
            labels = [a[1] for a in major_minor]
            set_ticks(locs, labels)

    plt.show()


def draw_scatterplot(plt, si):
    x = si.data['x']
    y = si.data['y']
    size = si.data["size"]

    def colortuple(color):
        return color.redF(), color.greenF(), color.blueF(), color.alphaF()

    import numpy as np

    edgecolor = np.array([colortuple(a.color()) for a in si.data["pen"]])
    facecolor = np.array([colortuple(a.color()) for a in si.data["brush"]])

    # each marker requires one call to matplotlib's scatter!

    def matplotlib_marker(m):
        if m == "t":
            return "^"
        elif m == "t2":
            return ">"
        elif m == "t3":
            return "<"
        elif m == "star":
            return "*"
        elif m == "+":
            return "P"
        elif m == "x":
            return "X"
        return m

    # possible_markers for scatterplot are in .graph.CurveSymbols

    # labels are missing

    markers = np.array([matplotlib_marker(m) for m in si.data["symbol"]])
    for m in set(markers):
        indices = np.where(markers == m)[0]
        plt.scatter(x=x[indices], y=y[indices], s=size[indices]**2/4, marker=m,
                    facecolors=facecolor[indices], edgecolors=edgecolor[indices])


class MatplotlibCode(FileFormat):
    EXTENSIONS = ('.py',)
    DESCRIPTION = 'Python code'
    PRIORITY = 100

    @classmethod
    def write_image(cls, filename, scene):
        print(filename, scene)
        matplotlib_output(scene)

    @classmethod
    def write(cls, filename, scene):
        if type(scene) == dict:
            scene = scene['scene']
        cls.write_image(filename, scene)


if hasattr(QtGui, "QPdfWriter"):
    class PdfFormat(ImgFormat):
        EXTENSIONS = ('.pdf', )
        DESCRIPTION = 'Portable Document Format'
        PRIORITY = 110

        @classmethod
        def write_image(cls, filename, scene):
            # export via svg to temp file then print that
            # NOTE: can't use NamedTemporaryFile with delete = True
            # (see https://bugs.python.org/issue14243)
            fd, tmpname = tempfile.mkstemp(suffix=".svg")
            os.close(fd)
            try:
                SvgFormat.write_image(tmpname, scene)
                with open(tmpname, "rb") as f:
                    svgcontents = f.read()
            finally:
                os.unlink(tmpname)

            svgrend = QtSvg.QSvgRenderer(QtCore.QByteArray(svgcontents))
            vbox = svgrend.viewBox()
            if not vbox.isValid():
                size = svgrend.defaultSize()
            else:
                size = vbox.size()
            writer = QtGui.QPdfWriter(filename)
            writer.setPageSizeMM(QtCore.QSizeF(size) * 0.282)
            painter = QtGui.QPainter(writer)
            svgrend.render(painter)
            painter.end()
            del svgrend
            del painter
