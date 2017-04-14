import os
import tempfile

from AnyQt import QtGui, QtCore, QtSvg
from AnyQt.QtCore import QMimeData
from AnyQt.QtWidgets import (
    QGraphicsScene, QGraphicsView, QWidget, QApplication
)

from Orange.data.io import FileFormat
from Orange.widgets.utils.webview import WebviewWidget


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
        from pyqtgraph.exporters.SVGExporter import SVGExporter
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
