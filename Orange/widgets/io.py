from PyQt4 import QtGui, QtCore, QtSvg
from PyQt4.QtGui import QGraphicsScene, QGraphicsView, QWidget

from Orange.util import abstract
from Orange.data.io import FileFormat


@abstract
class ImgFormat(FileFormat):
    @staticmethod
    @abstract
    def _get_buffer(size, filename):
        pass

    @staticmethod
    @abstract
    def _get_target(scene, painter, buffer):
        pass

    @staticmethod
    @abstract
    def _save_buffer(buffer, filename):
        pass

    @staticmethod
    @abstract
    def _get_exporter():
        pass

    @staticmethod
    @abstract
    def _export(self, exporter, filename):
        pass

    @classmethod
    def write_image(cls, filename, scene):
        try:
            exporter = cls._get_exporter()
            cls._export(exporter(scene), filename)
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
                scene.render(painter)  # PyQt4 QWidget.render() takes different params
            cls._save_buffer(buffer, filename)
            painter.end()

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
        pass

    @staticmethod
    def _get_exporter():
        from pyqtgraph.exporters.SVGExporter import SVGExporter
        return SVGExporter

    @staticmethod
    def _export(exporter, filename):
        exporter.export(filename)
