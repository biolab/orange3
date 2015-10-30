from PyQt4 import QtGui, QtCore, QtSvg

from Orange.util import abstract
from Orange.data.io import FileFormat

from pyqtgraph.graphicsItems.GraphicsWidget import GraphicsWidget
from pyqtgraph.exporters.SVGExporter import SVGExporter
from pyqtgraph.exporters.ImageExporter import ImageExporter


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

    @classmethod
    def write_image(cls, filename, scene):
        if isinstance(scene, GraphicsWidget):
            exporter = cls._get_exporter()
            exp = exporter(scene)
            exp.export(filename)
        else:
            source = scene.itemsBoundingRect().adjusted(-15, -15, 15, 15)
            buffer = cls._get_buffer(source.size(), filename)

            painter = QtGui.QPainter()
            painter.begin(buffer)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)

            target = cls._get_target(scene, painter, buffer, source)
            scene.render(painter, target, source)
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

    @staticmethod
    def _get_buffer(size, filename):
        return QtGui.QPixmap(int(size.width()), int(size.height()))

    @staticmethod
    def _get_target(scene, painter, buffer, source):
        brush = scene.backgroundBrush()
        if brush.style() == QtCore.Qt.NoBrush:
            brush = QtGui.QBrush(scene.palette().color(QtGui.QPalette.Base))
        painter.fillRect(buffer.rect(), brush)
        return QtCore.QRectF(0, 0, source.width(), source.height())

    @staticmethod
    def _save_buffer(buffer, filename):
        buffer.save(filename)

    @staticmethod
    def _get_exporter():
        return ImageExporter


class SvgFormat(ImgFormat):
    EXTENSIONS = ('.svg',)
    DESCRIPTION = 'Scalable Vector Graphics'

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
        return SVGExporter
