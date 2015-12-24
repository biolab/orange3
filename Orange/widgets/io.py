from PyQt4 import QtGui, QtCore, QtSvg

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
    def _export(self, exporter, filename):
        raise NotImplementedError(
            "Descendants of ImgFormat must override method _export")

    @classmethod
    def write_image(cls, filename, scene):
        from pyqtgraph.graphicsItems.GraphicsWidget import GraphicsWidget

        try:
            exporter = cls._get_exporter()
            cls._export(exporter(scene), filename)
        except:
            source = scene.sceneRect().adjusted(-15, -15, 15, 15)
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
    OWSAVE_PRIORITY = 50

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
    OWSAVE_PRIORITY = 100

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
