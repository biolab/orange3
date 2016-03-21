from AnyQt import QtGui, QtCore, QtSvg
from AnyQt.QtCore import QMimeData
from AnyQt.QtWidgets import (
    QGraphicsScene, QGraphicsView, QWidget, QApplication
)

from Orange.data.io import FileFormat


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
        pass

    @staticmethod
    def _get_exporter():
        from pyqtgraph.exporters.SVGExporter import SVGExporter
        return SVGExporter

    @staticmethod
    def _export(exporter, filename):
        exporter.export(filename)
