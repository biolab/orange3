from Orange.data.io import FileFormats
from PyQt4 import QtGui, QtCore, QtSvg


class ImgFormat:
    @staticmethod
    def _get_buffer(size, filename):
        raise NotImplementedError(
            "Descendants of ImgFormat must override method _get_buffer")

    @staticmethod
    def _get_target(self, scene, painter, buffer):
        raise NotImplementedError(
            "Descendants of ImgFormat must override method _get_target")

    @staticmethod
    def _save_buffer(self, buffer, filename):
        raise NotImplementedError(
            "Descendants of ImgFormat must override method _get_target")

    @classmethod
    def write_image(cls, filename, scene):
        source = QtCore.QRectF()
        for item in scene.items():
            if item.isVisible():
                source = source.united(item.boundingRect().translated(item.pos()))
        source = source.adjusted(-15, -15, 15, 15)
        buffer = cls._get_buffer(source.size(), filename)

        painter = QtGui.QPainter()
        painter.begin(buffer)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        target = cls._get_target(scene, painter, buffer, source)
        scene.render(painter, target, source)
        cls._save_buffer(buffer, filename)
        painter.end()

    def write(self, filename, scene):
        if type(scene) == dict:
            scene = scene['scene']
        self.write_image(filename, scene)


@FileFormats.register("Portable Network Graphics", ".png")
class PngFormat(ImgFormat):
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


@FileFormats.register("Scalable Vector Graphics", ".svg")
class SvgFormat(ImgFormat):
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
