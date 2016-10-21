from AnyQt.QtGui import QPainterPath, QBrush, QPen, QColor
from AnyQt.QtCore import QPointF

from . import TestItems

from ..graphicspathobject import GraphicsPathObject, shapeFromPath


def area(rect):
    return rect.width() * rect.height()


class TestGraphicsPathObject(TestItems):

    def test_graphicspathobject(self):
        obj = GraphicsPathObject()
        path = QPainterPath()
        obj.setFlag(GraphicsPathObject.ItemIsMovable)

        path.addEllipse(20, 20, 50, 50)

        obj.setPath(path)
        self.assertEqual(obj.path(), path)

        self.assertTrue(obj.path() is not path,
                        msg="setPath stores the path not a copy")

        brect = obj.boundingRect()
        self.assertTrue(brect.contains(path.boundingRect()))

        with self.assertRaises(TypeError):
            obj.setPath("This is not a path")

        brush = QBrush(QColor("#ffbb11"))
        obj.setBrush(brush)

        self.assertEqual(obj.brush(), brush)

        self.assertTrue(obj.brush() is not brush,
                        "setBrush stores the brush not a copy")

        pen = QPen(QColor("#FFFFFF"), 1.4)
        obj.setPen(pen)

        self.assertEqual(obj.pen(), pen)

        self.assertTrue(obj.pen() is not pen,
                        "setPen stores the pen not a copy")

        brect = obj.boundingRect()
        self.assertGreaterEqual(area(brect), (50 + 1.4 * 2) ** 2)

        self.assertIsInstance(obj.shape(), QPainterPath)

        positions = []
        obj.positionChanged[QPointF].connect(positions.append)

        pos = QPointF(10, 10)
        obj.setPos(pos)

        self.assertEqual(positions, [pos])

        self.scene.addItem(obj)
        self.view.show()

        self.app.exec_()

    def test_shapeFromPath(self):
        path = QPainterPath()
        path.addRect(10, 10, 20, 20)

        pen = QPen(QColor("#FFF"), 2.0)
        path = shapeFromPath(path, pen)

        self.assertGreaterEqual(area(path.controlPointRect()),
                                (20 + 2.0) ** 2)
