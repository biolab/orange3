from AnyQt.QtWidgets import QGraphicsRectItem, QGraphicsLineItem
from AnyQt.QtCore import QRectF, QMargins, QLineF

from . import TestItems

from ..controlpoints import ControlPoint, ControlPointRect, ControlPointLine


class TestControlPoints(TestItems):
    def test_controlpoint(self):
        point = ControlPoint()

        self.scene.addItem(point)

        point.setAnchor(ControlPoint.Left)
        self.assertEqual(point.anchor(), ControlPoint.Left)

    def test_controlpointrect(self):
        control = ControlPointRect()
        rect = QGraphicsRectItem(QRectF(10, 10, 100, 200))
        self.scene.addItem(rect)
        self.scene.addItem(control)

        control.setRect(rect.rect())
        control.setFocus()
        control.rectChanged.connect(rect.setRect)

        control.setRect(QRectF(20, 20, 100, 200))
        self.assertEqual(control.rect(), rect.rect())
        self.assertEqual(control.rect(), QRectF(20, 20, 100, 200))

        control.setControlMargins(5)
        self.assertEqual(control.controlMargins(), QMargins(5, 5, 5, 5))
        control.rectEdited.connect(rect.setRect)

        self.view.show()
        self.app.exec_()

        self.assertEqual(rect.rect(), control.rect())

    def test_controlpointline(self):
        control = ControlPointLine()
        line = QGraphicsLineItem(10, 10, 200, 200)

        self.scene.addItem(line)
        self.scene.addItem(control)

        control.setLine(line.line())
        control.setFocus()
        control.lineChanged.connect(line.setLine)

        control.setLine(QLineF(30, 30, 180, 180))
        self.assertEqual(control.line(), line.line())
        self.assertEqual(line.line(), QLineF(30, 30, 180, 180))

        control.lineEdited.connect(line.setLine)

        self.view.show()
        self.app.exec_()

        self.assertEqual(control.line(), line.line())
