
import unittest.mock
from AnyQt.QtCore import Qt, QEvent
from AnyQt.QtTest import QTest
from AnyQt.QtWidgets import QWidget, QHBoxLayout, QStyle, QApplication

from Orange.widgets.tests.base import GuiTest
from Orange.widgets.utils.overlay import (
    OverlayWidget, MessageOverlayWidget
)

class TestOverlay(GuiTest):
    def test_overlay_message(self):
        container = QWidget()
        overlay = MessageOverlayWidget(parent=container)
        overlay.setWidget(container)
        overlay.setIcon(QStyle.SP_MessageBoxInformation)
        container.show()
        QTest.qWaitForWindowExposed(container)

        self.assertTrue(overlay.isVisible())

        overlay.setText("Hello world! It's so nice here")
        QApplication.sendPostedEvents(overlay, QEvent.LayoutRequest)
        self.assertTrue(overlay.geometry().isValid())

        button_ok = overlay.addButton(MessageOverlayWidget.Ok)
        button_close = overlay.addButton(MessageOverlayWidget.Close)
        button_help = overlay.addButton(MessageOverlayWidget.Help)

        self.assertTrue(all([button_ok, button_close, button_help]))
        self.assertIs(overlay.button(MessageOverlayWidget.Ok), button_ok)
        self.assertIs(overlay.button(MessageOverlayWidget.Close), button_close)
        self.assertIs(overlay.button(MessageOverlayWidget.Help), button_help)

        button = overlay.addButton("Click Me!",
                                   MessageOverlayWidget.AcceptRole)
        self.assertIsNot(button, None)
        self.assertTrue(overlay.buttonRole(button),
                        MessageOverlayWidget.AcceptRole)

        mock = unittest.mock.MagicMock()
        overlay.accepted.connect(mock)
        QTest.mouseClick(button, Qt.LeftButton)
        self.assertFalse(overlay.isVisible())

        mock.assert_called_once_with()

    def test_layout(self):
        container = QWidget()
        container.setLayout(QHBoxLayout())
        container1 = QWidget()
        container.layout().addWidget(container1)
        container.show()
        QTest.qWaitForWindowExposed(container)
        container.resize(600, 600)

        overlay = OverlayWidget(parent=container)
        overlay.setWidget(container)
        overlay.resize(20, 20)
        overlay.show()

        center = overlay.geometry().center()
        self.assertTrue(290 < center.x() < 310)
        self.assertTrue(290 < center.y() < 310)

        overlay.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        geom = overlay.geometry()
        self.assertEqual(geom.top(), 0)
        self.assertTrue(290 < geom.center().x() < 310)

        overlay.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        geom = overlay.geometry()
        self.assertEqual(geom.left(), 0)
        self.assertTrue(290 < geom.center().y() < 310)

        overlay.setAlignment(Qt.AlignBottom | Qt.AlignRight)
        geom = overlay.geometry()
        self.assertEqual(geom.right(), 600 - 1)
        self.assertEqual(geom.bottom(), 600 - 1)

        overlay.setWidget(container1)
        geom = overlay.geometry()

        self.assertEqual(geom.right(), container1.geometry().right())
        self.assertEqual(geom.bottom(), container1.geometry().bottom())
