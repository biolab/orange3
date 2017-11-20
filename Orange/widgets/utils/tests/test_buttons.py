from AnyQt.QtCore import Qt
from AnyQt.QtGui import QFocusEvent
from AnyQt.QtWidgets import QStyle, QApplication
from Orange.widgets.tests.base import GuiTest
from Orange.widgets.utils import buttons


class SimpleButtonTest(GuiTest):
    def test_button(self):
        # Run through various state change and drawing code for coverage
        b = buttons.SimpleButton()
        b.setIcon(b.style().standardIcon(QStyle.SP_ComputerIcon))

        QApplication.sendEvent(b, QFocusEvent(QFocusEvent.FocusIn))
        QApplication.sendEvent(b, QFocusEvent(QFocusEvent.FocusOut))

        b.grab()
        b.setDown(True)
        b.grab()
        b.setCheckable(True)
        b.setChecked(True)
        b.grab()
