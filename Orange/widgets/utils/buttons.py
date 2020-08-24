from AnyQt.QtCore import Qt, QSize, QEvent
from AnyQt.QtGui import QKeySequence
from AnyQt.QtWidgets import QToolButton, QSizePolicy, QStyle, QToolTip

from orangewidget.utils.buttons import (
    VariableTextPushButton, SimpleButton,
)
__all__ = [
    "VariableTextPushButton", "SimpleButton", "FixedSizeButton",
]


class FixedSizeButton(QToolButton):

    def __init__(self, *args, defaultAction=None, **kwargs):
        super().__init__(*args, **kwargs)
        sh = self.sizePolicy()
        sh.setHorizontalPolicy(QSizePolicy.Fixed)
        sh.setVerticalPolicy(QSizePolicy.Fixed)
        self.setSizePolicy(sh)
        self.setAttribute(Qt.WA_WState_OwnSizePolicy, True)

        if defaultAction is not None:
            self.setDefaultAction(defaultAction)

    def sizeHint(self):
        style = self.style()
        size = (style.pixelMetric(QStyle.PM_SmallIconSize) +
                style.pixelMetric(QStyle.PM_ButtonMargin))
        return QSize(size, size)

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.ToolTip and self.toolTip():
            action = self.defaultAction()
            if action is not None:
                text = "<span>{}</span>&nbsp;&nbsp;<kbd>{}</kbd>".format(
                    action.toolTip(),
                    action.shortcut().toString(QKeySequence.NativeText)
                )
                QToolTip.showText(event.globalPos(), text)
            else:
                QToolTip.hideText()
            return True
        else:
            return super().event(event)
