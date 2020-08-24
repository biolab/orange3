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
            if action is not None and action.toolTip():
                text = tooltip_with_shortcut(action.toolTip(),
                                             action.shortcut())
                QToolTip.showText(event.globalPos(), text)
                return True
        return super().event(event)


def tooltip_with_shortcut(tool_tip, shortcut: QKeySequence) -> str:
    text = []
    if tool_tip:
        text.append("<span>{}</span>".format(tool_tip))
    if not shortcut.isEmpty():
        text.append("<kbd>{}</kbd>"
                    .format(shortcut.toString(QKeySequence.NativeText)))
    return "&nbsp;&nbsp;".join(text)
