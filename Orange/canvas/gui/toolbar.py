"""
A custom toolbar.

"""
from __future__ import division

import logging

from PyQt4.QtGui import QToolBar

from PyQt4.QtCore import Qt, QSize, QEvent

log = logging.getLogger(__name__)


class DynamicResizeToolBar(QToolBar):
    """A QToolBar subclass that dynamically resizes its tool buttons
    to fit available space (this is done by setting fixed size on the
    button instances).

    .. note:: the class does not support `QWidgetAction`s, separators, etc.

    """

    def __init__(self, parent=None, *args, **kwargs):
        QToolBar.__init__(self, *args, **kwargs)

    def resizeEvent(self, event):
        QToolBar.resizeEvent(self, event)
        size = event.size()
        self.__layout(size)

    def actionEvent(self, event):
        QToolBar.actionEvent(self, event)
        if event.type() == QEvent.ActionAdded or \
                event.type() == QEvent.ActionRemoved:
            self.__layout(self.size())

    def sizeHint(self):
        hint = QToolBar.sizeHint(self)
        width, height = hint.width(), hint.height()
        dx1, dy1, dw1, dh1 = self.getContentsMargins()
        dx2, dy2, dw2, dh2 = self.layout().getContentsMargins()
        dx, dy = dx1 + dx2, dy1 + dy2
        dw, dh = dw1 + dw2, dh1 + dh2

        count = len(self.actions())
        spacing = self.layout().spacing()
        space_spacing = max(count - 1, 0) * spacing

        if self.orientation() == Qt.Horizontal:
            width = int(height * 1.618) * count + space_spacing + dw + dx
        else:
            height = int(width * 1.618) * count + space_spacing + dh + dy
        return QSize(width, height)

    def __layout(self, size):
        """Layout the buttons to fit inside size.
        """
        mygeom = self.geometry()
        mygeom.setSize(size)

        # Adjust for margins (both the widgets and the layouts.
        dx, dy, dw, dh = self.getContentsMargins()
        mygeom.adjust(dx, dy, -dw, -dh)

        dx, dy, dw, dh = self.layout().getContentsMargins()
        mygeom.adjust(dx, dy, -dw, -dh)

        actions = self.actions()
        widgets = map(self.widgetForAction, actions)

        orientation = self.orientation()
        if orientation == Qt.Horizontal:
            widgets = sorted(widgets, key=lambda w: w.pos().x())
        else:
            widgets = sorted(widgets, key=lambda w: w.pos().y())

        spacing = self.layout().spacing()
        uniform_layout_helper(widgets, mygeom, orientation,
                              spacing=spacing)


def uniform_layout_helper(items, contents_rect, expanding, spacing):
    """Set fixed sizes on 'items' so they can be lay out in
    contents rect anf fil the whole space.

    """
    if len(items) == 0:
        return

    spacing_space = (len(items) - 1) * spacing

    if expanding == Qt.Horizontal:
        space = contents_rect.width() - spacing_space
        setter = lambda w, s: w.setFixedWidth(max(s, 0))
    else:
        space = contents_rect.height() - spacing_space
        setter = lambda w, s: w.setFixedHeight(max(s, 0))

    base_size = space / len(items)
    remainder = space % len(items)

    for i, item in enumerate(items):
        item_size = base_size + (1 if i < remainder else 0)
        setter(item, item_size)
