"""
=======================
Collapsible Dock Widget
=======================

A dock widget that can be a collapsed/expanded.

"""

import logging

from PyQt4.QtGui import (
    QDockWidget, QAbstractButton, QSizePolicy, QStyle, QIcon, QTransform
)

from PyQt4.QtCore import Qt, QEvent

from PyQt4.QtCore import pyqtProperty as Property, pyqtSignal as Signal

from .stackedwidget import AnimatedStackedWidget
from .utils import QWIDGETSIZE_MAX

log = logging.getLogger(__name__)


class CollapsibleDockWidget(QDockWidget):
    """
    This :class:`QDockWidget` subclass overrides the `close` header
    button to instead collapse to a smaller size. The contents contents
    to show when in each state can be set using the ``setExpandedWidget``
    and ``setCollapsedWidget``.

    .. note:: Do  not use the base class ``QDockWidget.setWidget`` method
              to set the docks contents. Use set[Expanded|Collapsed]Widget
              instead.

    """

    #: Emitted when the dock widget's expanded state changes.
    expandedChanged = Signal(bool)

    def __init__(self, *args, **kwargs):
        QDockWidget.__init__(self, *args, **kwargs)

        self.__expandedWidget = None
        self.__collapsedWidget = None
        self.__expanded = True

        self.__trueMinimumWidth = -1

        self.setFeatures(QDockWidget.DockWidgetClosable | \
                         QDockWidget.DockWidgetMovable)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.featuresChanged.connect(self.__onFeaturesChanged)
        self.dockLocationChanged.connect(self.__onDockLocationChanged)

        # Use the toolbar horizontal extension button icon as the default
        # for the expand/collapse button
        pm = self.style().standardPixmap(
            QStyle.SP_ToolBarHorizontalExtensionButton
        )

        # Rotate the icon
        transform = QTransform()
        transform.rotate(180)

        pm_rev = pm.transformed(transform)

        self.__iconRight = QIcon(pm)
        self.__iconLeft = QIcon(pm_rev)

        close = self.findChild(QAbstractButton,
                               name="qt_dockwidget_closebutton")

        close.installEventFilter(self)
        self.__closeButton = close

        self.__stack = AnimatedStackedWidget()

        self.__stack.setSizePolicy(QSizePolicy.Fixed,
                                   QSizePolicy.Expanding)

        self.__stack.transitionStarted.connect(self.__onTransitionStarted)
        self.__stack.transitionFinished.connect(self.__onTransitionFinished)

        QDockWidget.setWidget(self, self.__stack)

        self.__closeButton.setIcon(self.__iconLeft)

    def setExpanded(self, state):
        """
        Set the widgets `expanded` state.
        """
        if self.__expanded != state:
            self.__expanded = state
            if state and self.__expandedWidget is not None:
                log.debug("Dock expanding.")
                self.__stack.setCurrentWidget(self.__expandedWidget)
            elif not state and self.__collapsedWidget is not None:
                log.debug("Dock collapsing.")
                self.__stack.setCurrentWidget(self.__collapsedWidget)
            self.__fixIcon()

            self.expandedChanged.emit(state)

    def expanded(self):
        """
        Is the dock widget in expanded state. If `True` the
        ``expandedWidget`` will be shown, and ``collapsedWidget`` otherwise.

        """
        return self.__expanded

    expanded_ = Property(bool, fset=setExpanded, fget=expanded)

    def setWidget(self, w):
        raise NotImplementedError(
                "Please use the 'setExpandedWidget'/'setCollapsedWidget' "
                "methods to set the contents of the dock widget."
              )

    def setExpandedWidget(self, widget):
        """
        Set the widget with contents to show while expanded.
        """
        if widget is self.__expandedWidget:
            return

        if self.__expandedWidget is not None:
            self.__stack.removeWidget(self.__expandedWidget)

        self.__stack.insertWidget(0, widget)
        self.__expandedWidget = widget

        if self.__expanded:
            self.__stack.setCurrentWidget(widget)
            self.updateGeometry()

    def expandedWidget(self):
        """
        Return the widget previously set with ``setExpandedWidget``,
        or ``None`` if no widget has been set.

        """
        return self.__expandedWidget

    def setCollapsedWidget(self, widget):
        """
        Set the widget with contents to show while collapsed.
        """
        if widget is self.__collapsedWidget:
            return

        if self.__collapsedWidget is not None:
            self.__stack.removeWidget(self.__collapsedWidget)

        self.__stack.insertWidget(1, widget)
        self.__collapsedWidget = widget

        if not self.__expanded:
            self.__stack.setCurrentWidget(widget)
            self.updateGeometry()

    def collapsedWidget(self):
        """
        Return the widget previously set with ``setCollapsedWidget``,
        or ``None`` if no widget has been set.

        """
        return self.__collapsedWidget

    def setAnimationEnabled(self, animationEnabled):
        """
        Enable/disable the transition animation.
        """
        self.__stack.setAnimationEnabled(animationEnabled)

    def animationEnabled(self):
        """
        Is transition animation enabled.
        """
        return self.__stack.animationEnabled()

    def currentWidget(self):
        """
        Return the current shown widget depending on the `expanded` state.
        """
        if self.__expanded:
            return self.__expandedWidget
        else:
            return self.__collapsedWidget

    def expand(self):
        """
        Expand the dock (same as ``setExpanded(True)``)
        """
        self.setExpanded(True)

    def collapse(self):
        """
        Collapse the dock (same as ``setExpanded(False)``)
        """
        self.setExpanded(False)

    def eventFilter(self, obj, event):
        if obj is self.__closeButton:
            etype = event.type()
            if etype == QEvent.MouseButtonPress:
                self.setExpanded(not self.__expanded)
                return True
            elif etype == QEvent.MouseButtonDblClick or \
                    etype == QEvent.MouseButtonRelease:
                return True
            # TODO: which other events can trigger the button (is the button
            # focusable).

        return QDockWidget.eventFilter(self, obj, event)

    def event(self, event):
        if event.type() == QEvent.LayoutRequest:
            self.__fixMinimumWidth()

        return QDockWidget.event(self, event)

    def __onFeaturesChanged(self, features):
        pass

    def __onDockLocationChanged(self, area):
        if area == Qt.LeftDockWidgetArea:
            self.setLayoutDirection(Qt.LeftToRight)
        else:
            self.setLayoutDirection(Qt.RightToLeft)

        self.__stack.setLayoutDirection(self.parentWidget().layoutDirection())
        self.__fixIcon()

    def __onTransitionStarted(self):
        log.debug("Dock transition started.")

    def __onTransitionFinished(self):
        log.debug("Dock transition finished (new width %i)",
                  self.size().width())

    def __fixMinimumWidth(self):
        # A workaround for forcing the QDockWidget layout to disregard the
        # default minimumSize which can be to wide for us (overriding the
        # minimumSizeHint or setting the minimum size directly does not
        # seem to have an effect (Qt 4.8.3).
        size = self.__stack.sizeHint()
        if size.isValid() and not size.isEmpty():
            left, _, right, _ = self.getContentsMargins()
            width = size.width() + left + right

            if width < self.minimumSizeHint().width():
                if not self.__hasFixedWidth():
                    log.debug("Overriding default minimum size "
                              "(setFixedWidth(%i))", width)
                    self.__trueMinimumWidth = self.minimumSizeHint().width()
                self.setFixedWidth(width)
            else:
                if self.__hasFixedWidth():
                    if width >= self.__trueMinimumWidth:
                        # Unset the fixed size.
                        log.debug("Restoring default minimum size "
                                  "(setFixedWidth(%i))", QWIDGETSIZE_MAX)
                        self.__trueMinimumWidth = -1
                        self.setFixedWidth(QWIDGETSIZE_MAX)
                        self.updateGeometry()
                    else:
                        self.setFixedWidth(width)

    def __hasFixedWidth(self):
        return self.__trueMinimumWidth >= 0

    def __fixIcon(self):
        """Fix the dock close icon.
        """
        direction = self.layoutDirection()
        if direction == Qt.LeftToRight:
            if self.__expanded:
                icon = self.__iconLeft
            else:
                icon = self.__iconRight
        else:
            if self.__expanded:
                icon = self.__iconRight
            else:
                icon = self.__iconLeft

        self.__closeButton.setIcon(icon)
