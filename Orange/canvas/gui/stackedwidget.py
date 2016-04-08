"""
=====================
AnimatedStackedWidget
=====================

A widget similar to :class:`QStackedWidget` supporting animated
transitions between widgets.

"""

import logging

from PyQt4.QtGui import QWidget, QFrame, QStackedLayout, QPixmap, \
                        QPainter, QSizePolicy

from PyQt4.QtCore import Qt, QPoint, QRect, QSize, QPropertyAnimation

from PyQt4.QtCore import pyqtSignal as Signal
from PyQt4.QtCore import pyqtProperty as Property

from .utils import updates_disabled

log = logging.getLogger(__name__)


def clipMinMax(size, minSize, maxSize):
    """
    Clip the size so it is bigger then minSize but smaller than maxSize.
    """
    return size.expandedTo(minSize).boundedTo(maxSize)


def fixSizePolicy(size, hint, policy):
    """
    Fix size so it conforms to the size policy and the given size hint.
    """
    width, height = hint.width(), hint.height()
    expanding = policy.expandingDirections()
    hpolicy, vpolicy = policy.horizontalPolicy(), policy.verticalPolicy()

    if expanding & Qt.Horizontal:
        width = max(width, size.width())

    if hpolicy == QSizePolicy.Maximum:
        width = min(width, size.width())

    if expanding & Qt.Vertical:
        height = max(height, size.height())

    if vpolicy == QSizePolicy.Maximum:
        height = min(height, hint.height())

    return QSize(width, height).boundedTo(size)


class StackLayout(QStackedLayout):
    """
    A stacked layout with ``sizeHint`` always the same as that of the
    `current` widget.

    """
    def __init__(self, parent=None):
        QStackedLayout.__init__(self, parent)
        self.currentChanged.connect(self._onCurrentChanged)

    def sizeHint(self):
        current = self.currentWidget()
        if current:
            hint = current.sizeHint()
            # Clip the hint with min/max sizes.
            hint = clipMinMax(hint, current.minimumSize(),
                              current.maximumSize())
            return hint
        else:
            return QStackedLayout.sizeHint(self)

    def minimumSize(self):
        current = self.currentWidget()
        if current:
            return current.minimumSize()
        else:
            return QStackedLayout.minimumSize(self)

    def maximumSize(self):
        current = self.currentWidget()
        if current:
            return current.maximumSize()
        else:
            return QStackedLayout.maximumSize(self)

    def setGeometry(self, rect):
        QStackedLayout.setGeometry(self, rect)
        for i in range(self.count()):
            w = self.widget(i)
            hint = w.sizeHint()
            geom = QRect(rect)
            size = clipMinMax(rect.size(), w.minimumSize(), w.maximumSize())
            size = fixSizePolicy(size, hint, w.sizePolicy())
            geom.setSize(size)
            if geom != w.geometry():
                w.setGeometry(geom)

    def _onCurrentChanged(self, index):
        """
        Current widget changed, invalidate the layout.
        """
        self.invalidate()


class AnimatedStackedWidget(QFrame):
    # Current widget has changed
    currentChanged = Signal(int)

    # Transition animation has started
    transitionStarted = Signal()

    # Transition animation has finished
    transitionFinished = Signal()

    def __init__(self, parent=None, animationEnabled=True):
        QFrame.__init__(self, parent)
        self.__animationEnabled = animationEnabled

        layout = StackLayout()

        self.__fadeWidget = CrossFadePixmapWidget(self)

        self.transitionAnimation = \
            QPropertyAnimation(self.__fadeWidget, b"blendingFactor_", self)
        self.transitionAnimation.setStartValue(0.0)
        self.transitionAnimation.setEndValue(1.0)
        self.transitionAnimation.setDuration(100 if animationEnabled else 0)
        self.transitionAnimation.finished.connect(
            self.__onTransitionFinished
        )

        layout.addWidget(self.__fadeWidget)
        layout.currentChanged.connect(self.__onLayoutCurrentChanged)

        self.setLayout(layout)

        self.__widgets = []
        self.__currentIndex = -1
        self.__nextCurrentIndex = -1

    def setAnimationEnabled(self, animationEnabled):
        """
        Enable/disable transition animations.
        """
        if self.__animationEnabled != animationEnabled:
            self.__animationEnabled = animationEnabled
            self.transitionAnimation.setDuration(
                100 if animationEnabled else 0
            )

    def animationEnabled(self):
        """
        Is the transition animation enabled.
        """
        return self.__animationEnabled

    def addWidget(self, widget):
        """
        Append the widget to the stack and return its index.
        """
        return self.insertWidget(self.layout().count(), widget)

    def insertWidget(self, index, widget):
        """
        Insert `widget` into the stack at `index`.
        """
        index = min(index, self.count())
        self.__widgets.insert(index, widget)
        if index <= self.__currentIndex or self.__currentIndex == -1:
            self.__currentIndex += 1
        return self.layout().insertWidget(index, widget)

    def removeWidget(self, widget):
        """
        Remove `widget` from the stack.

        .. note:: The widget is hidden but is not deleted.

        """
        index = self.__widgets.index(widget)
        self.layout().removeWidget(widget)
        self.__widgets.pop(index)

    def widget(self, index):
        """
        Return the widget at `index`
        """
        return self.__widgets[index]

    def indexOf(self, widget):
        """
        Return the index of `widget` in the stack.
        """
        return self.__widgets.index(widget)

    def count(self):
        """
        Return the number of widgets in the stack.
        """
        return max(self.layout().count() - 1, 0)

    def setCurrentWidget(self, widget):
        """
        Set the current shown widget.
        """
        index = self.__widgets.index(widget)
        self.setCurrentIndex(index)

    def setCurrentIndex(self, index):
        """
        Set the current shown widget index.
        """
        index = max(min(index, self.count() - 1), 0)
        if self.__currentIndex == -1:
            self.layout().setCurrentIndex(index)
            self.__currentIndex = index
            return

#        if not self.animationEnabled():
#            self.layout().setCurrentIndex(index)
#            self.__currentIndex = index
#            return

        # else start the animation
        current = self.__widgets[self.__currentIndex]
        next_widget = self.__widgets[index]

        current_pix = QPixmap.grabWidget(current)
        next_pix = QPixmap.grabWidget(next_widget)

        with updates_disabled(self):
            self.__fadeWidget.setPixmap(current_pix)
            self.__fadeWidget.setPixmap2(next_pix)
            self.__nextCurrentIndex = index
            self.__transitionStart()

    def currentIndex(self):
        """
        Return the current shown widget index.
        """
        return self.__currentIndex

    def sizeHint(self):
        hint = QFrame.sizeHint(self)
        if hint.isEmpty():
            hint = QSize(0, 0)
        return hint

    def __transitionStart(self):
        """
        Start the transition.
        """
        log.debug("Stack transition start (%s)", str(self.objectName()))
        # Set the fade widget as the current widget
        self.__fadeWidget.blendingFactor_ = 0.0
        self.layout().setCurrentWidget(self.__fadeWidget)
        self.transitionAnimation.start()
        self.transitionStarted.emit()

    def __onTransitionFinished(self):
        """
        Transition has finished.
        """
        log.debug("Stack transition finished (%s)" % str(self.objectName()))
        self.__fadeWidget.blendingFactor_ = 1.0
        self.__currentIndex = self.__nextCurrentIndex
        with updates_disabled(self):
            self.layout().setCurrentIndex(self.__currentIndex)
        self.transitionFinished.emit()

    def __onLayoutCurrentChanged(self, index):
        # Suppress transitional __fadeWidget current widget
        if index != self.count():
            self.currentChanged.emit(index)


class CrossFadePixmapWidget(QWidget):
    """
    A widget for cross fading between two pixmaps.
    """
    def __init__(self, parent=None, pixmap1=None, pixmap2=None):
        QWidget.__init__(self, parent)
        self.setPixmap(pixmap1)
        self.setPixmap2(pixmap2)
        self.blendingFactor_ = 0.0
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def setPixmap(self, pixmap):
        """
        Set pixmap 1
        """
        self.pixmap1 = pixmap
        self.updateGeometry()

    def setPixmap2(self, pixmap):
        """
        Set pixmap 2
        """
        self.pixmap2 = pixmap
        self.updateGeometry()

    def setBlendingFactor(self, factor):
        """
        Set the blending factor between the two pixmaps.
        """
        self.__blendingFactor = factor
        self.updateGeometry()

    def blendingFactor(self):
        """
        Pixmap blending factor between 0.0 and 1.0
        """
        return self.__blendingFactor

    blendingFactor_ = Property(float, fget=blendingFactor,
                               fset=setBlendingFactor)

    def sizeHint(self):
        """
        Return an interpolated size between pixmap1.size()
        and pixmap2.size()

        """
        if self.pixmap1 and self.pixmap2:
            size1 = self.pixmap1.size()
            size2 = self.pixmap2.size()
            return size1 + self.blendingFactor_ * (size2 - size1)
        else:
            return QWidget.sizeHint(self)

    def paintEvent(self, event):
        """
        Paint the interpolated pixmap image.
        """
        p = QPainter(self)
        p.setClipRect(event.rect())
        factor = self.blendingFactor_ ** 2
        if self.pixmap1 and 1. - factor:
            p.setOpacity(1. - factor)
            p.drawPixmap(QPoint(0, 0), self.pixmap1)
        if self.pixmap2 and factor:
            p.setOpacity(factor)
            p.drawPixmap(QPoint(0, 0), self.pixmap2)
