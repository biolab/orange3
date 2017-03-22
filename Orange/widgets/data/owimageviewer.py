"""
Image Viewer Widget
-------------------

"""
import sys
import os
import weakref
import logging
import enum
import itertools
from xml.sax.saxutils import escape
from collections import namedtuple
from functools import partial
from itertools import zip_longest

import numpy

from AnyQt.QtWidgets import (
    QGraphicsScene, QGraphicsView, QGraphicsWidget, QGraphicsItem,
    QGraphicsTextItem, QGraphicsRectItem, QGraphicsLinearLayout,
    QGraphicsGridLayout, QSizePolicy, QApplication, QWidget, QLabel,
    QStyle, QShortcut
)
from AnyQt.QtGui import (
    QPixmap, QPen, QBrush, QColor, QPainter, QPainterPath, QImageReader
)
from AnyQt.QtCore import (
    Qt, QObject, QEvent, QThread, QSize, QPoint, QRect,
    QSizeF, QRectF, QPointF, QUrl, QDir, QMargins
)
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot
from AnyQt.QtNetwork import (
    QNetworkAccessManager, QNetworkDiskCache, QNetworkRequest, QNetworkReply
)

import Orange.data
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.overlay import proxydoc

from concurrent.futures import Future

_log = logging.getLogger(__name__)


class GraphicsPixmapWidget(QGraphicsWidget):
    """
    A QGraphicsWidget displaying a QPixmap
    """
    def __init__(self, pixmap=None, parent=None):
        super().__init__(parent)
        self.setCacheMode(QGraphicsItem.ItemCoordinateCache)
        self._pixmap = QPixmap(pixmap) if pixmap is not None else QPixmap()
        self._keepAspect = True
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def setPixmap(self, pixmap):
        self._pixmap = QPixmap(pixmap)
        self.updateGeometry()
        self.update()

    def pixmap(self):
        return QPixmap(self._pixmap)

    def setKeepAspectRatio(self, keep):
        if self._keepAspect != keep:
            self._keepAspect = bool(keep)
            self.update()

    def keepAspectRatio(self):
        return self._keepAspect

    def setGeometry(self, rect):
        self.prepareGeometryChange()
        super().setGeometry(rect)

    def sizeHint(self, which, constraint=QSizeF()):
        if which == Qt.PreferredSize:
            return QSizeF(self._pixmap.size())
        else:
            return QGraphicsWidget.sizeHint(self, which, constraint)

    def paint(self, painter, option, widget=0):
        if self._pixmap.isNull():
            return

        rect = self.contentsRect()
        pixsize = QSizeF(self._pixmap.size())
        aspectmode = (Qt.KeepAspectRatio if self._keepAspect
                      else Qt.IgnoreAspectRatio)
        pixsize.scale(rect.size(), aspectmode)
        pixrect = QRectF(QPointF(0, 0), pixsize)
        pixrect.moveCenter(rect.center())

        painter.save()
        painter.setPen(QPen(QColor(0, 0, 0, 50), 3))
        painter.drawRoundedRect(pixrect, 2, 2)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        source = QRectF(QPointF(0, 0), QSizeF(self._pixmap.size()))
        painter.drawPixmap(pixrect, self._pixmap, source)
        painter.restore()


class GraphicsTextWidget(QGraphicsWidget):
    def __init__(self, text, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.labelItem = QGraphicsTextItem(self)
        self.setHtml(text)

        self.labelItem.document().documentLayout().documentSizeChanged.connect(
            self.onLayoutChanged
        )

    def setGeometry(self, rect):
        self.prepareGeometryChange()
        super().setGeometry(rect)

    def onLayoutChanged(self, *args):
        self.updateGeometry()

    def sizeHint(self, which, constraint=QSizeF()):
        if which == Qt.MinimumSize:
            return self.labelItem.boundingRect().size()
        else:
            return self.labelItem.boundingRect().size()

    def setTextWidth(self, width):
        self.labelItem.setTextWidth(width)

    def setHtml(self, text):
        self.labelItem.setHtml(text)


class GraphicsThumbnailWidget(QGraphicsWidget):
    def __init__(self, pixmap, title="", parentItem=None, **kwargs):
        super().__init__(parentItem, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)
        self._title = None
        self._size = QSizeF()

        layout = QGraphicsLinearLayout(Qt.Vertical, self)
        layout.setSpacing(2)
        layout.setContentsMargins(5, 5, 5, 5)
        self.setContentsMargins(0, 0, 0, 0)

        self.pixmapWidget = GraphicsPixmapWidget(pixmap, self)
        self.labelWidget = GraphicsTextWidget(title, self)

        layout.addItem(self.pixmapWidget)
        layout.addItem(self.labelWidget)
        layout.addStretch()
        layout.setAlignment(self.pixmapWidget, Qt.AlignCenter)
        layout.setAlignment(self.labelWidget, Qt.AlignHCenter | Qt.AlignBottom)

        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.setFlag(QGraphicsItem.ItemIsSelectable, True)

        self.setTitle(title)
        self.setTitleWidth(100)

    def setGeometry(self, rect):
        self.prepareGeometryChange()
        super().setGeometry(rect)

    def setPixmap(self, pixmap):
        self.pixmapWidget.setPixmap(pixmap)
        self._updatePixmapSize()

    def pixmap(self):
        return self.pixmapWidget.pixmap()

    def setTitle(self, title):
        if self._title != title:
            self._title = title
            self.labelWidget.setHtml(
                '<center>' + escape(title) + '</center>'
            )
            self.layout().invalidate()

    def title(self):
        return self._title

    def setThumbnailSize(self, size):
        if self._size != size:
            self._size = QSizeF(size)
            self._updatePixmapSize()
            self.labelWidget.setTextWidth(max(100, size.width()))

    def setTitleWidth(self, width):
        self.labelWidget.setTextWidth(width)
        self.layout().invalidate()

    def paint(self, painter, option, widget=0):
        contents = self.contentsRect()

        if option.state & (QStyle.State_Selected | QStyle.State_HasFocus):
            painter.save()
            if option.state & QStyle.State_HasFocus:
                painter.setPen(QPen(QColor(125, 0, 0, 192)))
            else:
                painter.setPen(QPen(QColor(125, 162, 206, 192)))
            if option.state & QStyle.State_Selected:
                painter.setBrush(QBrush(QColor(217, 232, 252, 192)))
            painter.drawRoundedRect(
                QRectF(contents.topLeft(), self.geometry().size()), 3, 3)
            painter.restore()

    def _updatePixmapSize(self):
        pixsize = QSizeF(self._size)
        self.pixmapWidget.setMinimumSize(pixsize)
        self.pixmapWidget.setMaximumSize(pixsize)


class GraphicsThumbnailGrid(QGraphicsWidget):

    class LayoutMode(enum.Enum):
        FixedColumnCount, AutoReflow = 0, 1
    FixedColumnCount, AutoReflow = LayoutMode

    #: Signal emitted when the current (thumbnail) changes
    currentThumbnailChanged = Signal(object)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__layoutMode = GraphicsThumbnailGrid.AutoReflow
        self.__columnCount = -1
        self.__thumbnails = []  # type: List[GraphicsThumbnailWidget]
        #: The current 'focused' thumbnail item. This is the item that last
        #: received the keyboard focus (though it does not necessarily have
        #: it now)
        self.__current = None  # type: Optional[GraphicsThumbnailWidget]
        self.__reflowPending = False

        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.setContentsMargins(10, 10, 10, 10)
        # NOTE: Keeping a reference to the layout. self.layout()
        # returns a QGraphicsLayout wrapper (i.e. strips the
        # QGraphicsGridLayout-nes of the object).
        self.__layout = QGraphicsGridLayout()
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(10)
        self.setLayout(self.__layout)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if event.newSize().width() != event.oldSize().width() and \
                self.__layoutMode == GraphicsThumbnailGrid.AutoReflow:
            self.__reflow()

    def setGeometry(self, rect):
        self.prepareGeometryChange()
        super().setGeometry(rect)

    def count(self):
        """
        Returns
        -------
        count: int
            Number of thumbnails in the widget
        """
        return len(self.__thumbnails)

    def addThumbnail(self, thumbnail):
        """
        Add/append a thumbnail to the widget

        Parameters
        ----------
        thumbnail: Union[GraphicsThumbnailWidget, QPixmap]
            The thumbnail to insert
        """
        self.insertThumbnail(self.count(), thumbnail)

    def insertThumbnail(self, index, thumbnail):
        """
        Insert a new thumbnail into a widget.

        Raise a ValueError if thumbnail is already in the view.

        Parameters
        ----------
        index : int
            Index where to insert
        thumbnail : Union[GraphicsThumbnailWidget, QPixmap]
            The thumbnail to insert. GraphicsThumbnailGrid takes ownership
            of the item.
        """
        if isinstance(thumbnail, QPixmap):
            thumbnail = GraphicsThumbnailWidget(thumbnail, parentItem=self)
        elif thumbnail in self.__thumbnails:
            raise ValueError("{!r} is already inserted".format(thumbnail))
        elif not isinstance(thumbnail, GraphicsThumbnailWidget):
            raise TypeError

        index = max(min(index, self.count()), 0)

        moved = self.__takeItemsFrom(index)
        assert moved == self.__thumbnails[index:]
        self.__thumbnails.insert(index, thumbnail)
        self.__appendItems([thumbnail] + moved)
        thumbnail.setParentItem(self)
        thumbnail.installEventFilter(self)
        assert self.count() == self.layout().count()

        self.__scheduleLayout()

    def removeThumbnail(self, thumbnail):
        """
        Remove a single thumbnail from the grid.

        Raise a ValueError if thumbnail is not in the grid.

        Parameters
        ----------
        thumbnail : GraphicsThumbnailWidget
            Thumbnail to remove. Items ownership is transferred to the caller.
        """
        index = self.__thumbnails.index(thumbnail)
        moved = self.__takeItemsFrom(index)

        del self.__thumbnails[index]
        assert moved[0] is thumbnail and self.__thumbnails[index:] == moved[1:]
        self.__appendItems(moved[1:])

        thumbnail.removeEventFilter(self)
        if thumbnail.parentItem() is self:
            thumbnail.setParentItem(None)

        if self.__current is thumbnail:
            self.__current = None
            self.currentThumbnailChanged.emit(None)

        assert self.count() == self.layout().count()

    def thumbnailAt(self, index):
        """
        Return the thumbnail widget at `index`

        Parameters
        ----------
        index : int

        Returns
        -------
        thumbnail : GraphicsThumbnailWidget

        """
        return self.__thumbnails[index]

    def clear(self):
        """
        Remove all thumbnails from the grid.
        """
        removed = self.__takeItemsFrom(0)
        assert removed == self.__thumbnails
        self.__thumbnails = []
        for thumb in removed:
            thumb.removeEventFilter(self)
            if thumb.parentItem() is self:
                thumb.setParentItem(None)
        if self.__current is not None:
            self.__current = None
            self.currentThumbnailChanged.emit(None)

    def __takeItemsFrom(self, fromindex):
        # remove all items starting at fromindex from the layout and
        # return them
        # NOTE: Operate on layout only
        layout = self.__layout
        taken = []
        for i in reversed(range(fromindex, layout.count())):
            item = layout.itemAt(i)
            layout.removeAt(i)
            taken.append(item)
        return list(reversed(taken))

    def __appendItems(self, items):
        # Append/insert items into the layout at the end
        # NOTE: Operate on layout only
        layout = self.__layout
        columns = max(layout.columnCount(), 1)
        for i, item in enumerate(items, layout.count()):
            layout.addItem(item, i // columns, i % columns)

    def __scheduleLayout(self):
        if not self.__reflowPending:
            self.__reflowPending = True
            QApplication.postEvent(self, QEvent(QEvent.LayoutRequest),
                                   Qt.HighEventPriority)

    def event(self, event):
        if event.type() == QEvent.LayoutRequest:
            if self.__layoutMode == GraphicsThumbnailGrid.AutoReflow:
                self.__reflow()
            else:
                self.__gridlayout()

            if self.parentLayoutItem() is None:
                sh = self.effectiveSizeHint(Qt.PreferredSize)
                self.resize(sh)

            if self.layout():
                self.layout().activate()

        return super().event(event)

    def setFixedColumnCount(self, count):
        if count < 0:
            if self.__layoutMode != GraphicsThumbnailGrid.AutoReflow:
                self.__layoutMode = GraphicsThumbnailGrid.AutoReflow
                self.__reflow()
        else:
            if self.__layoutMode != GraphicsThumbnailGrid.FixedColumnCount:
                self.__layoutMode = GraphicsThumbnailGrid.FixedColumnCount

            if self.__columnCount != count:
                self.__columnCount = count
                self.__gridlayout()

    def __reflow(self):
        self.__reflowPending = False
        layout = self.__layout
        width = self.contentsRect().width()
        hints = [item.effectiveSizeHint(Qt.PreferredSize)
                 for item in self.__thumbnails]

        widths = [max(24, h.width()) for h in hints]
        ncol = self._fitncols(widths, layout.horizontalSpacing(), width)

        self.__relayoutGrid(ncol)

    def __gridlayout(self):
        assert self.__layoutMode == GraphicsThumbnailGrid.FixedColumnCount
        self.__relayoutGrid(self.__columnCount)

    def __relayoutGrid(self, columnCount):
        layout = self.__layout
        if columnCount == layout.columnCount():
            return

        # remove all items from the layout, then re-add them back in
        # updated positions
        items = self.__takeItemsFrom(0)
        for i, item in enumerate(items):
            layout.addItem(item, i // columnCount, i % columnCount)

    def items(self):
        """
        Return all thumbnail items.

        Returns
        -------
        thumbnails : List[GraphicsThumbnailWidget]
        """
        return list(self.__thumbnails)

    def currentItem(self):
        """
        Return the current (last focused) thumbnail item.
        """
        return self.__current

    def _fitncols(self, widths, spacing, constraint):
        def sliced(seq, ncol):
            return [seq[i:i + ncol] for i in range(0, len(seq), ncol)]

        def flow_width(widths, spacing, ncol):
            W = sliced(widths, ncol)
            col_widths = map(max, zip_longest(*W, fillvalue=0))
            return sum(col_widths) + (ncol - 1) * spacing

        ncol_best = 1
        for ncol in range(2, len(widths) + 1):
            w = flow_width(widths, spacing, ncol)
            if w <= constraint:
                ncol_best = ncol
            else:
                break

        return ncol_best

    def keyPressEvent(self, event):
        if event.key() in [Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down]:
            self._moveCurrent(event.key(), event.modifiers())
            event.accept()
            return
        super().keyPressEvent(event)

    def eventFilter(self, receiver, event):
        if isinstance(receiver, GraphicsThumbnailWidget) and \
                event.type() == QEvent.FocusIn and \
                receiver in self.__thumbnails:
            self.__current = receiver
            self.currentThumbnailChanged.emit(receiver)

        return super().eventFilter(receiver, event)

    def _moveCurrent(self, key, modifiers=Qt.NoModifier):
        """
        Move the current thumbnail focus (`currentItem`) based on a key press
        (Qt.Key{Up,Down,Left,Right})

        Parameters
        ----------
        key : Qt.Key
        modifiers : Qt.Modifiers
        """
        current = self.__current
        layout = self.__layout
        columns = layout.columnCount()
        rows = layout.rowCount()
        itempos = {}
        for i, j in itertools.product(range(rows), range(columns)):
            if i * columns + j >= layout.count():
                break
            item = layout.itemAt(i, j)
            if item is not None:
                itempos[item] = (i, j)
        pos = itempos.get(current, None)

        if pos is None:
            return False

        i, j = pos
        index = i * columns + j
        if key == Qt.Key_Left:
            index = index - 1
        elif key == Qt.Key_Right:
            index = index + 1
        elif key == Qt.Key_Down:
            index = index + columns
        elif key == Qt.Key_Up:
            index = index - columns

        index = min(max(index, 0), layout.count() - 1)
        i = index // columns
        j = index % columns
        newcurrent = layout.itemAt(i, j)
        assert newcurrent is self.__thumbnails[index]

        if newcurrent is not None:
            if not modifiers & (Qt.ShiftModifier | Qt.ControlModifier):
                for item in self.__thumbnails:
                    if item is not newcurrent:
                        item.setSelected(False)
                # self.scene().clearSelection()

            newcurrent.setSelected(True)
            newcurrent.setFocus(Qt.TabFocusReason)
            newcurrent.ensureVisible()

        if self.__current is not newcurrent:
            self.__current = newcurrent
            self.currentThumbnailChanged.emit(newcurrent)


class GraphicsScene(QGraphicsScene):
    selectionRectPointChanged = Signal(QPointF)

    def __init__(self, *args):
        QGraphicsScene.__init__(self, *args)
        self.selectionRect = None

    def mousePressEvent(self, event):
        QGraphicsScene.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            screenPos = event.screenPos()
            buttonDown = event.buttonDownScreenPos(Qt.LeftButton)
            if (screenPos - buttonDown).manhattanLength() > 2.0:
                self.updateSelectionRect(event)
        QGraphicsScene.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.selectionRect:
                self.removeItem(self.selectionRect)
                self.selectionRect = None
        QGraphicsScene.mouseReleaseEvent(self, event)

    def updateSelectionRect(self, event):
        pos = event.scenePos()
        buttonDownPos = event.buttonDownScenePos(Qt.LeftButton)
        rect = QRectF(pos, buttonDownPos).normalized()
        rect = rect.intersected(self.sceneRect())
        if not self.selectionRect:
            self.selectionRect = QGraphicsRectItem()
            self.selectionRect.setBrush(QColor(10, 10, 10, 20))
            self.selectionRect.setPen(QPen(QColor(200, 200, 200, 200)))
            self.addItem(self.selectionRect)
        self.selectionRect.setRect(rect)
        if event.modifiers() & Qt.ControlModifier or \
                        event.modifiers() & Qt.ShiftModifier:
            path = self.selectionArea()
        else:
            path = QPainterPath()
        path.addRect(rect)
        self.setSelectionArea(path)
        self.selectionRectPointChanged.emit(pos)


class ThumbnailView(QGraphicsView):
    """
    A widget displaying a image thumbnail grid in a scroll area
    """
    FixedColumnCount, AutoReflow = GraphicsThumbnailGrid.LayoutMode

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.TextAntialiasing)

        self.__layoutMode = ThumbnailView.AutoReflow
        self.__columnCount = -1

        self.__grid = GraphicsThumbnailGrid()
        self.__grid.currentThumbnailChanged.connect(
            self.__onCurrentThumbnailChanged
        )
        self.__previewWidget = None
        scene = GraphicsScene(self)
        scene.addItem(self.__grid)
        scene.selectionRectPointChanged.connect(
            self.__ensureVisible, Qt.QueuedConnection
        )
        self.setScene(scene)

        sh = QShortcut(Qt.Key_Space, self,
                       context=Qt.WidgetWithChildrenShortcut)
        sh.activated.connect(self.__previewToogle)

        self.__grid.geometryChanged.connect(self.__updateSceneRect)

    @proxydoc(GraphicsThumbnailGrid.addThumbnail)
    def addThumbnail(self, thumbnail):
        self.__grid.addThumbnail(thumbnail)

    @proxydoc(GraphicsThumbnailGrid.insertThumbnail)
    def insertThumbnail(self, index, thumbnail):
        self.__grid.insertThumbnail(index, thumbnail)

    @proxydoc(GraphicsThumbnailGrid.setFixedColumnCount)
    def setFixedColumnCount(self, count):
        self.__grid.setFixedColumnCount(count)

    @proxydoc(GraphicsThumbnailGrid.count)
    def count(self):
        return self.__grid.count()

    def clear(self):
        """
        Clear all thumbnails and close/delete the preview window if used.
        """
        self.__grid.clear()

        if self.__previewWidget is not None:
            self.__closePreview()

    def sizeHint(self):
        return QSize(480, 640)

    def __updateSceneRect(self):
        self.scene().setSceneRect(self.scene().itemsBoundingRect())
        # Full viewport update, otherwise contents outside the new
        # sceneRect can persist on the viewport
        self.viewport().update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            width = event.size().width() - 2

            self.__grid.setMaximumWidth(width)
            self.__grid.setMinimumWidth(width)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.__previewWidget is not None:
            self.__closePreview()
            event.accept()
            return
        return super().keyPressEvent(event)

    def __previewToogle(self):
        if self.__previewWidget is None and self.__grid.currentItem() is not None:
            focusitem = self.__grid.currentItem()
            preview = self.__getPreviewWidget()
            preview.show()
            preview.raise_()
            preview.setPixmap(focusitem.pixmap())
        else:
            self.__closePreview()

    def __getPreviewWidget(self):
        # return the preview image view widget
        if self.__previewWidget is None:
            self.__previewWidget = Preview(self)
            self.__previewWidget.setWindowFlags(
                Qt.WindowStaysOnTopHint | Qt.Tool)
            self.__previewWidget.setAttribute(
                Qt.WA_ShowWithoutActivating)
            self.__previewWidget.setFocusPolicy(Qt.NoFocus)
            self.__previewWidget.installEventFilter(self)

        return self.__previewWidget

    def __updatePreviewPixmap(self):
        current = self.__grid.currentItem()
        if isinstance(current, GraphicsThumbnailWidget) and \
                current.parentItem() is self.__grid and \
                self.__previewWidget is not None:
            self.__previewWidget.setPixmap(current.pixmap())

    def __closePreview(self):
        if self.__previewWidget is not None:
            self.__previewWidget.close()
            self.__previewWidget.setPixmap(QPixmap())
            self.__previewWidget.deleteLater()
            self.__previewWidget = None

    def eventFilter(self, receiver, event):
        if receiver is self.__previewWidget and \
                event.type() == QEvent.KeyPress:
            if event.key() in [Qt.Key_Left, Qt.Key_Right,
                               Qt.Key_Down, Qt.Key_Up]:
                self.__grid._moveCurrent(event.key())
                event.accept()
                return True
            elif event.key() in [Qt.Key_Escape, Qt.Key_Space]:
                self.__closePreview()
                event.accept()
                return True
        return super().eventFilter(receiver, event)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.__closePreview()

    def __onCurrentThumbnailChanged(self, thumbnail):
        if thumbnail is not None:
            self.__updatePreviewPixmap()
        else:
            self.__closePreview()

    @Slot(QPointF)
    def __ensureVisible(self, point):
        self.ensureVisible(QRectF(point, QSizeF(1, 1)), 5, 5),


class Preview(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__pixmap = QPixmap()
        # Flag indicating if the widget was resized as a result of user
        # initiated window resize. When false the widget will automatically
        # resize/re-position based on pixmap size.
        self.__hasExplicitSize = False
        self.__inUpdateWindowGeometry = False

    def setPixmap(self, pixmap):
        if self.__pixmap != pixmap:
            self.__pixmap = QPixmap(pixmap)
            self.__updateWindowGeometry()
            self.update()
            self.updateGeometry()

    def pixmap(self):
        return QPixmap(self.__pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.isVisible() and self.isWindow() and \
                not self.__inUpdateWindowGeometry:
            # mark that we have an explicit user provided size
            self.__hasExplicitSize = True

    def __updateWindowGeometry(self):
        if not self.isWindow() or self.__hasExplicitSize:
            return

        def framemargins(widget):
            frame, geom = widget.frameGeometry(), widget.geometry()
            return QMargins(geom.left() - frame.left(),
                            geom.top() - frame.top(),
                            geom.right() - frame.right(),
                            geom.bottom() - frame.bottom())

        def fitRect(rect, targetrect):
            size = rect.size().boundedTo(targetgeom.size())
            newrect = QRect(rect.topLeft(), size)
            dx, dy = 0, 0
            if newrect.left() < targetrect.left():
                dx = targetrect.left() - newrect.left()
            if newrect.top() < targetrect.top():
                dy = targetrect.top() - newrect.top()
            if newrect.right() > targetrect.right():
                dx = targetrect.right() - newrect.right()
            if newrect.bottom() > targetrect.bottom():
                dy = targetrect.bottom() - newrect.bottom()
            return newrect.translated(dx, dy)

        margins = framemargins(self)
        minsize = QSize(120, 120)
        pixsize = self.__pixmap.size()
        available = QApplication.desktop().availableGeometry(self)
        available = available.adjusted(margins.left(), margins.top(),
                                       -margins.right(), -margins.bottom())
        # extra adjustment so the preview does not cover the whole desktop
        available = available.adjusted(10, 10, -10, -10)
        targetsize = pixsize.boundedTo(available.size()).expandedTo(minsize)
        pixsize.scale(targetsize, Qt.KeepAspectRatio)

        if not self.testAttribute(Qt.WA_WState_Created) or \
                self.testAttribute(Qt.WA_WState_Hidden):
            center = available.center()
        else:
            center = self.geometry().center()
        targetgeom = QRect(QPoint(0, 0), pixsize)
        targetgeom.moveCenter(center)
        if not available.contains(targetgeom):
            targetgeom = fitRect(targetgeom, available)
        self.__inUpdateWindowGeometry = True
        self.setGeometry(targetgeom)
        self.__inUpdateWindowGeometry = False

    def sizeHint(self):
        return self.__pixmap.size()

    def paintEvent(self, event):
        if self.__pixmap.isNull():
            return

        sourcerect = QRect(QPoint(0, 0), self.__pixmap.size())
        pixsize = QSizeF(self.__pixmap.size())
        rect = self.contentsRect()
        pixsize.scale(QSizeF(rect.size()), Qt.KeepAspectRatio)
        targetrect = QRectF(QPointF(0, 0), pixsize)
        targetrect.moveCenter(QPointF(rect.center()))
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(targetrect, self.__pixmap, QRectF(sourcerect))
        painter.end()


_ImageItem = namedtuple(
    "_ImageItem",
    ["index",   # Row index in the input data table
     "widget",  # GraphicsThumbnailWidget displaying the image.
     "url",     # Composed final image url.
     "future"]  # Future instance yielding an QImage
)


class OWImageViewer(widget.OWWidget):
    name = "Image Viewer"
    description = "View images referred to in the data."
    icon = "icons/ImageViewer.svg"
    priority = 4050

    inputs = [("Data", Orange.data.Table, "setData")]
    outputs = [("Data", Orange.data.Table, )]

    settingsHandler = settings.DomainContextHandler()

    imageAttr = settings.ContextSetting(0)
    titleAttr = settings.ContextSetting(0)

    imageSize = settings.Setting(100)
    autoCommit = settings.Setting(True)

    buttons_area_orientation = Qt.Vertical
    graph_name = "scene"

    UserAdviceMessages = [
        widget.Message(
            "Pressing the 'Space' key while the thumbnail view has focus and "
            "a selected item will open a window with a full image",
            persistent_id="preview-introduction")
    ]

    def __init__(self):
        super().__init__()
        self.data = None
        self.allAttrs = []
        self.stringAttrs = []

        self.selectedIndices = []

        #: List of _ImageItems
        self.items = []

        self._errcount = 0
        self._successcount = 0

        self.info = gui.widgetLabel(
            gui.vBox(self.controlArea, "Info"),
            "Waiting for input.\n"
        )

        self.imageAttrCB = gui.comboBox(
            self.controlArea, self, "imageAttr",
            box="Image Filename Attribute",
            tooltip="Attribute with image filenames",
            callback=[self.clearScene, self.setupScene],
            contentsLength=12,
            addSpace=True,
        )

        self.titleAttrCB = gui.comboBox(
            self.controlArea, self, "titleAttr",
            box="Title Attribute",
            tooltip="Attribute with image title",
            callback=self.updateTitles,
            contentsLength=12,
            addSpace=True
        )

        gui.hSlider(
            self.controlArea, self, "imageSize",
            box="Image Size", minValue=32, maxValue=1024, step=16,
            callback=self.updateSize,
            createLabel=False
        )
        gui.rubber(self.controlArea)

        gui.auto_commit(self.buttonsArea, self, "autoCommit", "Send", box=False)

        self.thumbnailView = ThumbnailView(
            alignment=Qt.AlignTop | Qt.AlignLeft,  # scene alignment,
            focusPolicy=Qt.StrongFocus,
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn
        )
        self.mainArea.layout().addWidget(self.thumbnailView)
        self.scene = self.thumbnailView.scene()
        self.scene.selectionChanged.connect(self.onSelectionChanged)
        self.loader = ImageLoader(self)

    def sizeHint(self):
        return QSize(800, 600)

    def setData(self, data):
        self.closeContext()
        self.clear()

        self.data = data

        if data is not None:
            domain = data.domain
            self.allAttrs = (domain.class_vars + domain.metas +
                             domain.attributes)
            self.stringAttrs = [a for a in domain.metas if a.is_string]

            self.stringAttrs = sorted(
                self.stringAttrs,
                key=lambda attr: 0 if "type" in attr.attributes else 1
            )

            indices = [i for i, var in enumerate(self.stringAttrs)
                       if var.attributes.get("type") == "image"]
            if indices:
                self.imageAttr = indices[0]

            self.imageAttrCB.setModel(VariableListModel(self.stringAttrs))
            self.titleAttrCB.setModel(VariableListModel(self.allAttrs))

            self.openContext(data)

            self.imageAttr = max(min(self.imageAttr, len(self.stringAttrs) - 1), 0)
            self.titleAttr = max(min(self.titleAttr, len(self.allAttrs) - 1), 0)

            if self.stringAttrs:
                self.setupScene()
        else:
            self.info.setText("Waiting for input.\n")

    def clear(self):
        self.data = None
        self.error()
        self.imageAttrCB.clear()
        self.titleAttrCB.clear()
        self.clearScene()

    def setupScene(self):
        self.error()
        if self.data:
            attr = self.stringAttrs[self.imageAttr]
            titleAttr = self.allAttrs[self.titleAttr]
            assert self.thumbnailView.count() == 0
            size = QSizeF(self.imageSize, self.imageSize)

            for i, inst in enumerate(self.data):
                if not numpy.isfinite(inst[attr]):  # skip missing
                    continue
                url = self.urlFromValue(inst[attr])
                title = str(inst[titleAttr])

                thumbnail = GraphicsThumbnailWidget(QPixmap(), title=title)
                thumbnail.setThumbnailSize(size)
                thumbnail.setToolTip(url.toString())
                thumbnail.instance = inst
                self.thumbnailView.addThumbnail(thumbnail)

                if url.isValid() and url.isLocalFile():
                    reader = QImageReader(url.toLocalFile())
                    image = reader.read()
                    if image.isNull():
                        error = reader.errorString()
                        thumbnail.setToolTip(
                            thumbnail.toolTip() + "\n" + error)
                        self._errcount += 1
                    else:
                        pixmap = QPixmap.fromImage(image)
                        thumbnail.setPixmap(pixmap)
                        self._successcount += 1

                    future = Future()
                    future.set_result(image)
                    future._reply = None
                elif url.isValid():
                    future = self.loader.get(url)

                    @future.add_done_callback
                    def set_pixmap(future, thumb=thumbnail):
                        if future.cancelled():
                            return

                        assert future.done()

                        if future.exception():
                            # Should be some generic error image.
                            pixmap = QPixmap()
                            thumb.setToolTip(thumb.toolTip() + "\n" +
                                             str(future.exception()))
                        else:
                            pixmap = QPixmap.fromImage(future.result())

                        thumb.setPixmap(pixmap)

                        self._noteCompleted(future)
                else:
                    future = None

                self.items.append(_ImageItem(i, thumbnail, url, future))

            if any(it.future is not None and not it.future.done()
                   for it in self.items):
                self.info.setText("Retrieving...\n")
            else:
                self._updateStatus()

    def urlFromValue(self, value):
        variable = value.variable
        origin = variable.attributes.get("origin", "")
        if origin and QDir(origin).exists():
            origin = QUrl.fromLocalFile(origin)
        elif origin:
            origin = QUrl(origin)
            if not origin.scheme():
                origin.setScheme("file")
        else:
            origin = QUrl("")
        base = origin.path()
        if base.strip() and not base.endswith("/"):
            origin.setPath(base + "/")

        if os.path.exists(str(value)):
            url = QUrl.fromLocalFile(str(value))
        else:
            name = QUrl(str(value))
            url = origin.resolved(name)
        if not url.scheme():
            url.setScheme("file")
        return url

    def _cancelAllFutures(self):
        for item in self.items:
            if item.future is not None:
                item.future.cancel()
                if item.future._reply is not None:
                    item.future._reply.close()
                    item.future._reply.deleteLater()
                    item.future._reply = None

    def clearScene(self):
        self._cancelAllFutures()

        self.items = []
        self.thumbnailView.clear()
        self._errcount = 0
        self._successcount = 0

    def thumbnailItems(self):
        return [item.widget for item in self.items]

    def updateSize(self):
        size = QSizeF(self.imageSize, self.imageSize)
        for item in self.thumbnailItems():
            item.setThumbnailSize(size)

    def updateTitles(self):
        titleAttr = self.allAttrs[self.titleAttr]
        for item in self.items:
            item.widget.setTitle(str(item.widget.instance[titleAttr]))

    def onSelectionChanged(self):
        selected = [item for item in self.items if item.widget.isSelected()]
        self.selectedIndices = [item.index for item in selected]
        self.commit()

    def commit(self):
        if self.data:
            if self.selectedIndices:
                selected = self.data[self.selectedIndices]
            else:
                selected = None
            self.send("Data", selected)
        else:
            self.send("Data", None)

    def _noteCompleted(self, future):
        # Note the completed future's state
        if future.cancelled():
            return

        if future.exception():
            self._errcount += 1
            _log.debug("Error: %r", future.exception())
        else:
            self._successcount += 1

        self._updateStatus()

    def _updateStatus(self):
        count = len([item for item in self.items if item.future is not None])
        self.info.setText(
            "Retrieving:\n" +
            "{} of {} images".format(self._successcount, count))

        if self._errcount + self._successcount == count:
            if self._errcount:
                self.info.setText(
                    "Done:\n" +
                    "{} images, {} errors".format(count, self._errcount)
                )
            else:
                self.info.setText(
                    "Done:\n" +
                    "{} images".format(count)
                )
            attr = self.stringAttrs[self.imageAttr]
            if self._errcount == count and "type" not in attr.attributes:
                self.error("No images found! Make sure the '%s' attribute "
                           "is tagged with 'type=image'" % attr.name)

    def onDeleteWidget(self):
        self._cancelAllFutures()
        self.clear()


class ImageLoader(QObject):
    #: A weakref to a QNetworkAccessManager used for image retrieval.
    #: (we can only have one QNetworkDiskCache opened on the same
    #: directory)
    _NETMANAGER_REF = None

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        assert QThread.currentThread() is QApplication.instance().thread()

        netmanager = self._NETMANAGER_REF and self._NETMANAGER_REF()
        if netmanager is None:
            netmanager = QNetworkAccessManager()
            cache = QNetworkDiskCache()
            cache.setCacheDirectory(
                os.path.join(settings.widget_settings_dir(),
                             __name__ + ".ImageLoader.Cache")
            )
            netmanager.setCache(cache)
            ImageLoader._NETMANAGER_REF = weakref.ref(netmanager)
        self._netmanager = netmanager

    def get(self, url):
        future = Future()
        url = QUrl(url)
        request = QNetworkRequest(url)
        request.setRawHeader(b"User-Agent", b"OWImageViewer/1.0")
        request.setAttribute(
            QNetworkRequest.CacheLoadControlAttribute,
            QNetworkRequest.PreferCache
        )

        # Future yielding a QNetworkReply when finished.
        reply = self._netmanager.get(request)
        future._reply = reply

        @future.add_done_callback
        def abort_on_cancel(f):
            # abort the network request on future.cancel()
            if f.cancelled() and f._reply is not None:
                f._reply.abort()

        n_redir = 0

        def on_reply_ready(reply, future):
            nonlocal n_redir
            # schedule deferred delete to ensure the reply is closed
            # otherwise we will leak file/socket descriptors
            reply.deleteLater()
            future._reply = None
            if reply.error() == QNetworkReply.OperationCanceledError:
                # The network request was cancelled
                reply.close()
                future.cancel()
                return

            if reply.error() != QNetworkReply.NoError:
                # XXX Maybe convert the error into standard
                # http and urllib exceptions.
                future.set_exception(Exception(reply.errorString()))
                reply.close()
                return

            # Handle a possible redirection
            location = reply.attribute(
                QNetworkRequest.RedirectionTargetAttribute)

            if location is not None and n_redir < 1:
                n_redir += 1
                location = reply.url().resolved(location)
                # Retry the original request with a new url.
                request = QNetworkRequest(reply.request())
                request.setUrl(location)
                newreply = self._netmanager.get(request)
                future._reply = newreply
                newreply.finished.connect(
                    partial(on_reply_ready, newreply, future))
                reply.close()
                return

            reader = QImageReader(reply)
            image = reader.read()
            reply.close()

            if image.isNull():
                future.set_exception(Exception(reader.errorString()))
            else:
                future.set_result(image)

        reply.finished.connect(partial(on_reply_ready, reply, future))
        return future


def main(argv=sys.argv):
    import sip

    app = QApplication(argv)
    argv = app.arguments()
    w = OWImageViewer()
    w.show()
    w.raise_()

    if len(argv) > 1:
        data = Orange.data.Table(argv[1])
    else:
        data = Orange.data.Table('zoo-with-images')

    w.setData(data)
    rval = app.exec_()
    w.saveSettings()
    w.onDeleteWidget()
    sip.delete(w)
    app.processEvents()
    return rval


if __name__ == "__main__":
    main()
