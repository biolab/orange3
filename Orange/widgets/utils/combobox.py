
from typing import Optional

from AnyQt.QtCore import (
    Qt, QEvent, QObject, QAbstractItemModel, QModelIndex,
    QSortFilterProxyModel, QSize, QRect, QMargins, QCoreApplication
)

from AnyQt.QtGui import QMouseEvent, QKeyEvent
from AnyQt.QtWidgets import (
    QWidget, QComboBox, QLineEdit, QAbstractItemView, QListView,
    QStyleOptionComboBox, QStyle, QStylePainter, QApplication
)


class ComboBoxSearch(QComboBox):
    """
    A drop down list combo box with filter/search.

    The popup list view is filtered by text entered in the filter field.

    Note
    ----
    `popup`, `lineEdit` and `completer` from the base QComboBox class are
    unused. Setting/modifying them will have no effect.
    """
    # NOTE: Setting editable + QComboBox.NoInsert policy + ... did not achieve
    # the same results.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__searchline = QLineEdit(self, visible=False, frame=False)
        self.__searchline.setAttribute(Qt.WA_MacShowFocusRect, False)
        self.__searchline.setFocusProxy(self)
        self.__popup = None  # type: Optional[QAbstractItemModel]
        self.__proxy = None  # type: Optional[QSortFilterProxyModel]

    def showPopup(self):
        # type: () -> None
        """
        Reimplemented from QComboBox.showPopup

        Popup up a customized view and filter edit line.

        Note
        ----
        The .popup(), .lineEdit(), .completer() of the base class are not used.
        """
        if self.__popup is not None:
            self.__popup.hide()
            self.__popup.deleteLater()
            self.__popup = self.__proxy = None

        if self.count() == 0:
            return

        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        popup = QListView(
            uniformItemSizes=True,
            horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOff,
            verticalScrollBarPolicy=Qt.ScrollBarAsNeeded,
            iconSize=self.iconSize(),
        )
        popup.setFocusProxy(self.__searchline)
        popup.setParent(self, Qt.Popup | Qt.FramelessWindowHint)
        proxy = QSortFilterProxyModel(
            popup, filterCaseSensitivity=Qt.CaseInsensitive
        )
        proxy.setFilterKeyColumn(self.modelColumn())
        proxy.setSourceModel(self.model())
        popup.setModel(proxy)
        root = proxy.mapFromSource(self.rootModelIndex())
        popup.setRootIndex(root)

        self.__popup = popup
        self.__proxy = proxy
        self.__searchline.setText("")
        self.__searchline.setPlaceholderText("Filter...")
        self.__searchline.setVisible(True)
        # focus proxy is cleared in hidePopup
        self.__searchline.setFocusProxy(self)
        self.__searchline.textEdited.connect(proxy.setFilterFixedString)

        style = self.style()  # type: QStyle

        popuprect_origin = style.subControlRect(
            QStyle.CC_ComboBox, opt, QStyle.SC_ComboBoxListBoxPopup, self
        )  # type: QRect
        popuprect_origin = QRect(
            self.mapToGlobal(popuprect_origin.topLeft()),
            popuprect_origin.size()
        )
        editrect = style.subControlRect(
            QStyle.CC_ComboBox, opt, QStyle.SC_ComboBoxEditField, self
        )  # type: QRect
        self.__searchline.setGeometry(editrect)
        screenrect = QApplication.desktop().screenGeometry(self)  # type: QRect

        # get the height for the view
        listrect = QRect()
        for i in range(min(proxy.rowCount(root), self.maxVisibleItems())):
            index = proxy.index(i, self.modelColumn(), root)
            if index.isValid():
                listrect = listrect.united(popup.visualRect(index))
            if listrect.height() >= screenrect.height():
                break
        window = popup.window()  # type: QWidget
        window.ensurePolished()
        if window.layout() is not None:
            window.layout().activate()
        else:
            QApplication.sendEvent(window, QEvent(QEvent.LayoutRequest))

        margins = qwidget_margin_within(popup.viewport(), window)
        height = (listrect.height() + 2 * popup.spacing() +
                  margins.top() + margins.bottom())

        popup_size = (QSize(popuprect_origin.width(), height)
                      .expandedTo(window.minimumSize())
                      .boundedTo(window.maximumSize())
                      .boundedTo(screenrect.size()))
        popuprect = QRect(popuprect_origin.bottomLeft(), popup_size)

        # if the popup extends bellow the screen and there is more room above
        # the popup origin ...
        if popuprect.bottom() > screenrect.bottom() \
                and popuprect_origin.center().y() > screenrect.center().y():
            # ...flip the rect about the popup_origin so it extends upwards
            popuprect.moveBottom(popuprect_origin.topLeft().y())

        # fixup horizontal position if it extends outside the screen
        if popuprect.left() < screenrect.left():
            popuprect.moveLeft(screenrect.left())
        if popuprect.right() > screenrect.right():
            popuprect.moveRight(screenrect.right())

        # bound by screen geometry
        popuprect = popuprect.intersected(screenrect)

        popup.setGeometry(popuprect)

        current = proxy.mapFromSource(
            self.model().index(self.currentIndex(), self.modelColumn(),
                               self.rootModelIndex()))
        popup.setCurrentIndex(current)
        popup.scrollTo(current, QAbstractItemView.EnsureVisible)
        popup.show()
        popup.setFocus(Qt.PopupFocusReason)
        popup.installEventFilter(self)
        popup.viewport().installEventFilter(self)
        self.update()

    def hidePopup(self):
        """Reimplemented"""
        if self.__popup is not None:
            self.__popup.deleteLater()
            self.__popup = self.__proxy = None
        # need to call base hidePopup even though the base showPopup was not
        # called (update internal state wrt. 'pressed' arrow, ...
        super().hidePopup()
        self.__searchline.setFocusProxy(None)
        self.__searchline.hide()
        self.update()

    def initStyleOption(self, option):
        # type: (QStyleOptionComboBox) -> None
        super().initStyleOption(option)
        option.editable = True

    def __updateGeometries(self):
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        editarea = self.style().subControlRect(
            QStyle.CC_ComboBox, opt, QStyle.SC_ComboBoxEditField, self)
        self.__searchline.setGeometry(editarea)

    def resizeEvent(self, event):
        """Reimplemented."""
        super().resizeEvent(event)
        self.__updateGeometries()

    def paintEvent(self, event):
        """Reimplemented."""
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)

        painter = QStylePainter(self)
        painter.drawComplexControl(QStyle.CC_ComboBox, opt)
        if not self.__searchline.isVisibleTo(self):
            opt.editable = False
            painter.drawControl(QStyle.CE_ComboBoxLabel, opt)

    def eventFilter(self, obj, event):
        # type: (QObject, QEvent) -> bool
        """Reimplemented."""
        etype = event.type()
        if etype == QEvent.FocusOut and self.__popup is not None:
            self.hidePopup()
            return True
        if etype == QEvent.Hide and self.__popup is not None:
            self.hidePopup()
            return False

        if etype == QEvent.KeyPress or etype == QEvent.KeyRelease \
                and obj is self.__popup:
            event = event  # type: QKeyEvent
            key, modifiers = event.key(), event.modifiers()
            if key in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Select):
                current = self.__popup.currentIndex()
                if current.isValid():
                    self.__activateProxyIndex(current)
            elif key in (Qt.Key_Up, Qt.Key_Down,
                         Qt.Key_PageUp, Qt.Key_PageDown):
                return False
            elif key in (Qt.Key_End, Qt.Key_Home) \
                    and not modifiers & Qt.ControlModifier:
                return False
            elif key in (Qt.Key_Tab, Qt.Key_Backtab):
                pass
            elif key == Qt.Key_Escape or \
                    (key == Qt.Key_F4 and modifiers & Qt.AltModifier):
                self.__popup.hide()
            else:
                # pass the input events to the filter edit line
                QCoreApplication.sendEvent(self.__searchline, event)
                return True

        if etype == QEvent.MouseButtonRelease and self.__popup is not None \
                and obj is self.__popup.viewport():
            event = event  # type: QMouseEvent
            index = self.__popup.indexAt(event.pos())
            if index.isValid():
                self.__activateProxyIndex(index)

        return super().eventFilter(obj, event)

    def __activateProxyIndex(self, index):
        # type: (QModelIndex) -> None
        # Set current and activate the source index corresponding to the proxy
        # index in the popup's model.
        if self.__popup is not None and index.isValid():
            proxy = self.__popup.model()
            assert index.model() is proxy
            index = proxy.mapToSource(index)
            assert index.model() is self.model()
            if index.isValid() and \
                    index.flags() & (Qt.ItemIsEnabled | Qt.ItemIsSelectable):
                self.hidePopup()
                text = self.itemText(index.row())
                self.setCurrentIndex(index.row())
                self.activated[int].emit(index.row())
                self.activated[str].emit(text)


def qwidget_margin_within(widget, ancestor):
    # type: (QWidget, QWidget) -> QMargins
    """
    Return the 'margins' of widget within its 'ancestor'

    Ancestor must be within the widget's parent hierarchy and both widgets must
    share the same top level window.

    Parameters
    ----------
    widget : QWidget
    ancestor : QWidget

    Returns
    -------
    margins: QMargins
    """
    assert ancestor.isAncestorOf(widget)
    assert ancestor.window() is widget.window()
    r1 = widget.geometry()
    r2 = ancestor.geometry()
    topleft = r1.topLeft()
    bottomright = r1.bottomRight()
    topleft = widget.mapTo(ancestor, topleft)
    bottomright = widget.mapTo(ancestor, bottomright)
    return QMargins(topleft.x(), topleft.y(),
                    r2.right() - bottomright.x(),
                    r2.bottom() - bottomright.y())
