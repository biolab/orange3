"""
==============
Tool Box Widget
==============

A reimplementation of the :class:`QToolBox` widget that keeps all the tabs
in a single :class:`QScrollArea` instance and can keep multiple open tabs.

"""

from collections import namedtuple
from operator import eq, attrgetter

from PyQt4.QtGui import (
    QWidget, QFrame, QSizePolicy, QIcon, QFontMetrics, QPainter, QStyle,
    QStyleOptionToolButton, QStyleOptionToolBoxV2, QPalette, QBrush, QPen,
    QColor, QScrollArea, QVBoxLayout, QToolButton, QAction, QActionGroup
)

from PyQt4.QtCore import (
    Qt, QObject, QSize, QRect, QPoint, QSignalMapper, QEvent
)

from PyQt4.QtCore import pyqtSignal as Signal, pyqtProperty as Property

from .utils import brush_darker

_ToolBoxPage = namedtuple(
    "_ToolBoxPage",
    ["index",
     "widget",
     "action",
     "button"]
    )


FOCUS_OUTLINE_COLOR = "#609ED7"


class ToolBoxTabButton(QToolButton):
    """
    A tab button for an item in a :class:`ToolBox`.
    """

    def setNativeStyling(self, state):
        """
        Render tab buttons as native (or css styled) :class:`QToolButtons`.
        If set to `False` (default) the button is pained using a custom
        paint routine.

        """
        self.__nativeStyling = state
        self.update()

    def nativeStyling(self):
        """
        Use :class:`QStyle`'s to paint the class:`QToolButton` look.
        """
        return self.__nativeStyling

    nativeStyling_ = Property(bool,
                              fget=nativeStyling,
                              fset=setNativeStyling,
                              designable=True)

    def __init__(self, *args, **kwargs):
        self.__nativeStyling = False
        self.position = QStyleOptionToolBoxV2.OnlyOneTab
        self.selected = QStyleOptionToolBoxV2.NotAdjacent

        QToolButton.__init__(self, *args, **kwargs)

    def paintEvent(self, event):
        if self.__nativeStyling:
            QToolButton.paintEvent(self, event)
        else:
            self.__paintEventNoStyle()

    def __paintEventNoStyle(self):
        p = QPainter(self)
        opt = QStyleOptionToolButton()
        self.initStyleOption(opt)

        fm = QFontMetrics(opt.font)
        palette = opt.palette

        # highlight brush is used as the background for the icon and background
        # when the tab is expanded and as mouse hover color (lighter).
        brush_highlight = palette.highlight()
        if opt.state & QStyle.State_Sunken:
            # State 'down' pressed during a mouse press (slightly darker).
            background_brush = brush_darker(brush_highlight, 110)
        elif opt.state & QStyle.State_MouseOver:
            background_brush = brush_darker(brush_highlight, 95)
        elif opt.state & QStyle.State_On:
            background_brush = brush_highlight
        else:
            # The default button brush.
            background_brush = palette.button()

        rect = opt.rect
        icon = opt.icon
        icon_size = opt.iconSize

        # TODO: add shift for pressed as set by the style (PM_ButtonShift...)

        pm = None
        if not icon.isNull():
            if opt.state & QStyle.State_Enabled:
                mode = QIcon.Normal
            else:
                mode = QIcon.Disabled

            pm = opt.icon.pixmap(
                    rect.size().boundedTo(icon_size), mode,
                    QIcon.On if opt.state & QStyle.State_On else QIcon.Off)

        icon_area_rect = QRect(rect)
        icon_area_rect.setRight(int(icon_area_rect.height() * 1.26))

        text_rect = QRect(rect)
        text_rect.setLeft(icon_area_rect.right() + 10)

        # Background  (TODO: Should the tab button have native
        # toolbutton shape, drawn using PE_PanelButtonTool or even
        # QToolBox tab shape)

        # Default outline pen
        pen = QPen(palette.color(QPalette.Mid))

        p.save()
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(background_brush))
        p.drawRect(rect)

        # Draw the background behind the icon if the background_brush
        # is different.
        if not opt.state & QStyle.State_On:
            p.setBrush(brush_highlight)
            p.drawRect(icon_area_rect)
            # Line between the icon and text
            p.setPen(pen)
            p.drawLine(icon_area_rect.topRight(),
                       icon_area_rect.bottomRight())

        if opt.state & QStyle.State_HasFocus:
            # Set the focus frame pen and draw the border
            pen = QPen(QColor(FOCUS_OUTLINE_COLOR))
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            # Adjust for pen
            rect = rect.adjusted(0, 0, -1, -1)
            p.drawRect(rect)

        else:
            p.setPen(pen)
            # Draw the top/bottom border
            if self.position == QStyleOptionToolBoxV2.OnlyOneTab or \
                    self.position == QStyleOptionToolBoxV2.Beginning or \
                    self.selected & \
                        QStyleOptionToolBoxV2.PreviousIsSelected:

                p.drawLine(rect.topLeft(), rect.topRight())

            p.drawLine(rect.bottomLeft(), rect.bottomRight())

        p.restore()

        p.save()
        text = fm.elidedText(opt.text, Qt.ElideRight, text_rect.width())
        p.setPen(QPen(palette.color(QPalette.ButtonText)))
        p.setFont(opt.font)

        p.drawText(text_rect,
                   int(Qt.AlignVCenter | Qt.AlignLeft) | \
                   int(Qt.TextSingleLine),
                   text)
        if pm:
            pm_rect = QRect(QPoint(0, 0), pm.size())
            centered_rect = QRect(pm_rect)
            centered_rect.moveCenter(icon_area_rect.center())
            p.drawPixmap(centered_rect, pm, pm_rect)
        p.restore()


class _ToolBoxScrollArea(QScrollArea):
    def eventFilter(self, obj, event):
        if obj is self.widget() and event.type() == QEvent.Resize:
            if event.size() == event.oldSize() and self.widgetResizable():
                # This is driving me insane. This should not have happened.
                # Before the event is sent QWidget specifically makes sure the
                # sizes are different, but somehow I still get this, and enter
                # an infinite recursion if I enter QScrollArea.eventFilter.
                # I can only duplicate this on one development machine a
                # Mac OSX using fink and Qt 4.7.3
                return False

        return QScrollArea.eventFilter(self, obj, event)


class ToolBox(QFrame):
    """
    A tool box widget.
    """
    # Emitted when a tab is toggled.
    tabToogled = Signal(int, bool)

    def setExclusive(self, exclusive):
        """
        Set exclusive tabs (only one tab can be open at a time).
        """
        if self.__exclusive != exclusive:
            self.__exclusive = exclusive
            self.__tabActionGroup.setExclusive(exclusive)
            checked = self.__tabActionGroup.checkedAction()
            if checked is None:
                # The action group can be out of sync with the actions state
                # when switching between exclusive states.
                actions_checked = [page.action for page in self.__pages
                                   if page.action.isChecked()]
                if actions_checked:
                    checked = actions_checked[0]

            # Trigger/toggle remaining open pages
            if exclusive and checked is not None:
                for page in self.__pages:
                    if checked != page.action and page.action.isChecked():
                        page.action.trigger()

    def exclusive(self):
        """
        Are the tabs in the toolbox exclusive.
        """
        return self.__exclusive

    exclusive_ = Property(bool,
                          fget=exclusive,
                          fset=setExclusive,
                          designable=True,
                          doc="Exclusive tabs")

    def __init__(self, parent=None, **kwargs):
        QFrame.__init__(self, parent, **kwargs)

        self.__pages = []
        self.__tabButtonHeight = -1
        self.__tabIconSize = QSize()
        self.__exclusive = False
        self.__setupUi()

    def __setupUi(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area for the contents.
        self.__scrollArea = \
                _ToolBoxScrollArea(self, objectName="toolbox-scroll-area")

        self.__scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.__scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.__scrollArea.setSizePolicy(QSizePolicy.MinimumExpanding,
                                        QSizePolicy.MinimumExpanding)
        self.__scrollArea.setFrameStyle(QScrollArea.NoFrame)
        self.__scrollArea.setWidgetResizable(True)

        # A widget with all of the contents.
        # The tabs/contents are placed in the layout inside this widget
        self.__contents = QWidget(self.__scrollArea,
                                  objectName="toolbox-contents")

        # The layout where all the tab/pages are placed
        self.__contentsLayout = QVBoxLayout()
        self.__contentsLayout.setContentsMargins(0, 0, 0, 0)
        self.__contentsLayout.setSizeConstraint(QVBoxLayout.SetMinAndMaxSize)
        self.__contentsLayout.setSpacing(0)

        self.__contents.setLayout(self.__contentsLayout)

        self.__scrollArea.setWidget(self.__contents)

        layout.addWidget(self.__scrollArea)

        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Fixed,
                           QSizePolicy.MinimumExpanding)

        self.__tabActionGroup = \
                QActionGroup(self, objectName="toolbox-tab-action-group")

        self.__tabActionGroup.setExclusive(self.__exclusive)

        self.__actionMapper = QSignalMapper(self)
        self.__actionMapper.mapped[QObject].connect(self.__onTabActionToogled)

    def setTabButtonHeight(self, height):
        """
        Set the tab button height.
        """
        if self.__tabButtonHeight != height:
            self.__tabButtonHeight = height
            for page in self.__pages:
                page.button.setFixedHeight(height)

    def tabButtonHeight(self):
        """
        Return the tab button height.
        """
        return self.__tabButtonHeight

    def setTabIconSize(self, size):
        """
        Set the tab button icon size.
        """
        if self.__tabIconSize != size:
            self.__tabIconSize = size
            for page in self.__pages:
                page.button.setIconSize(size)

    def tabIconSize(self):
        """
        Return the tab icon size.
        """
        return self.__tabIconSize

    def tabButton(self, index):
        """
        Return the tab button at `index`
        """
        return self.__pages[index].button

    def tabAction(self, index):
        """
        Return open/close action for the tab at `index`.
        """
        return self.__pages[index].action

    def addItem(self, widget, text, icon=None, toolTip=None):
        """
        Append the `widget` in a new tab and return its index.

        Parameters
        ----------
        widget : :class:`QWidget`
            A widget to be inserted. The toolbox takes ownership
            of the widget.

        text : str
            Name/title of the new tab.

        icon : :class:`QIcon`, optional
            An icon for the tab button.

        toolTip : str, optional
            Tool tip for the tab button.

        """
        return self.insertItem(self.count(), widget, text, icon, toolTip)

    def insertItem(self, index, widget, text, icon=None, toolTip=None):
        """
        Insert the `widget` in a new tab at position `index`.

        See also
        --------
        ToolBox.addItem

        """
        button = self.createTabButton(widget, text, icon, toolTip)

        self.__contentsLayout.insertWidget(index * 2, button)
        self.__contentsLayout.insertWidget(index * 2 + 1, widget)

        widget.hide()

        page = _ToolBoxPage(index, widget, button.defaultAction(), button)
        self.__pages.insert(index, page)

        for i in range(index + 1, self.count()):
            self.__pages[i] = self.__pages[i]._replace(index=i)

        self.__updatePositions()

        # Show (open) the first tab.
        if self.count() == 1 and index == 0:
            page.action.trigger()

        self.__updateSelected()

        self.updateGeometry()
        return index

    def removeItem(self, index):
        """
        Remove the widget at `index`.

        .. note:: The widget hidden but is is not deleted.

        """
        self.__contentsLayout.takeAt(2 * index + 1)
        self.__contentsLayout.takeAt(2 * index)
        page = self.__pages.pop(index)

        # Update the page indexes
        for i in range(index, self.count()):
            self.__pages[i] = self.__pages[i]._replace(index=i)

        page.button.deleteLater()

        # Hide the widget and reparent to self
        # This follows QToolBox.removeItem
        page.widget.hide()
        page.widget.setParent(self)

        self.__updatePositions()
        self.__updateSelected()

        self.updateGeometry()

    def count(self):
        """
        Return the number of widgets inserted in the toolbox.
        """
        return len(self.__pages)

    def widget(self, index):
        """
        Return the widget at `index`.
        """
        return self.__pages[index].widget

    def createTabButton(self, widget, text, icon=None, toolTip=None):
        """
        Create the tab button for `widget`.
        """
        action = QAction(text, self)
        action.setCheckable(True)

        if icon:
            action.setIcon(icon)

        if toolTip:
            action.setToolTip(toolTip)
        self.__tabActionGroup.addAction(action)
        self.__actionMapper.setMapping(action, action)
        action.toggled.connect(self.__actionMapper.map)

        button = ToolBoxTabButton(self, objectName="toolbox-tab-button")
        button.setDefaultAction(action)
        button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        button.setSizePolicy(QSizePolicy.Expanding,
                             QSizePolicy.Fixed)

        if self.__tabIconSize.isValid():
            button.setIconSize(self.__tabIconSize)

        if self.__tabButtonHeight > 0:
            button.setFixedHeight(self.__tabButtonHeight)

        return button

    def ensureWidgetVisible(self, child, xmargin=50, ymargin=50):
        """
        Scroll the contents so child widget instance is visible inside
        the viewport.

        """
        self.__scrollArea.ensureWidgetVisible(child, xmargin, ymargin)

    def sizeHint(self):
        hint = self.__contentsLayout.sizeHint()

        if self.count():
            # Compute max width of hidden widgets also.
            scroll = self.__scrollArea
            scroll_w = scroll.verticalScrollBar().sizeHint().width()
            frame_w = self.frameWidth() * 2 + scroll.frameWidth() * 2
            max_w = max([p.widget.sizeHint().width() for p in self.__pages])
            hint = QSize(max(max_w, hint.width()) + scroll_w + frame_w,
                         hint.height())

        return QSize(200, 200).expandedTo(hint)

    def __onTabActionToogled(self, action):
        page = find(self.__pages, action, key=attrgetter("action"))
        on = action.isChecked()
        page.widget.setVisible(on)
        index = page.index

        if index > 0:
            # Update the `previous` tab buttons style hints
            previous = self.__pages[index - 1].button
            flag = QStyleOptionToolBoxV2.NextIsSelected
            if on:
                previous.selected |= flag
            else:
                previous.selected &= ~flag

            previous.update()

        if index < self.count() - 1:
            next = self.__pages[index + 1].button
            flag = QStyleOptionToolBoxV2.PreviousIsSelected
            if on:
                next.selected |= flag
            else:
                next.selected &= ~flag

            next.update()

        self.tabToogled.emit(index, on)

        self.__contentsLayout.invalidate()

    def __updateSelected(self):
        """Update the tab buttons selected style flags.
        """
        if self.count() == 0:
            return

        opt = QStyleOptionToolBoxV2

        def update(button, next_sel, prev_sel):
            if next_sel:
                button.selected |= opt.NextIsSelected
            else:
                button.selected &= ~opt.NextIsSelected

            if prev_sel:
                button.selected |= opt.PreviousIsSelected
            else:
                button.selected &= ~ opt.PreviousIsSelected

            button.update()

        if self.count() == 1:
            update(self.__pages[0].button, False, False)
        elif self.count() >= 2:
            pages = self.__pages
            for i in range(1, self.count() - 1):
                update(pages[i].button,
                       pages[i + 1].action.isChecked(),
                       pages[i - 1].action.isChecked())

    def __updatePositions(self):
        """Update the tab buttons position style flags.
        """
        if self.count() == 0:
            return
        elif self.count() == 1:
            self.__pages[0].button.position = QStyleOptionToolBoxV2.OnlyOneTab
        else:
            self.__pages[0].button.position = QStyleOptionToolBoxV2.Beginning
            self.__pages[-1].button.position = QStyleOptionToolBoxV2.End
            for p in self.__pages[1:-1]:
                p.button.position = QStyleOptionToolBoxV2.Middle

        for p in self.__pages:
            p.button.update()


def identity(arg):
    return arg


def find(iterable, *what, **kwargs):
    """
    find(iterable, [what, [key=None, [predicate=operator.eq]]])
    """
    if what:
        what = what[0]
    key, predicate = kwargs.get("key", identity), kwargs.get("predicate", eq)
    for item in iterable:
        item_key = key(item)
        if predicate(item_key, what):
            return item
    else:
        raise ValueError(what)
