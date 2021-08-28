"""
Wrappers for controls used in widgets
"""
import math

import logging
import sys
import warnings
import weakref
from collections.abc import Sequence

from AnyQt import QtWidgets, QtCore, QtGui
from AnyQt.QtCore import Qt, QSize, QItemSelection
from AnyQt.QtGui import QColor, QWheelEvent
from AnyQt.QtWidgets import QWidget, QListView, QComboBox

from orangewidget.utils.itemdelegates import (
    BarItemDataDelegate as _BarItemDataDelegate
)
# re-export relevant objects
from orangewidget.gui import (
    OWComponent, OrangeUserRole, TableView, resource_filename,
    miscellanea, setLayout, separator, rubber, widgetBox, hBox, vBox,
    comboBox as gui_comboBox,
    indentedBox, widgetLabel, label, spin, doubleSpin, checkBox, lineEdit,
    button, toolButton, radioButtons, radioButtonsInBox, appendRadioButton,
    hSlider, labeledSlider, valueSlider, auto_commit, auto_send, auto_apply,
    deferred,

    # ItemDataRole's
    BarRatioRole, BarBrushRole, SortOrderRole, LinkRole,

    IndicatorItemDelegate, BarItemDelegate, LinkStyledItemDelegate,
    ColoredBarItemDelegate, HorizontalGridDelegate, VerticalItemDelegate,
    VerticalLabel, tabWidget, createTabPage, table, tableItem,
    VisibleHeaderSectionContextEventFilter,
    checkButtonOffsetHint, toolButtonSizeHint, FloatSlider,
    CalendarWidgetWithTime, DateTimeEditWCalendarTime,
    ControlGetter, VerticalScrollArea, ProgressBar,
    ControlledCallback, ControlledCallFront, ValueCallback, connectControl,
    is_macstyle
)


try:
    # Some Orange widgets might expect this here
    # pylint: disable=unused-import
    from Orange.widgets.utils.webview import WebviewWidget
except ImportError:
    pass  # Neither WebKit nor WebEngine are available

import Orange.data
from Orange.widgets.utils import getdeepattr
from Orange.data import \
    ContinuousVariable, StringVariable, TimeVariable, DiscreteVariable, \
    Variable, Value
from Orange.widgets.utils import vartype

__all__ = [
    # Re-exported
    "OWComponent", "OrangeUserRole", "TableView", "resource_filename",
    "miscellanea", "setLayout", "separator", "rubber",
    "widgetBox", "hBox", "vBox", "indentedBox",
    "widgetLabel", "label", "spin", "doubleSpin",
    "checkBox", "lineEdit", "button", "toolButton", "comboBox",
    "radioButtons", "radioButtonsInBox", "appendRadioButton",
    "hSlider", "labeledSlider", "valueSlider",
    "auto_commit", "auto_send", "auto_apply", "ProgressBar",
    "VerticalLabel", "tabWidget", "createTabPage", "table", "tableItem",
    "VisibleHeaderSectionContextEventFilter", "checkButtonOffsetHint",
    "toolButtonSizeHint", "FloatSlider", "ControlGetter",  "VerticalScrollArea",
    "CalendarWidgetWithTime", "DateTimeEditWCalendarTime",
    "BarRatioRole", "BarBrushRole", "SortOrderRole", "LinkRole",
    "BarItemDelegate", "IndicatorItemDelegate", "LinkStyledItemDelegate",
    "ColoredBarItemDelegate", "HorizontalGridDelegate", "VerticalItemDelegate",
    "ValueCallback", 'is_macstyle',
    # Defined here
    "createAttributePixmap", "attributeIconDict", "attributeItem",
    "listView", "ListViewWithSizeHint", "listBox", "OrangeListBox",
    "TableValueRole", "TableClassValueRole", "TableDistribution",
    "TableVariable", "TableBarItem", "palette_combo_box"
]


log = logging.getLogger(__name__)


def palette_combo_box(initial_palette):
    from Orange.widgets.utils import itemmodels
    cb = QComboBox()
    model = itemmodels.ContinuousPalettesModel()
    cb.setModel(model)
    cb.setCurrentIndex(model.indexOf(initial_palette))
    cb.setIconSize(QSize(64, 16))
    return cb


def createAttributePixmap(char, background=Qt.black, color=Qt.white):
    """
    Create a QIcon with a given character. The icon is 13 pixels high and wide.

    :param char: The character that is printed in the icon
    :type char: str
    :param background: the background color (default: black)
    :type background: QColor
    :param color: the character color (default: white)
    :type color: QColor
    :rtype: QIcon
    """
    icon = QtGui.QIcon()
    for size in (13, 16, 18, 20, 22, 24, 28, 32, 64):
        pixmap = QtGui.QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QtGui.QPainter()
        painter.begin(pixmap)
        painter.setRenderHints(painter.Antialiasing | painter.TextAntialiasing |
                               painter.SmoothPixmapTransform)
        painter.setPen(background)
        painter.setBrush(background)
        margin = 1 + size // 16
        text_margin = size // 20
        rect = QtCore.QRectF(margin, margin,
                             size - 2 * margin, size - 2 * margin)
        painter.drawRoundedRect(rect, 30.0, 30.0, Qt.RelativeSize)
        painter.setPen(color)
        font = painter.font()  # type: QtGui.QFont
        font.setPixelSize(size - 2 * margin - 2 * text_margin)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, char)
        painter.end()
        icon.addPixmap(pixmap)
    return icon


class __AttributeIconDict(dict):
    def __getitem__(self, key):
        if not self:
            for tpe, char, col in ((vartype(ContinuousVariable("c")),
                                    "N", (202, 0, 32)),
                                   (vartype(DiscreteVariable("d")),
                                    "C", (26, 150, 65)),
                                   (vartype(StringVariable("s")),
                                    "S", (0, 0, 0)),
                                   (vartype(TimeVariable("t")),
                                    "T", (68, 170, 255)),
                                   (-1, "?", (128, 128, 128))):
                self[tpe] = createAttributePixmap(char, QtGui.QColor(*col))
        if key not in self:
            key = vartype(key) if isinstance(key, Variable) else -1
        return super().__getitem__(key)


#: A dict that returns icons for different attribute types. The dict is
#: constructed on first use since icons cannot be created before initializing
#: the application.
#:
#: Accepted keys are variable type codes and instances
#: of :obj:`Orange.data.Variable`: `attributeIconDict[var]` will give the
#: appropriate icon for variable `var` or a question mark if the type is not
#: recognized
attributeIconDict = __AttributeIconDict()


def attributeItem(var):
    """
    Construct a pair (icon, name) for inserting a variable into a combo or
    list box

    :param var: variable
    :type var: Orange.data.Variable
    :rtype: tuple with QIcon and str
    """
    return attributeIconDict[var], var.name


class ListViewWithSizeHint(QListView):
    def __init__(self, *args, preferred_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(preferred_size, tuple):
            preferred_size = QSize(*preferred_size)
        self.preferred_size = preferred_size

    def sizeHint(self):
        return self.preferred_size if self.preferred_size is not None \
            else super().sizeHint()


def listView(widget, master, value=None, model=None, box=None, callback=None,
             sizeHint=None, *, viewType=ListViewWithSizeHint, **misc):
    if box:
        bg = vBox(widget, box, addToLayout=False)
    else:
        bg = widget
    view = viewType(preferred_size=sizeHint)
    view.setModel(model)
    if value is not None:
        connectControl(master, value, callback,
                       view.selectionModel().selectionChanged,
                       CallFrontListView(view),
                       CallBackListView(model, view, master, value))
    misc.setdefault('uniformItemSizes', True)
    miscellanea(view, bg, widget, **misc)
    return view


def listBox(widget, master, value=None, labels=None, box=None, callback=None,
            selectionMode=QtWidgets.QListWidget.SingleSelection,
            enableDragDrop=False, dragDropCallback=None,
            dataValidityCallback=None, sizeHint=None, **misc):
    """
    Insert a list box.

    The value with which the box's value synchronizes (`master.<value>`)
    is a list of indices of selected items.

    :param widget: the widget into which the box is inserted
    :type widget: QWidget or None
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param value: the name of the master's attribute with which the value is
        synchronized (list of ints - indices of selected items)
    :type value: str
    :param labels: the name of the master's attribute with the list of items
        (as strings or tuples with icon and string)
    :type labels: str
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :param callback: a function that is called when the selection state is
        changed
    :type callback: function
    :param selectionMode: selection mode - single, multiple etc
    :type selectionMode: QAbstractItemView.SelectionMode
    :param enableDragDrop: flag telling whether drag and drop is available
    :type enableDragDrop: bool
    :param dragDropCallback: callback function on drop event
    :type dragDropCallback: function
    :param dataValidityCallback: function that check the validity on enter
        and move event; it should return either `ev.accept()` or `ev.ignore()`.
    :type dataValidityCallback: function
    :param sizeHint: size hint
    :type sizeHint: QSize
    :rtype: OrangeListBox
    """
    if box:
        bg = hBox(widget, box, addToLayout=False)
    else:
        bg = widget
    lb = OrangeListBox(master, enableDragDrop, dragDropCallback,
                       dataValidityCallback, sizeHint, bg)
    lb.setSelectionMode(selectionMode)
    lb.ogValue = value
    lb.ogLabels = labels
    lb.ogMaster = master

    if labels is not None:
        setattr(master, labels, getdeepattr(master, labels))
        master.connect_control(labels, CallFrontListBoxLabels(lb))
    if value is not None:
        clist = getdeepattr(master, value)
        if not isinstance(clist, (int, ControlledList)):
            clist = ControlledList(clist, lb)
            master.__setattr__(value, clist)
        setattr(master, value, clist)
        connectControl(master, value, callback, lb.itemSelectionChanged,
                       CallFrontListBox(lb), CallBackListBox(lb, master))

    miscellanea(lb, bg, widget, **misc)
    return lb


class OrangeListBox(QtWidgets.QListWidget):
    """
    List box with drag and drop functionality. Function :obj:`listBox`
    constructs instances of this class; do not use the class directly.

    .. attribute:: master

        The widget into which the listbox is inserted.

    .. attribute:: ogLabels

        The name of the master's attribute that holds the strings with items
        in the list box.

    .. attribute:: ogValue

        The name of the master's attribute that holds the indices of selected
        items.

    .. attribute:: enableDragDrop

        A flag telling whether drag-and-drop is enabled.

    .. attribute:: dragDropCallback

        A callback that is called at the end of drop event.

    .. attribute:: dataValidityCallback

        A callback that is called on dragEnter and dragMove events and returns
        either `ev.accept()` or `ev.ignore()`.

    .. attribute:: defaultSizeHint

        The size returned by the `sizeHint` method.
    """
    def __init__(self, master, enableDragDrop=False, dragDropCallback=None,
                 dataValidityCallback=None, sizeHint=None, *args):
        """
        :param master: the master widget
        :type master: OWWidget or OWComponent
        :param enableDragDrop: flag telling whether drag and drop is enabled
        :type enableDragDrop: bool
        :param dragDropCallback: callback for the end of drop event
        :type dragDropCallback: function
        :param dataValidityCallback: callback that accepts or ignores dragEnter
            and dragMove events
        :type dataValidityCallback: function with one argument (event)
        :param sizeHint: size hint
        :type sizeHint: QSize
        :param args: optional arguments for the inherited constructor
        """
        self.master = master
        super().__init__(*args)
        self.drop_callback = dragDropCallback
        self.valid_data_callback = dataValidityCallback
        if not sizeHint:
            self.size_hint = QtCore.QSize(150, 100)
        else:
            self.size_hint = sizeHint
        if enableDragDrop:
            self.setDragEnabled(True)
            self.setAcceptDrops(True)
            self.setDropIndicatorShown(True)

    def sizeHint(self):
        return self.size_hint

    def minimumSizeHint(self):
        return self.size_hint

    def dragEnterEvent(self, event):
        super().dragEnterEvent(event)
        if self.valid_data_callback:
            self.valid_data_callback(event)
        elif isinstance(event.source(), OrangeListBox):
            event.setDropAction(Qt.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        event.setDropAction(Qt.MoveAction)
        super().dropEvent(event)

        items = self.update_master()
        if event.source() is not self:
            event.source().update_master(exclude=items)

        if self.drop_callback:
            self.drop_callback()

    def update_master(self, exclude=()):
        control_list = [self.item(i).data(Qt.UserRole)
                        for i in range(self.count())
                        if self.item(i).data(Qt.UserRole) not in exclude]
        if self.ogLabels:
            master_list = getattr(self.master, self.ogLabels)

            if master_list != control_list:
                setattr(self.master, self.ogLabels, control_list)
        return control_list

    def updateGeometries(self):
        # A workaround for a bug in Qt
        # (see: http://bugreports.qt.nokia.com/browse/QTBUG-14412)
        if getattr(self, "_updatingGeometriesNow", False):
            return
        self._updatingGeometriesNow = True
        try:
            return super().updateGeometries()
        finally:
            self._updatingGeometriesNow = False


class ControlledList(list):
    """
    A class derived from a list that is connected to a
    :obj:`QListBox`: the list contains indices of items that are
    selected in the list box. Changing the list content changes the
    selection in the list box.
    """
    def __init__(self, content, listBox=None):
        super().__init__(content if content is not None else [])
        # Controlled list is created behind the back by gui.listBox and
        # commonly used as a setting which gets synced into a GLOBAL
        # SettingsHandler and which keeps the OWWidget instance alive via a
        # reference in listBox (see gui.listBox)
        if listBox is not None:
            self.listBox = weakref.ref(listBox)
        else:
            self.listBox = lambda: None

    def __reduce__(self):
        # cannot pickle self.listBox, but can't discard it
        # (ControlledList may live on)
        import copyreg
        return copyreg._reconstructor, (list, list, ()), None, self.__iter__()

    # TODO ControllgedList.item2name is probably never used
    def item2name(self, item):
        item = self.listBox().labels[item]
        if isinstance(item, tuple):
            return item[1]
        else:
            return item

    def __setitem__(self, index, item):
        def unselect(i):
            try:
                item = self.listBox().item(i)
            except RuntimeError:  # Underlying C/C++ object has been deleted
                item = None
            if item is None:
                # Labels changed before clearing the selection: clear everything
                self.listBox().selectionModel().clear()
            else:
                item.setSelected(0)

        if isinstance(index, int):
            unselect(self[index])
            item.setSelected(1)
        else:
            for i in self[index]:
                unselect(i)
            for i in item:
                self.listBox().item(i).setSelected(1)
        super().__setitem__(index, item)

    def __delitem__(self, index):
        if isinstance(index, int):
            self.listBox().item(self[index]).setSelected(0)
        else:
            for i in self[index]:
                self.listBox().item(i).setSelected(0)
        super().__delitem__(index)

    def append(self, item):
        super().append(item)
        item.setSelected(1)

    def extend(self, items):
        super().extend(items)
        for i in items:
            self.listBox().item(i).setSelected(1)

    def insert(self, index, item):
        item.setSelected(1)
        super().insert(index, item)

    def pop(self, index=-1):
        i = super().pop(index)
        self.listBox().item(i).setSelected(0)

    def remove(self, item):
        item.setSelected(0)
        super().remove(item)


def comboBox(*args, **kwargs):
    if "valueType" in kwargs:
        del kwargs["valueType"]
        warnings.warn("Argument 'valueType' is deprecated and ignored",
                      DeprecationWarning)
    return gui_comboBox(*args, **kwargs)


class CallBackListView(ControlledCallback):
    def __init__(self, model, view, widget, attribute):
        super().__init__(widget, attribute)
        self.model = model
        self.view = view

    # triggered by selectionModel().selectionChanged()
    def __call__(self, *_):
        # This must be imported locally to avoid circular imports
        from Orange.widgets.utils.itemmodels import PyListModel
        values = [i.row()
                  for i in self.view.selectionModel().selection().indexes()]
        if values:
            # FIXME: irrespective of PyListModel check, this might/should always
            # callback with values!
            if isinstance(self.model, PyListModel):
                values = [self.model[i] for i in values]
            if self.view.selectionMode() == self.view.SingleSelection:
                values = values[0]
            self.acyclic_setattr(values)


class CallBackListBox:
    def __init__(self, control, widget):
        self.control = control
        self.widget = widget
        self.disabled = 0

    def __call__(self, *_):  # triggered by selectionChange()
        if not self.disabled and self.control.ogValue is not None:
            clist = getdeepattr(self.widget, self.control.ogValue)
            control = self.control
            selection = [i for i in range(control.count())
                         if control.item(i).isSelected()]
            if isinstance(clist, int):
                self.widget.__setattr__(
                    self.control.ogValue, selection[0] if selection else None)
            else:
                list.__setitem__(clist, slice(0, len(clist)), selection)
                self.widget.__setattr__(self.control.ogValue, clist)


##############################################################################
# call fronts (change of the attribute value changes the related control)


class CallFrontListView(ControlledCallFront):
    def action(self, values):
        view = self.control
        model = view.model()
        sel_model = view.selectionModel()

        if not isinstance(values, Sequence):
            values = [values]

        selection = QItemSelection()
        for value in values:
            index = None
            if not isinstance(value, int):
                if isinstance(value, Variable):
                    search_role = TableVariable
                else:
                    search_role = Qt.DisplayRole
                    value = str(value)
                for i in range(model.rowCount()):
                    if model.data(model.index(i), search_role) == value:
                        index = i
                        break
            else:
                index = value
            if index is not None:
                selection.select(model.index(index), model.index(index))
        sel_model.select(selection, sel_model.ClearAndSelect)


class CallFrontListBox(ControlledCallFront):
    def action(self, value):
        if value is not None:
            if isinstance(value, int):
                for i in range(self.control.count()):
                    self.control.item(i).setSelected(i == value)
            else:
                if not isinstance(value, ControlledList):
                    setattr(self.control.ogMaster, self.control.ogValue,
                            ControlledList(value, self.control))
                for i in range(self.control.count()):
                    shouldBe = i in value
                    if shouldBe != self.control.item(i).isSelected():
                        self.control.item(i).setSelected(shouldBe)


class CallFrontListBoxLabels(ControlledCallFront):
    unknownType = None

    def action(self, values):
        self.control.clear()
        if values:
            for value in values:
                if isinstance(value, tuple):
                    text, icon = value
                    if isinstance(icon, int):
                        item = QtWidgets.QListWidgetItem(attributeIconDict[icon], text)
                    else:
                        item = QtWidgets.QListWidgetItem(icon, text)
                elif isinstance(value, Variable):
                    item = QtWidgets.QListWidgetItem(*attributeItem(value))
                else:
                    item = QtWidgets.QListWidgetItem(value)

                item.setData(Qt.UserRole, value)
                self.control.addItem(item)


#: Role to retrieve Orange.data.Value
TableValueRole = next(OrangeUserRole)
#: Role to retrieve class value for a row
TableClassValueRole = next(OrangeUserRole)
# Role to retrieve distribution of a column
TableDistribution = next(OrangeUserRole)
#: Role to retrieve the column's variable
TableVariable = next(OrangeUserRole)


class TableBarItem(_BarItemDataDelegate):
    BarRole = next(OrangeUserRole)
    BarColorRole = next(OrangeUserRole)
    __slots__ = ("color_schema",)

    def __init__(
            self, parent=None, color=QColor(255, 170, 127), width=5,
            barFillRatioRole=BarRole, barColorRole=BarColorRole,
            color_schema=None,
            **kwargs
    ):
        """
        :param QObject parent: Parent object.
        :param QColor color: Default color of the distribution bar.
        :param color_schema:
            If not None it must be an instance of
            :class:`OWColorPalette.ColorPaletteGenerator` (note: this
            parameter, if set, overrides the ``color``)
        :type color_schema: :class:`OWColorPalette.ColorPaletteGenerator`
        """
        super().__init__(
            parent, color=color, penWidth=width,
            barFillRatioRole=barFillRatioRole, barColorRole=barColorRole,
            **kwargs
        )
        self.color_schema = color_schema

    def barColorData(self, index):
        class_ = self.cachedData(index, TableClassValueRole)
        if self.color_schema is not None \
                and isinstance(class_, Value) \
                and isinstance(class_.variable, DiscreteVariable) \
                and not math.isnan(class_):
            return self.color_schema[int(class_)]
        return self.cachedData(index, self.BarColorRole)


from Orange.widgets.utils.colorpalettes import patch_variable_colors
patch_variable_colors()


class HScrollStepMixin:
    """
    Overrides default TableView horizontal behavior (scrolls 1 page at a time)
    to a friendlier scroll speed akin to that of vertical scrolling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.horizontalScrollBar().setSingleStep(20)

    def wheelEvent(self, event: QWheelEvent):
        if event.source() == Qt.MouseEventNotSynthesized and \
                (event.modifiers() & Qt.ShiftModifier and sys.platform == 'darwin' or
                 event.modifiers() & Qt.AltModifier and sys.platform != 'darwin'):
            new_event = QWheelEvent(
                event.pos(), event.globalPos(), event.pixelDelta(),
                event.angleDelta(), event.buttons(), Qt.NoModifier,
                event.phase(), event.inverted(), Qt.MouseEventSynthesizedByApplication
            )
            event.accept()
            super().wheelEvent(new_event)
        else:
            super().wheelEvent(event)
