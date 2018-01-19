from collections import Sequence

from AnyQt.QtWidgets import QListWidget, QListView, QListWidgetItem
from AnyQt.QtCore import Qt, QSize, QItemSelection

from Orange.widgets.utils import getdeepattr
from Orange.widgets.utils.delegates import TableVariable

from .base import miscellanea
from .boxes import vBox, hBox
from .callbacks import connect_control

try:
    from Orange.data import Variable
    from .varicons import attributeItem, attributeIconDict
except ImportError:
    Variable = ()  # This will return False in isinstance(x, Variable)

__all__ = ["listBox", "listView", "ListViewWithSizeHint"]


class ListViewWithSizeHint(QListView):
    def __init__(self, *args, preferred_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(preferred_size, tuple):
            preferred_size = QSize(*preferred_size)
        self.preferred_size = preferred_size

    def sizeHint(self):
        return self.preferred_size if self.preferred_size is not None \
            else super().sizeHint()


class OrangeListBox(QListWidget):
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
        self.size_hint = sizeHint if sizeHint is not None else QSize(150, 100)
        if enableDragDrop:
            self.setDragEnabled(True)
            self.setAcceptDrops(True)
            self.setDropIndicatorShown(True)

    def sizeHint(self):
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
            return None
        self._updatingGeometriesNow = True
        try:
            return super().updateGeometries()
        finally:
            self._updatingGeometriesNow = False
        return None

def listView(widget, master, value=None, model=None, box=None, callback=None,
             sizeHint=None, **misc):
    def update_selection(values):
        sel_model = view.selectionModel()
        if not isinstance(values, Sequence):
            values = [values]
        selection = QItemSelection()
        for value in values:
            if not isinstance(value, int):
                if isinstance(value, Variable):
                    search_role = TableVariable
                else:
                    search_role = Qt.DisplayRole
                    value = str(value)
                for i in range(model.rowCount()):
                    if model.data(model.index(i), search_role) == value:
                        value = i
                        break
            selection.select(model.index(value), model.index(value))
        sel_model.select(selection, sel_model.ClearAndSelect)

    def update_attribute(*_):
        # This must be imported locally to avoid circular imports
        from Orange.widgets.utils.itemmodels import PyListModel
        values = [i.row() for i in view.selectionModel().selection().indexes()]
        if values:
            # FIXME: irrespective of PyListModel check, this might/should always
            # callback with values!
            if isinstance(model, PyListModel):
                values = [model[i] for i in values]
            if view.selectionMode() == view.SingleSelection:
                values = values[0]
            if getattr(master, value) != values:  # avoid redundant signals
                setattr(master, value, values)

    bg = vBox(widget, box, addToLayout=False) if box else widget
    view = ListViewWithSizeHint(preferred_size=sizeHint)
    view.setModel(model)
    if value is not None:
        connect_control(
            master, value, view,
            signal=view.selectionModel().selectionChanged,
            update_control=update_selection,
            update_value=update_attribute,
            callback=callback)
    misc.setdefault('addSpace', True)
    miscellanea(view, bg, widget, **misc)
    return view


def listBox(widget, master, value=None, labels=None, box=None, callback=None,
            selectionMode=QListWidget.SingleSelection,
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
    def update_labels(values):
        lb.clear()
        if values:
            for val in values:
                if isinstance(val, tuple):
                    text, icon = val
                    if isinstance(icon, int):
                        item = QListWidgetItem(attributeIconDict[icon], text)
                    else:
                        item = QListWidgetItem(icon, text)
                elif isinstance(val, Variable):
                    item = QListWidgetItem(*attributeItem(val))
                else:
                    item = QListWidgetItem(val)
                item.setData(Qt.UserRole, val)
                lb.addItem(item)

    def update_selection(val):
        if val is not None:
            if isinstance(val, int):
                for i in range(lb.count()):
                    lb.item(i).setSelected(i == val)
            else:
                if not isinstance(val, ControlledList):
                    setattr(master, value, ControlledList(val, lb))
                for i in range(lb.count()):
                    shouldBe = i in val
                    if shouldBe != lb.item(i).isSelected():
                        lb.item(i).setSelected(shouldBe)

    def update_attribute():
        clist = getdeepattr(master, value)
        selection = [i for i in range(lb.count()) if lb.item(i).isSelected()]
        if isinstance(clist, int):
            master.__setattr__(value, selection[0] if selection else None)
        else:
            list.__setitem__(clist, slice(0, len(clist)), selection)
            master.__setattr__(value, clist)

    bg = hBox(widget, box, addToLayout=False) if box else widget
    lb = OrangeListBox(master, enableDragDrop, dragDropCallback,
                       dataValidityCallback, sizeHint, bg)
    lb.setSelectionMode(selectionMode)

    if labels is not None:
        setattr(master, labels, getdeepattr(master, labels))
        master.connect_control(labels, update_labels)
    if value is not None:
        clist = getdeepattr(master, value)
        if not isinstance(clist, (int, ControlledList)):
            clist = ControlledList(clist, lb)
            master.__setattr__(value, clist)
        setattr(master, value, clist)
        connect_control(
            master, value, lb,
            signal=lb.itemSelectionChanged,
            update_control=update_selection,
            update_value=update_attribute,
            callback=callback)

    misc.setdefault('addSpace', True)
    miscellanea(lb, bg, widget, **misc)
    return lb


class ControlledList(list):
    """
    A class derived from a list that is connected to a
    :obj:`QListBox`: the list contains indices of items that are
    selected in the list box. Changing the list content changes the
    selection in the list box.
    """
    def __init__(self, content, listBox=None):
        super().__init__(content if content is not None else [])
        self.listBox = listBox

    def __reduce__(self):
        # cannot pickle self.listBox, but can't discard it
        # (ControlledList may live on)
        import copyreg
        return copyreg._reconstructor, (list, list, ()), None, self.__iter__()

    # TODO ControllgedList.item2name is probably never used
    def item2name(self, item):
        item = self.listBox.labels[item]
        if isinstance(item, tuple):
            return item[1]
        else:
            return item

    def __setitem__(self, index, item):
        def unselect(i):
            try:
                item = self.listBox.item(i)
            except RuntimeError:  # Underlying C/C++ object has been deleted
                item = None
            if item is None:
                # Labels changed before clearing the selection: clear everything
                self.listBox.selectionModel().clear()
            else:
                item.setSelected(0)

        if isinstance(index, int):
            unselect(self[index])
            item.setSelected(1)
        else:
            for i in self[index]:
                unselect(i)
            for i in item:
                self.listBox.item(i).setSelected(1)
        super().__setitem__(index, item)

    def __delitem__(self, index):
        if isinstance(index, int):
            self.listBox.item(self[index]).setSelected(0)
        else:
            for i in self[index]:
                self.listBox.item(i).setSelected(0)
        super().__delitem__(index)

    def append(self, item):
        super().append(item)
        item.setSelected(1)

    def extend(self, items):
        super().extend(items)
        for i in items:
            self.listBox.item(i).setSelected(1)

    def insert(self, index, item):
        item.setSelected(1)
        super().insert(index, item)

    def pop(self, index=-1):
        i = super().pop(index)
        self.listBox.item(i).setSelected(0)

    def remove(self, item):
        item.setSelected(0)
        super().remove(item)
