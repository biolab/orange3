from AnyQt.QtWidgets import QListView, QLineEdit, QCompleter
from AnyQt.QtGui import QDrag
from AnyQt.QtCore import (
    Qt, QObject, QEvent, QModelIndex,
    QAbstractItemModel, QSortFilterProxyModel, QStringListModel,
)
from AnyQt.QtCore import pyqtSignal as Signal

from Orange.widgets.utils.itemmodels import VariableListModel, PyListModel
import Orange


def slices(indices):
    """ Group the given integer indices into slices
    """
    indices = list(sorted(indices))
    if indices:
        first = last = indices[0]
        for i in indices[1:]:
            if i == last + 1:
                last = i
            else:
                yield first, last + 1
                first = last = i
        yield first, last + 1


def delslice(model, start, end):
    """ Delete the start, end slice (rows) from the model.
    """
    if isinstance(model, PyListModel):
        del model[start:end]
    elif isinstance(model, QAbstractItemModel):
        model.removeRows(start, end - start)
    else:
        raise TypeError(type(model))


class VariablesListItemView(QListView):
    """ A Simple QListView subclass initialized for displaying
    variables.
    """
    #: Emitted with a Qt.DropAction when a drag/drop (originating from this
    #: view) completed successfully
    dragDropActionDidComplete = Signal(int)

    def __init__(self, parent=None, acceptedType=Orange.data.Variable):
        super().__init__(parent)
        self.setSelectionMode(self.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(self.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDragDropOverwriteMode(False)
        self.viewport().setAcceptDrops(True)

        #: type | Tuple[type]
        self.__acceptedType = acceptedType

    def startDrag(self, supported_actions):
        indices = self.selectionModel().selectedIndexes()
        indices = [i for i in indices if i.flags() & Qt.ItemIsDragEnabled]
        if indices:
            data = self.model().mimeData(indices)
            if not data:
                return

            drag = QDrag(self)
            drag.setMimeData(data)

            default_action = Qt.IgnoreAction
            if self.defaultDropAction() != Qt.IgnoreAction and \
                    supported_actions & self.defaultDropAction():
                default_action = self.defaultDropAction()
            elif (supported_actions & Qt.CopyAction and
                  self.dragDropMode() != self.InternalMove):
                default_action = Qt.CopyAction
            res = drag.exec_(supported_actions, default_action)
            if res == Qt.MoveAction:
                selected = self.selectionModel().selectedIndexes()
                rows = list(map(QModelIndex.row, selected))
                for s1, s2 in reversed(list(slices(rows))):
                    delslice(self.model(), s1, s2)
            self.dragDropActionDidComplete.emit(res)

    def dragEnterEvent(self, event):
        """
        Reimplemented from QListView.dragEnterEvent
        """
        if self.acceptsDropEvent(event):
            event.accept()
        else:
            event.ignore()

    def acceptsDropEvent(self, event):
        """
        Should the drop event be accepted?
        """
        # disallow drag/drops between windows
        if event.source() is not None and \
                event.source().window() is not self.window():
            return False

        mime = event.mimeData()
        vars = mime.property('_items')
        if vars is None:
            return False

        if not all(isinstance(var, self.__acceptedType) for var in vars):
            return False

        event.accept()
        return True


class VariableFilterProxyModel(QSortFilterProxyModel):
    """ A proxy model for filtering a list of variables based on
    their names and labels.

    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._filter_string = ""

    def set_filter_string(self, filter):
        self._filter_string = str(filter).lower()
        self.invalidateFilter()

    def filter_accepts_variable(self, var):
        row_str = var.name + " ".join(("%s=%s" % item)
                                      for item in var.attributes.items())
        row_str = row_str.lower()
        filters = self._filter_string.split()

        return all(f in row_str for f in filters)

    def filterAcceptsRow(self, source_row, source_parent):
        model = self.sourceModel()
        if isinstance(model, VariableListModel):
            var = model[source_row]
            return self.filter_accepts_variable(var)
        else:
            return True


class CompleterNavigator(QObject):
    """ An event filter to be installed on a QLineEdit, to enable
    Key up/ down to navigate between posible completions.
    """
    def eventFilter(self, obj, event):
        if (event.type() == QEvent.KeyPress and
                isinstance(obj, QLineEdit)):
            if event.key() == Qt.Key_Down:
                diff = 1
            elif event.key() == Qt.Key_Up:
                diff = -1
            else:
                return False
            completer = obj.completer()
            if completer is not None and completer.completionCount() > 0:
                current = completer.currentRow()
                current += diff
                completer.setCurrentRow(current % completer.completionCount())
                completer.complete()
            return True
        else:
            return False


class VariablesFilter:
    def __init__(self, parent=None, model=None):
        self.original_completer_items = []
        super().__init__()
        self.filter_edit = QLineEdit()
        self.filter_edit.setToolTip("Filter the list of available variables.")
        if hasattr(self.filter_edit, "setPlaceholderText"):
            self.filter_edit.setPlaceholderText("Filter")

        self.completer = QCompleter()
        self.completer.setCompletionMode(QCompleter.InlineCompletion)
        self.completer_model = QStringListModel()
        self.completer.setModel(self.completer_model)
        self.completer.setModelSorting(
            QCompleter.CaseSensitivelySortedModel)

        self.filter_edit.setCompleter(self.completer)
        self.completer_navigator = CompleterNavigator(parent)
        self.filter_edit.installEventFilter(self.completer_navigator)

        self.available_attrs = model
        self.available_attrs_proxy = VariableFilterProxyModel()
        self.available_attrs_proxy.setSourceModel(self.available_attrs)
        self.available_attrs_view = VariablesListItemView(
            acceptedType=Orange.data.Variable)
        self.available_attrs_view.setModel(self.available_attrs_proxy)

        aa = self.available_attrs
        aa.dataChanged.connect(self.update_completer_model)
        aa.rowsInserted.connect(self.update_completer_model)
        aa.rowsRemoved.connect(self.update_completer_model)

        self.filter_edit.textChanged.connect(self.update_completer_prefix)
        self.filter_edit.textChanged.connect(
            self.available_attrs_proxy.set_filter_string)

    def update_completer_model(self, *_):
        """ This gets called when the model for available attributes changes
        through either drag/drop or the left/right button actions.

        """
        vars = list(self.available_attrs)
        items = [var.name for var in vars]
        items += ["%s=%s" % item for v in vars for item in v.attributes.items()]

        new = sorted(set(items))
        if new != self.original_completer_items:
            self.original_completer_items = new
            self.completer_model.setStringList(self.original_completer_items)

    def update_completer_prefix(self, filter):
        """ Prefixes all items in the completer model with the current
        already done completion to enable the completion of multiple keywords.
        """
        prefix = str(self.completer.completionPrefix())
        if not prefix.endswith(" ") and " " in prefix:
            prefix, _ = prefix.rsplit(" ", 1)
            items = [prefix + " " + item
                     for item in self.original_completer_items]
        else:
            items = self.original_completer_items
        old = list(map(str, self.completer_model.stringList()))

        if set(old) != set(items):
            self.completer_model.setStringList(items)
