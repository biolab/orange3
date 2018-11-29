"""
Edit Domain
-----------

A widget for manual editing of a domain's attributes.

"""
import warnings
from xml.sax.saxutils import escape
from itertools import zip_longest
from contextlib import contextmanager
from collections import namedtuple
from functools import singledispatch

from typing import (
    Tuple, List, Any, Optional, Union, Dict, Sequence, Iterable, NamedTuple,
)

from AnyQt.QtWidgets import (
    QWidget, QListView, QTreeView, QVBoxLayout, QHBoxLayout, QFormLayout,
    QToolButton, QLineEdit, QAction, QActionGroup, QStackedWidget, QGroupBox,
    QStyledItemDelegate, QStyleOptionViewItem, QStyle, QSizePolicy, QToolTip,
    QDialogButtonBox, QPushButton
)
from AnyQt.QtGui import QStandardItemModel, QStandardItem, QKeySequence, QIcon
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot
from AnyQt.QtCore import Qt, QEvent, QSize, QModelIndex

import numpy as np

import Orange.data

from Orange.preprocess.transformation import Identity, Lookup
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output

#: An ordered sequence of key, value pairs (variable annotations)
AnnotationsType = Tuple[Tuple[str, str], ...]


# Define abstract representation of the variable types edited

class Categorical(
    NamedTuple("Categorical", [
        ("name", str),
        ("categories", Tuple[str, ...]),
        ("base", Optional[str]),
        ("annotations", AnnotationsType),
    ])): pass


class Real(
    NamedTuple("Real", [
        ("name", str),
        # a precision (int, and a format specifier('f', 'g', or '')
        ("format", Tuple[int, str]),
        ("annotations", AnnotationsType),
    ])): pass


class String(
    NamedTuple("String", [
        ("name", str),
        ("annotations", AnnotationsType),
    ])): pass


class Time(
    NamedTuple("Time", [
        ("name", str),
        ("annotations", AnnotationsType),
    ])): pass


Variable = Union[Categorical, Real, Time, String]
VariableTypes = (Categorical, Real, Time, String)


# Define variable transformations.

class Rename(namedtuple("Rename", ["name"])):
    """
    Rename a variable.

    Parameters
    ----------
    name : str
        The new name
    """
    def __call__(self, var):
        # type: (Variable) -> Variable
        return var._replace(name=self.name)


#: Mapping of categories.
#: A list of pairs with the first element the original value and the second
#: element the new value. If the first element is None then a category level
#: is added. If the second element is None than the corresponding first level
#: is dropped. The translation list is sorted by the translated levels.
CategoriesMappingType = List[Tuple[Optional[str], Optional[str]]]


class CategoriesMapping(namedtuple("CategoriesMapping", ["mapping"])):
    """
    Change categories of a categorical variable.

    Parameters
    ----------
    mapping : CategoriesMappingType
    """
    def __call__(self, var):
        # type: (Categorical) -> Categorical
        mapping = self.mapping  # type: CategoriesMappingType
        cat = [cj for _, cj in mapping if cj is not None]
        mm = dict(self.mapping)
        if var.base is not None:
            base = mm.get(var.base)
        else:
            base = None
        return var._replace(categories=cat, base=base)


class Annotate(namedtuple("Annotate", ["annotations"])):
    """
    Replace variable annotations.
    """
    def __call__(self, var):
        return var._replace(annotations=self.annotations)


Transform = Union[Rename, CategoriesMapping, Annotate]
TransformTypes = (Rename, CategoriesMapping, Annotate)


def deconstruct(obj):
    # type: (tuple) -> Tuple[str, Tuple[Any, ...]]
    """
    Deconstruct a tuple subclass to its class name and its contents.

    Parameters
    ----------
    obj : A tuple

    Returns
    -------
    value: Tuple[str, Tuple[Any, ...]]
    """
    cname = type(obj).__name__
    args = tuple(obj)
    return cname, args


def reconstruct(tname, args):
    # type: (str, Tuple[Any, ...]) -> Tuple[Any, ...]
    """
    Reconstruct a tuple subclass (inverse of deconstruct).

    Parameters
    ----------
    tname : str
        Type name
    args : Tuple[Any, ...]

    Returns
    -------
    rval: Tuple[Any, ...]
    """
    try:
        constructor = globals()[tname]
    except KeyError:
        raise NameError(tname)
    return constructor(*args)


class DictItemsModel(QStandardItemModel):
    """A Qt Item Model class displaying the contents of a python
    dictionary.

    """
    # Implement a proper model with in-place editing.
    # (Maybe it should be a TableModel with 2 columns)
    def __init__(self, parent=None, dict=None):
        super().__init__(parent)
        self._dict = {}
        self.setHorizontalHeaderLabels(["Key", "Value"])
        if dict is not None:
            self.set_dict(dict)

    def set_dict(self, dict):
        # type: (Dict[str, str]) -> None
        self._dict = dict
        self.setRowCount(0)
        for key, value in sorted(dict.items()):
            key_item = QStandardItem(key)
            value_item = QStandardItem(value)
            key_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            value_item.setFlags(value_item.flags() | Qt.ItemIsEditable)
            self.appendRow([key_item, value_item])

    def get_dict(self):
        # type: () -> Dict[str, str]
        rval = {}
        for row in range(self.rowCount()):
            key_item = self.item(row, 0)
            value_item = self.item(row, 1)
            rval[key_item.text()] = value_item.text()
        return rval


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

    def event(self, event):
        # type: (QEvent) -> bool
        if event.type() == QEvent.ToolTip and self.toolTip():
            action = self.defaultAction()
            if action is not None:
                text = "<span>{}</span>&nbsp;&nbsp;<kbd>{}</kbd>".format(
                    action.toolTip(),
                    action.shortcut().toString(QKeySequence.NativeText)
                )
                QToolTip.showText(event.globalPos(), text)
            else:
                QToolTip.hideText()
            return True
        else:
            return super().event(event)


class VariableEditor(QWidget):
    """
    An editor widget for a variable.

    Can edit the variable name, and its attributes dictionary.
    """
    variable_changed = Signal()

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.var = None  # type: Optional[Variable]

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.form = form = QFormLayout(
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
            objectName="editor-form-layout"
        )
        layout.addLayout(self.form)

        self.name_edit = QLineEdit(objectName="name-editor")
        self.name_edit.editingFinished.connect(
            lambda: self.name_edit.isModified() and self.on_name_changed()
        )
        form.addRow("Name:", self.name_edit)

        vlayout = QVBoxLayout(margin=0, spacing=1)
        self.labels_edit = view = QTreeView(
            objectName="annotation-pairs-edit",
            rootIsDecorated=False,
            editTriggers=QTreeView.DoubleClicked | QTreeView.EditKeyPressed,
        )
        self.labels_model = model = DictItemsModel()
        view.setModel(model)

        view.selectionModel().selectionChanged.connect(
            self.on_label_selection_changed)

        agrp = QActionGroup(view, objectName="annotate-action-group")
        action_add = QAction(
            "+", self, objectName="action-add-label",
            toolTip="Add a new label.",
            shortcut=QKeySequence(QKeySequence.New),
            shortcutContext=Qt.WidgetShortcut
        )
        action_delete = QAction(
            "\N{MINUS SIGN}", self, objectName="action-delete-label",
            toolTip="Remove selected label.",
            shortcut=QKeySequence(QKeySequence.Delete),
            shortcutContext=Qt.WidgetShortcut
        )
        agrp.addAction(action_add)
        agrp.addAction(action_delete)
        view.addActions([action_add, action_delete])

        def add_label():
            row = [QStandardItem(), QStandardItem()]
            model.appendRow(row)
            idx = model.index(model.rowCount() - 1, 0)
            view.setCurrentIndex(idx)
            view.edit(idx)

        def remove_label():
            rows = view.selectionModel().selectedRows(0)
            if rows:
                assert len(rows) == 1
                idx = rows[0].row()
                model.removeRow(idx)

        action_add.triggered.connect(add_label)
        action_delete.triggered.connect(remove_label)
        agrp.setEnabled(False)

        self.add_label_action = action_add
        self.remove_label_action = action_delete

        # Necessary signals to know when the labels change
        model.dataChanged.connect(self.on_labels_changed)
        model.rowsInserted.connect(self.on_labels_changed)
        model.rowsRemoved.connect(self.on_labels_changed)

        vlayout.addWidget(self.labels_edit)
        hlayout = QHBoxLayout()
        hlayout.setContentsMargins(0, 0, 0, 0)
        button = FixedSizeButton(
            self, defaultAction=self.add_label_action,
            accessibleName="Add",
        )
        hlayout.addWidget(button)

        button = FixedSizeButton(
            self, defaultAction=self.remove_label_action,
            accessibleName="Remove",
        )

        hlayout.addWidget(button)
        hlayout.addStretch(10)
        vlayout.addLayout(hlayout)
        form.addRow("Labels:", vlayout)

    def set_data(self, var, transform=()):
        # type: (Optional[Variable], Sequence[Transform]) -> None
        """
        Set the variable to edit.
        """
        self.clear()
        self.var = var
        if var is not None:
            name = var.name
            annotations = var.annotations
            for tr in transform:
                if isinstance(tr, Rename):
                    name = tr.name
                elif isinstance(tr, Annotate):
                    annotations = tr.annotations
            self.name_edit.setText(name)
            self.labels_model.set_dict(dict(annotations))
            self.add_label_action.actionGroup().setEnabled(True)
        else:
            self.add_label_action.actionGroup().setEnabled(False)

    def get_data(self):
        """Retrieve the modified variable.
        """
        if self.var is None:
            return None, []
        name = self.name_edit.text().strip()
        labels = tuple(sorted(self.labels_model.get_dict().items()))
        tr = []
        if self.var.name != name:
            tr.append(Rename(name))
        if self.var.annotations != labels:
            tr.append(Annotate(labels))
        return self.var, tr

    def clear(self):
        """Clear the editor state.
        """
        self.var = None
        self.name_edit.setText("")
        self.labels_model.setRowCount(0)

    @Slot()
    def on_name_changed(self):
        self.variable_changed.emit()

    @Slot()
    def on_labels_changed(self):
        self.variable_changed.emit()

    @Slot()
    def on_label_selection_changed(self):
        selected = self.labels_edit.selectionModel().selectedRows()
        self.remove_label_action.setEnabled(bool(len(selected)))


@contextmanager
def disconnected(signal, slot, connection_type=Qt.AutoConnection):
    signal.disconnect(slot)
    try:
        yield
    finally:
        signal.connect(slot, connection_type)


#: In 'reordable' models holds the original position of the item
#: (if applicable).
SourcePosRole = Qt.UserRole
#: The original name
SourceNameRole = Qt.UserRole + 2

#: The added/dropped state (type is ItemEditState)
EditStateRole = Qt.UserRole + 1


class ItemEditState:
    NoState = 0
    Dropped = 1
    Added = 2


class CategoriesEditDelegate(QStyledItemDelegate):
    """
    Display delegate for editing categories (styled display for add,
    remove and rename operations).
    """
    def initStyleOption(self, option, index):
        # type: (QStyleOptionViewItem, QModelIndex)-> None
        super().initStyleOption(option, index)
        text = str(index.data(Qt.EditRole))
        editstate = index.data(EditStateRole)
        sourcename = index.data(SourceNameRole)
        suffix = None
        if editstate == ItemEditState.Dropped:
            option.state &= ~QStyle.State_Enabled
            option.font.setStrikeOut(True)
            text = sourcename
            suffix = "(dropped)"
        elif editstate == ItemEditState.Added:
            suffix = "(added)"
        elif isinstance(sourcename, str) and sourcename != text:
            text = "{sourcename} \N{RIGHTWARDS ARROW} {text}".format(
                sourcename=sourcename, text=text
            )

        if suffix is not None:
            text = text + " " + suffix
        option.text = text


class DiscreteVariableEditor(VariableEditor):
    """An editor widget for editing a discrete variable.

    Extends the :class:`VariableEditor` to enable editing of
    variables values.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        form = self.layout().itemAt(0)
        assert isinstance(form, QFormLayout)

        #: A list model of discrete variable's values.
        self.values_model = itemmodels.PyListModel(
            flags=Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        )

        vlayout = QVBoxLayout(spacing=1, margin=0)
        self.values_edit = QListView(
            editTriggers=QListView.DoubleClicked | QListView.EditKeyPressed
        )
        self.values_edit.setItemDelegate(CategoriesEditDelegate(self))
        self.values_edit.setModel(self.values_model)
        self.values_model.dataChanged.connect(self.on_values_changed)

        self.values_edit.selectionModel().selectionChanged.connect(
            self.on_value_selection_changed)
        self.values_model.layoutChanged.connect(self.on_value_selection_changed)
        self.values_model.rowsMoved.connect(self.on_value_selection_changed)

        vlayout.addWidget(self.values_edit)
        hlayout = QHBoxLayout(spacing=1, margin=0)

        self.categories_action_group = group = QActionGroup(
            self, objectName="action-group-categories", enabled=False
        )
        self.move_value_up = QAction(
            "\N{UPWARDS ARROW}", group,
            toolTip="Move the selected item up.",
            shortcut=QKeySequence(Qt.ControlModifier | Qt.AltModifier |
                                  Qt.Key_BracketLeft),
            shortcutContext=Qt.WidgetShortcut,
        )
        self.move_value_up.triggered.connect(self.move_up)

        self.move_value_down = QAction(
            "\N{DOWNWARDS ARROW}", group,
            toolTip="Move the selected item down.",
            shortcut=QKeySequence(Qt.ControlModifier | Qt.AltModifier |
                                  Qt.Key_BracketRight),
            shortcutContext=Qt.WidgetShortcut,
        )
        self.move_value_down.triggered.connect(self.move_down)

        self.add_new_item = QAction(
            "+", group,
            objectName="action-add-item",
            toolTip="Append a new item.",
            shortcut=QKeySequence(QKeySequence.New),
            shortcutContext=Qt.WidgetShortcut,
        )
        self.remove_item = QAction(
            "\N{MINUS SIGN}", group,
            objectName="action-remove-item",
            toolTip="Delete the selected item.",
            shortcut=QKeySequence(QKeySequence.Delete),
            shortcutContext=Qt.WidgetShortcut,
        )

        self.add_new_item.triggered.connect(self._add_category)
        self.remove_item.triggered.connect(self._remove_category)

        button1 = FixedSizeButton(
            self, defaultAction=self.move_value_up,
            accessibleName="Move up"
        )
        button2 = FixedSizeButton(
            self, defaultAction=self.move_value_down,
            accessibleName="Move down"
        )
        button3 = FixedSizeButton(
            self, defaultAction=self.add_new_item,
            accessibleName="Add"
        )
        button4 = FixedSizeButton(
            self, defaultAction=self.remove_item,
            accessibleName="Remove"
        )
        self.values_edit.addActions([self.move_value_up, self.move_value_down,
                                     self.add_new_item, self.remove_item])
        hlayout.addWidget(button1)
        hlayout.addWidget(button2)
        hlayout.addSpacing(3)
        hlayout.addWidget(button3)
        hlayout.addWidget(button4)

        hlayout.addStretch(10)
        vlayout.addLayout(hlayout)

        form.insertRow(1, "Values:", vlayout)

        QWidget.setTabOrder(self.name_edit, self.values_edit)
        QWidget.setTabOrder(self.values_edit, button1)
        QWidget.setTabOrder(button1, button2)
        QWidget.setTabOrder(button2, button3)
        QWidget.setTabOrder(button3, button4)

    def set_data(self, var, transform=()):
        # type: (Optional[Categorical], Sequence[Transform]) -> None
        """
        Set the variable to edit.
        """
        super().set_data(var, transform)
        tr = None  # type: Optional[CategoriesMapping]
        for tr_ in transform:
            if isinstance(tr_, CategoriesMapping):
                tr = tr_

        items = []
        if tr is not None:
            ci_index = {c: i for i, c in enumerate(var.categories)}
            for ci, cj in tr.mapping:
                if ci is None and cj is not None:
                    # level added
                    item = {
                        Qt.EditRole: cj,
                        EditStateRole: ItemEditState.Added,
                        SourcePosRole: None
                    }
                elif ci is not None and cj is None:
                    # ci level dropped
                    item = {
                        Qt.EditRole: ci,
                        EditStateRole: ItemEditState.Dropped,
                        SourcePosRole: ci_index[ci],
                        SourceNameRole: ci
                    }
                elif ci is not None and cj is not None:
                    # rename or reorder
                    item = {
                        Qt.EditRole: cj,
                        EditStateRole: ItemEditState.NoState,
                        SourcePosRole: ci_index[ci],
                        SourceNameRole: ci
                    }
                else:
                    assert False, "invalid mapping: {!r}".format(tr.mapping)
                items.append(item)
        elif var is not None:
            items = [
                {Qt.EditRole: c,
                 EditStateRole: ItemEditState.NoState,
                 SourcePosRole: i,
                 SourceNameRole: c}
                for i, c in enumerate(var.categories)
            ]
        else:
            items = []

        with disconnected(self.values_model.dataChanged,
                          self.on_values_changed):
            self.values_model.clear()
            self.values_model.insertRows(0, len(items))
            for i, item in enumerate(items):
                self.values_model.setItemData(
                    self.values_model.index(i, 0),
                    item
                )
        self.add_new_item.actionGroup().setEnabled(var is not None)

    def __categories_mapping(self):
        # type: () -> CategoriesMappingType
        model = self.values_model
        source = self.var.categories

        res = []
        for i in range(model.rowCount()):
            midx = model.index(i, 0)
            category = midx.data(Qt.EditRole)
            source_pos = midx.data(SourcePosRole)  # type: Optional[int]
            if source_pos is not None:
                source_name = source[source_pos]
            else:
                source_name = None
            state = midx.data(EditStateRole)
            if state == ItemEditState.Dropped:
                res.append((source_name, None))
            elif state == ItemEditState.Added:
                res.append((None, category))
            else:
                res.append((source_name, category))
        return res

    def get_data(self):
        """Retrieve the modified variable
        """
        var, tr = super().get_data()
        if var is None:
            return var, tr
        mapping = self.__categories_mapping()
        if any(_1 != _2 or _2 != _3
               for (_1, _2), _3 in zip_longest(mapping, var.categories)):
            tr.append(CategoriesMapping(mapping))
        return var, tr

    def clear(self):
        """Clear the model state.
        """
        super().clear()
        self.values_model.clear()

    def move_rows(self, rows, offset):
        if not rows:
            return
        assert len(rows) == 1
        i = rows[0].row()
        if offset > 0:
            offset += 1
        self.values_model.moveRows(QModelIndex(), i, 1, QModelIndex(), i + offset)
        self.variable_changed.emit()

    def move_up(self):
        rows = self.values_edit.selectionModel().selectedRows()
        self.move_rows(rows, -1)

    def move_down(self):
        rows = self.values_edit.selectionModel().selectedRows()
        self.move_rows(rows, 1)

    @Slot()
    def on_values_changed(self):
        self.variable_changed.emit()

    @Slot()
    def on_value_selection_changed(self):
        rows = self.values_edit.selectionModel().selectedRows()
        if rows:
            i = rows[0].row()
            self.move_value_up.setEnabled(i)
            self.move_value_down.setEnabled(i != self.values_model.rowCount() - 1)
        else:
            self.move_value_up.setEnabled(False)
            self.move_value_down.setEnabled(False)

    def _remove_category(self):
        """
        Remove the current selected category.

        If the item is an existing category present in the source variable it
        is marked as removed in the view. But if it was added in the set
        transformation it is removed entirely from the model and view.
        """
        view = self.values_edit
        rows = view.selectionModel().selectedRows(0)
        if not rows:
            return
        assert len(rows) == 1
        index = rows[0]  # type: QModelIndex
        model = index.model()
        state = index.data(EditStateRole)
        pos = index.data(Qt.UserRole)
        if pos is not None and pos >= 0:
            # existing level -> only mark/toggle its dropped state
            model.setData(
                index,
                ItemEditState.Dropped if state != ItemEditState.Dropped
                else ItemEditState.NoState,
                EditStateRole)
        elif state == ItemEditState.Added:
            # new level -> remove it
            model.removeRow(index.row())
        else:
            assert False, "invalid state '{}' for {}" \
                .format(state, index.row())

    def _add_category(self):
        """
        Add a new category
        """
        view = self.values_edit
        model = view.model()

        with disconnected(model.dataChanged, self.on_values_changed,
                          Qt.UniqueConnection):
            row = model.rowCount()
            if not model.insertRow(model.rowCount()):
                return
            index = model.index(row, 0)
            model.setItemData(
                index, {
                    Qt.EditRole: "",
                    SourcePosRole: None,
                    EditStateRole: ItemEditState.Added
                }
            )
            view.setCurrentIndex(index)
            view.edit(index)
        self.on_values_changed()


class ContinuousVariableEditor(VariableEditor):
    # TODO: enable editing of display format...
    pass


class TimeVariableEditor(VariableEditor):
    # TODO: enable editing of display format...
    pass


def variable_icon(var):
    # type: (Variable) -> QIcon
    if isinstance(var, Categorical):
        return gui.attributeIconDict[1]
    elif isinstance(var, Real):
        return gui.attributeIconDict[2]
    elif isinstance(var, String):
        return gui.attributeIconDict[3]
    elif isinstance(var, Time):
        return gui.attributeIconDict[4]
    else:
        return gui.attributeIconDict[-1]


#: ItemDataRole storing the variable transform (`List[Transform]`)
TransformRole = Qt.UserRole + 42


class VariableEditDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        # type: (QStyleOptionViewItem, QModelIndex) -> None
        super().initStyleOption(option, index)
        item = index.data(Qt.EditRole)
        var = tr = None
        if isinstance(item, VariableTypes):
            var = item
            option.icon = variable_icon(item)
        elif isinstance(item, Orange.data.Variable):
            var = item
            option.icon = gui.attributeIconDict[var]

        if not option.icon.isNull():
            option.features |= QStyleOptionViewItem.HasDecoration

        transform = index.data(TransformRole)
        if not isinstance(transform, list):
            transform = []

        if var is not None:
            text = var.name
            for tr in transform:
                if isinstance(tr, Rename):
                    text = ("{} \N{RIGHTWARDS ARROW} {}"
                            .format(var.name, tr.name))
            option.text = text
        if tr:
            # mark as changed (maybe also change color, add text, ...)
            option.font.setItalic(True)


# Item model for edited variables (Variable). Define a display role to be the
# source variable name. This is used only in keyboard search. The display is
# otherwise completely handled by a delegate.
class VariableListModel(itemmodels.PyListModel):
    def data(self, index, role=Qt.DisplayRole):
        # type: (QModelIndex, Qt.ItemDataRole) -> Any
        row = index.row()
        if not index.isValid() or not 0 <= row < self.rowCount():
            return None
        if role == Qt.DisplayRole:
            item = self[row]
            if isinstance(item, VariableTypes):
                return item.name
        return super().data(index, role)


class OWEditDomain(widget.OWWidget):
    name = "Edit Domain"
    description = "Rename variables, edit categories and variable annotations."
    icon = "icons/EditDomain.svg"
    priority = 3125
    keywords = []

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        data = Output("Data", Orange.data.Table)

    class Error(widget.OWWidget.Error):
        duplicate_var_name = widget.Msg("A variable name is duplicated.")

    settingsHandler = settings.DomainContextHandler()
    settings_version = 2

    _domain_change_store = settings.ContextSetting({})
    _selected_item = settings.ContextSetting(None)  # type: Optional[str]

    want_control_area = False

    def __init__(self):
        super().__init__()
        self.data = None  # type: Optional[Orange.data.Table]
        #: The current selected variable index
        self.selected_index = -1
        self._invalidated = False

        mainlayout = self.mainArea.layout()
        assert isinstance(mainlayout, QVBoxLayout)
        layout = QHBoxLayout()
        mainlayout.addLayout(layout)
        box = QGroupBox("Variables")
        box.setLayout(QVBoxLayout())
        layout.addWidget(box)

        self.variables_model = VariableListModel(parent=self)
        self.variables_view = self.domain_view = QListView(
            selectionMode=QListView.SingleSelection,
            uniformItemSizes=True,
        )
        self.variables_view.setItemDelegate(VariableEditDelegate(self))
        self.variables_view.setModel(self.variables_model)
        self.variables_view.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        box.layout().addWidget(self.variables_view)

        box = QGroupBox("Edit", )
        box.setLayout(QVBoxLayout(margin=4))
        layout.addWidget(box)

        self.editor_stack = QStackedWidget()

        self.editor_stack.addWidget(DiscreteVariableEditor())
        self.editor_stack.addWidget(ContinuousVariableEditor())
        self.editor_stack.addWidget(TimeVariableEditor())
        self.editor_stack.addWidget(VariableEditor())

        box.layout().addWidget(self.editor_stack)

        bbox = QDialogButtonBox()
        bbox.setStyleSheet(
            "button-layout: {:d};".format(QDialogButtonBox.MacLayout))
        bapply = QPushButton(
            "Apply",
            objectName="button-apply",
            toolTip="Apply changes and commit data on output.",
            default=True,
            autoDefault=False
        )
        bapply.clicked.connect(self.commit)
        breset = QPushButton(
            "Reset Selected",
            objectName="button-reset",
            toolTip="Rest selected variable to its input state.",
            autoDefault=False
        )
        breset.clicked.connect(self.reset_selected)
        breset_all = QPushButton(
            "Reset All",
            objectName="button-reset-all",
            toolTip="Reset all variables to their input state.",
            autoDefault=False
        )
        breset_all.clicked.connect(self.reset_all)

        bbox.addButton(bapply, QDialogButtonBox.AcceptRole)
        bbox.addButton(breset, QDialogButtonBox.ResetRole)
        bbox.addButton(breset_all, QDialogButtonBox.ResetRole)

        mainlayout.addWidget(bbox)
        self.variables_view.setFocus(Qt.NoFocusReason)  # initial focus

    @Inputs.data
    def set_data(self, data):
        """Set input dataset."""
        self.closeContext()
        self.clear()
        self.data = data

        if self.data is not None:
            self.set_domain(data.domain)
            self.openContext(self.data)
            self._restore()

        self.commit()

    def clear(self):
        """Clear the widget state."""
        self.data = None
        self.variables_model.clear()
        assert self.selected_index == -1
        self.selected_index = -1

        self._selected_item = None
        self._domain_change_store = {}

    def reset_selected(self):
        """Reset the currently selected variable to its original state."""
        ind = self.selected_var_index()
        if ind >= 0:
            model = self.variables_model
            midx = model.index(ind)
            var = midx.data(Qt.EditRole)
            tr = midx.data(TransformRole)
            if not tr:
                return  # nothing to reset
            editor = self.editor_stack.currentWidget()
            with disconnected(editor.variable_changed,
                              self._on_variable_changed):
                model.setData(midx, [], TransformRole)
                editor.set_data(var, [])
            self._invalidate()

    def reset_all(self):
        """Reset all variables to their original state."""
        self._domain_change_store = {}
        if self.data is not None:
            model = self.variables_model
            for i in range(model.rowCount()):
                midx = model.index(i)
                model.setData(midx, [], TransformRole)
            index = self.selected_var_index()
            if index >= 0:
                self.open_editor(index)
            self._invalidate()

    def selected_var_index(self):
        """Return the current selected variable index."""
        rows = self.variables_view.selectedIndexes()
        assert len(rows) <= 1
        return rows[0].row() if rows else -1

    def set_domain(self, domain):
        # type: (Orange.data.Domain) -> None
        self.variables_model[:] = [abstract(v)
                                   for v in domain.variables + domain.metas]

    def _restore(self, ):
        """
        Restore the edit transform from saved state.
        """
        model = self.variables_model
        for i in range(model.rowCount()):
            midx = model.index(i, 0)
            var = model.data(midx, Qt.EditRole)
            tr = self._restore_transform(var)
            if tr:
                model.setData(midx, tr, TransformRole)

        # Restore the current variable selection
        i = -1
        if self._selected_item is not None:
            for i, var in enumerate(model):
                if var.name == self._selected_item:
                    break
        if i == -1 and model.rowCount():
            i = 0

        if i != -1:
            itemmodels.select_row(self.variables_view, i)

    def _on_selection_changed(self):
        self.selected_index = self.selected_var_index()
        if self.selected_index != -1:
            self._selected_item = self.variables_model[self.selected_index].name
        else:
            self._selected_item = None
        self.open_editor(self.selected_index)

    def open_editor(self, index):
        # type: (int) -> None
        self.clear_editor()
        model = self.variables_model
        if not 0 <= index < model.rowCount():
            return
        idx = model.index(index, 0)
        var = model.data(idx, Qt.EditRole)
        tr = model.data(idx, TransformRole)
        if tr is None:
            tr = []

        editors = {
            Categorical: 0,
            Real: 1,
            Time: 2,
            String: 3
        }

        editor_index = editors.get(type(var), 3)
        editor = self.editor_stack.widget(editor_index)
        self.editor_stack.setCurrentWidget(editor)
        editor.set_data(var, tr)
        editor.variable_changed.connect(
            self._on_variable_changed, Qt.UniqueConnection
        )

    def clear_editor(self):
        current = self.editor_stack.currentWidget()
        try:
            current.variable_changed.disconnect(self._on_variable_changed)
        except TypeError:
            pass
        current.set_data(None)

    @Slot()
    def _on_variable_changed(self):
        """User edited the current variable in editor."""
        assert 0 <= self.selected_index <= len(self.variables_model)
        editor = self.editor_stack.currentWidget()
        var, transform = editor.get_data()
        model = self.variables_model
        midx = model.index(self.selected_index, 0)
        model.setData(midx, transform, TransformRole)
        self._store_transform(var, transform)
        self._invalidate()

    def _store_transform(self, var, transform):
        # type: (Variable, List[Transform]) -> None
        self._domain_change_store[deconstruct(var)] = [deconstruct(t) for t in transform]

    def _restore_transform(self, var):
        # type: (Variable) -> List[Transform]
        tr_ = self._domain_change_store.get(deconstruct(var), [])
        tr = []

        for t in tr_:
            try:
                tr.append(reconstruct(*t))
            except (NameError, TypeError) as err:
                warnings.warn(
                    "Failed to restore transform: {}, {!r}".format(t, err),
                    UserWarning, stacklevel=2
                )
        return tr

    def _invalidate(self):
        self._set_modified(True)

    def _set_modified(self, state):
        self._invalidated = state
        b = self.findChild(QPushButton, "button-apply")
        if isinstance(b, QPushButton):
            f = b.font()
            f.setItalic(state)
            b.setFont(f)

    def commit(self):
        """
        Apply the changes to the input data and send the changed data to output.
        """
        self._set_modified(False)
        self.Error.duplicate_var_name.clear()

        data = self.data
        if data is None:
            self.Outputs.data.send(None)
            return
        model = self.variables_model

        def state(i):
            # type: (int) -> Tuple[Variable, List[Transform]]
            midx = self.variables_model.index(i, 0)
            return (model.data(midx, Qt.EditRole),
                    model.data(midx, TransformRole))

        state = [state(i) for i in range(model.rowCount())]
        if all(tr is None or not tr for _, tr in state):
            self.Outputs.data.send(data)
            return

        output_vars = []
        input_vars = data.domain.variables + data.domain.metas
        assert all(v_.name == v.name
                   for v, (v_, _) in zip(input_vars, state))
        for (_, tr), v in zip(state, input_vars):
            if tr is not None and len(tr) > 0:
                var = apply_transform(v, tr)
            else:
                var = v
            output_vars.append(var)

        if len(output_vars) != len({v.name for v in output_vars}):
            self.Error.duplicate_var_name()
            self.Outputs.data.send(None)
            return

        domain = data.domain
        nx = len(domain.attributes)
        ny = len(domain.class_vars)
        domain = Orange.data.Domain(
            output_vars[:nx], output_vars[nx: nx + ny], output_vars[nx + ny:]
        )
        new_data = data.transform(domain)
        # print(new_data)
        self.Outputs.data.send(new_data)

    def sizeHint(self):
        sh = super().sizeHint()
        return sh.expandedTo(QSize(660, 550))

    def send_report(self):

        if self.data is not None:
            model = self.variables_model
            state = ((model.data(midx, Qt.EditRole),
                      model.data(midx, TransformRole))
                     for i in range(model.rowCount())
                     for midx in [model.index(i)])
            parts = []
            for var, trs in state:
                if trs:
                    parts.append(report_transform(var, trs))
            if parts:
                html = ("<ul>" +
                        "".join(map("<li>{}</li>".format, parts)) +
                        "</ul>")
            else:
                html = "No changes"
            self.report_raw("", html)
        else:
            self.report_data(None)

    @classmethod
    def migrate_context(cls, context, version):
        # pylint: disable=bad-continuation
        if version is None or version <= 1:
            hints_ = context.values.get("domain_change_hints", ({}, -2))[0]
            store = []
            ns = "Orange.data.variable"
            mapping = {
                "DiscreteVariable":
                    lambda name, args, attrs:
                        ("Categorical", (name, tuple(args[0][1]), None, ())),
                "TimeVariable":
                    lambda name, _, attrs:
                        ("Time", (name, ())),
                "ContinuousVariable":
                    lambda name, _, attrs:
                        ("Real", (name, (3, "f"), ())),
                "StringVariable":
                    lambda name, _, attrs:
                        ("String", (name, ())),
            }
            for (module, class_name, *rest), target in hints_.items():
                if module != ns:
                    continue
                f = mapping.get(class_name)
                if f is None:
                    continue
                trs = []
                key_mapped = f(*rest)
                item_mapped = f(*target[2:])
                src = reconstruct(*key_mapped)   # type: Variable
                dst = reconstruct(*item_mapped)  # type: Variable
                if src.name != dst.name:
                    trs.append(Rename(dst.name))
                if src.annotations != dst.annotations:
                    trs.append(Annotate(dst.annotations))
                if isinstance(src, Categorical):
                    if src.categories != dst.categories:
                        assert len(src.categories) == len(dst.categories)
                        trs.append(CategoriesMapping(
                            list(zip(src.categories, dst.categories))))
                store.append((deconstruct(src), [deconstruct(tr) for tr in trs]))
            context.values["_domain_change_store"] = (dict(store), -2)


def report_transform(var, trs):
    # type: (Variable, List[Transform]) -> str
    """
    Return a html fragment summarizing the changes applied by `trs` list.

    Parameters
    ----------
    var : Variable
        A variable descriptor no which trs operates
    trs : List[Transform]
        A non empty list of `Transform` instances.

    Returns
    -------
    report : str
    """
    # pylint: disable=too-many-branches
    def strike(text):
        return "<s>{}</s>".format(escape(text))

    def i(text):
        return "<i>{}</i>".format(escape(text))

    def text(text):
        return "<span>{}</span>".format(escape(text))
    assert trs
    rename = annotate = catmap = None

    for tr in trs:
        if isinstance(tr, Rename):
            rename = tr
        elif isinstance(tr, Annotate):
            annotate = tr
        elif isinstance(tr, CategoriesMapping):
            catmap = tr
    if rename is not None:
        header = "{} → {}".format(var.name, rename.name)
    else:
        header = var.name
    values_section = None
    if catmap is not None:
        values_section = ("Values", [])
        lines = values_section[1]
        for ci, cj in catmap.mapping:
            if ci is None:
                item = cj + ("&nbsp;" * 3) + "(added)"
            elif cj is None:
                item = strike(ci)
            else:
                item = ci + " → " + cj
            lines.append(item)

    annotate_section = None
    if annotate is not None:
        annotate_section = ("Labels", [])
        lines = annotate_section[1]
        old = dict(var.annotations)
        new = dict(annotate.annotations)
        for name in sorted(set(old) - set(new)):
            lines.append(
                "<s>" + i(name) + " : " + text(old[name]) + "</s>"
            )
        for name in sorted(set(new) - set(old)):
            lines.append(
                i(name) + " : " + text(new[name]) + "&nbsp;" * 3 + i("(new)")
            )

        for name in sorted(set(new) & set(old)):
            if new[name] != old[name]:
                lines.append(
                    i(name) + " : " + text(old[name]) + " → " + text(new[name])
                )

    html = ["<div style='font-weight: bold;'>{}</div>".format(header)]
    for title, contents in filter(None, [values_section, annotate_section]):
        section_header = "<div>{}:</div>".format(title)
        section_contents = "<br/>\n".join(contents)
        html.append(section_header)
        html.append(
            "<div style='padding-left: 1em;'>" +
            section_contents +
            "</div>"
        )
    return "\n".join(html)


def abstract(var):
    # type: (Orange.data.Variable) -> Variable
    """
    Return `Varaible` descriptor for an `Orange.data.Variable` instance.

    Parameters
    ----------
    var : Orange.data.Variable

    Returns
    -------
    var : Variable
    """
    annotations = tuple(sorted(
        (key, str(value))
        for key, value in var.attributes.items()
    ))
    if isinstance(var, Orange.data.DiscreteVariable):
        values, base = var.values, var.base_value
        base = values[base] if base >= 0 else None
        return Categorical(var.name, tuple(values), base, annotations)
    elif isinstance(var, Orange.data.TimeVariable):
        return Time(var.name, annotations)
    elif isinstance(var, Orange.data.ContinuousVariable):
        return Real(var.name, (var.number_of_decimals, 'f'), annotations)
    elif isinstance(var, Orange.data.StringVariable):
        return String(var.name, annotations)
    else:
        raise TypeError


def _parse_attributes(mapping):
    # type: (Iterable[Tuple[str, str]]) -> Dict[str, Any]
    # Use the same functionality that parses attributes
    # when reading text files
    return Orange.data.Flags([
        "{}={}".format(*item) for item in mapping
    ]).attributes


@singledispatch
def apply_transform(var, trs):
    # type: (Orange.data.Variable, List[Transform]) -> Orange.data.Variable
    """
    Apply a list of `Transform` instances on an `Orange.data.Variable`.
    """
    raise NotImplementedError  # pragma: no cover


@apply_transform.register(Orange.data.DiscreteVariable)
def apply_transform_discete(var, trs):
    # type: (Orange.data.DiscreteVariable, ...) -> ...
    name, annotations = var.name, var.attributes
    base_value = var.base_value
    mapping = None
    for tr in trs:
        if isinstance(tr, Rename):
            name = tr.name
        elif isinstance(tr, CategoriesMapping):
            mapping = tr.mapping
        elif isinstance(tr, Annotate):
            annotations = _parse_attributes(tr.annotations)

    source_values = var.values
    if mapping is not None:
        dest_values = [cj for ci, cj in mapping if cj is not None]
    else:
        dest_values = var.values

    def positions(values):
        rval = {c: i for i, c in enumerate(values)}
        assert len(rval) == len(values)
        return rval
    source_codes = positions(source_values)
    dest_codes = positions(dest_values)
    if mapping is not None:
        # construct a lookup table
        lookup = np.full(len(source_values), np.nan, dtype=np.float)
        for ci, cj in mapping:
            if ci is not None and cj is not None:
                i, j = source_codes[ci], dest_codes[cj]
                lookup[i] = j

        if base_value != -1:
            base_value = lookup[base_value]
            if np.isnan(base_value):
                base_value = -1
        lookup = Lookup(var, lookup)
    else:
        lookup = Identity(var)
    variable = Orange.data.DiscreteVariable(
        name, values=dest_values, base_value=base_value, compute_value=lookup
    )
    variable.attributes.update(annotations)
    return variable


@apply_transform.register(Orange.data.ContinuousVariable)
def apply_transform_continuous(var, trs):
    # type: (Orange.data.ContinuousVariable, ...) -> ...
    name, annotations = var.name, var.attributes
    for tr in trs:
        if isinstance(tr, Rename):
            name = tr.name
        elif isinstance(tr, Annotate):
            annotations = _parse_attributes(tr.annotations)
    variable = Orange.data.ContinuousVariable(
        name=name, number_of_decimals=var.number_of_decimals,
        compute_value=Identity(var)
    )
    variable.attributes.update(annotations)
    return variable


@apply_transform.register(Orange.data.TimeVariable)
def apply_transform_time(var, trs):
    # type: (Orange.data.TimeVariable, ...) -> ...
    name, annotations = var.name, var.attributes
    for tr in trs:
        if isinstance(tr, Rename):
            name = tr.name
        elif isinstance(tr, Annotate):
            annotations = _parse_attributes(tr.annotations)
    variable = Orange.data.TimeVariable(
        name=name, have_date=var.have_date, have_time=var.have_time,
        compute_value=Identity(var)
    )
    variable.attributes.update(annotations)
    return variable


@apply_transform.register(Orange.data.StringVariable)
def apply_transform_string(var, trs):
    # type: (Orange.data.StringVariable, ...) -> ...
    name, annotations = var.name, var.attributes
    for tr in trs:
        if isinstance(tr, Rename):
            name = tr.name
        elif isinstance(tr, Annotate):
            annotations = _parse_attributes(tr.annotations)
    variable = Orange.data.StringVariable(
        name=name, compute_value=Identity(var)
    )
    variable.attributes.update(annotations)
    return variable


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWEditDomain).run(Orange.data.Table("iris"))
