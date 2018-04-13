"""
Edit Domain
-----------

A widget for manual editing of a domain's attributes.

"""
import unicodedata
from itertools import chain

from AnyQt.QtWidgets import (
    QWidget, QListView, QTreeView, QVBoxLayout, QHBoxLayout, QFormLayout,
    QToolButton, QLineEdit, QAction, QStackedWidget
)
from AnyQt.QtGui import QStandardItemModel, QStandardItem, QKeySequence
from AnyQt.QtCore import Qt, QSize
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

import Orange.data
import Orange.preprocess.transformation
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import Input, Output


def get_qualified(module, name):
    """Return a qualified module member ``name`` inside the named
    ``module``.

    The module (or package) first gets imported and the name
    is retrieved from the module's global namespace.

    """
    # see __import__.__doc__ for why 'fromlist' is used
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


def variable_description(var, skip_attributes=False):
    """Return a variable descriptor.

    A descriptor is a hashable tuple which should uniquely define
    the variable i.e. (module, type_name, variable_name,
    any_kwargs, sorted-attributes-items).

    """
    var_type = type(var)
    attributes = ()
    if not skip_attributes:
        attributes = tuple(sorted(var.attributes.items()))
    if var.is_discrete:
        return (var_type.__module__,
                var_type.__name__,
                var.name,
                (("values", tuple(var.values)),),
                attributes)
    else:
        return (var_type.__module__,
                var_type.__name__,
                var.name,
                (),
                attributes)


def variable_from_description(description, compute_value=None):
    """Construct a variable from its description (see
    :func:`variable_description`).

    """
    module, type_name, name, kwargs, attrs = description
    try:
        constructor = get_qualified(module, type_name)
    except (ImportError, AttributeError):
        raise ValueError("Invalid descriptor type '{}.{}"
                         "".format(module, type_name))

    var = constructor(name, compute_value=compute_value, **dict(list(kwargs)))
    var.attributes.update(attrs)
    return var


class DictItemsModel(QStandardItemModel):
    """A Qt Item Model class displaying the contents of a python
    dictionary.

    """
    # Implement a proper model with in-place editing.
    # (Maybe it should be a TableModel with 2 columns)
    def __init__(self, parent=None, dict={}):
        QStandardItemModel.__init__(self, parent)
        self.setHorizontalHeaderLabels(["Key", "Value"])
        self.set_dict(dict)

    def set_dict(self, dict):
        self._dict = dict
        self.clear()
        self.setHorizontalHeaderLabels(["Key", "Value"])
        for key, value in sorted(dict.items()):
            key_item = QStandardItem(str(key))
            value_item = QStandardItem(str(value))
            key_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            value_item.setFlags(value_item.flags() | Qt.ItemIsEditable)
            self.appendRow([key_item, value_item])

    def get_dict(self):
        # Use the same functionality that parses attributes
        # when reading text files
        return Orange.data.Flags([
            "{}={}".format(self.item(row, 0).text(),
                           self.item(row, 1).text())
            for row in range(self.rowCount())
        ]).attributes


class VariableEditor(QWidget):
    """An editor widget for a variable.

    Can edit the variable name, and its attributes dictionary.

    """
    variable_changed = Signal()

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.var = None
        self.setup_gui()

    def setup_gui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.main_form = QFormLayout()
        self.main_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.addLayout(self.main_form)

        self._setup_gui_name()
        self._setup_gui_labels()

    def _setup_gui_name(self):
        class OrangeLineEdit(QLineEdit):
            def keyPressEvent(self, event):
                if event.key() in [Qt.Key_Return, Qt.Key_Enter]:
                    self.parent().on_name_changed()
                else:
                    super().keyPressEvent(event)

        self.name_edit = OrangeLineEdit()
        self.main_form.addRow("Name:", self.name_edit)
        self.name_edit.editingFinished.connect(self.on_name_changed)

    def _setup_gui_labels(self):
        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(1)

        self.labels_edit = QTreeView()
        self.labels_edit.setEditTriggers(QTreeView.CurrentChanged)
        self.labels_edit.setRootIsDecorated(False)

        self.labels_model = DictItemsModel()
        self.labels_edit.setModel(self.labels_model)

        self.labels_edit.selectionModel().selectionChanged.connect(
            self.on_label_selection_changed)

        # Necessary signals to know when the labels change
        self.labels_model.dataChanged.connect(self.on_labels_changed)
        self.labels_model.rowsInserted.connect(self.on_labels_changed)
        self.labels_model.rowsRemoved.connect(self.on_labels_changed)

        vlayout.addWidget(self.labels_edit)
        hlayout = QHBoxLayout()
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.setSpacing(1)
        self.add_label_action = QAction(
            "+", self,
            toolTip="Add a new label.",
            triggered=self.on_add_label,
            enabled=False,
            shortcut=QKeySequence(QKeySequence.New))

        self.remove_label_action = QAction(
            unicodedata.lookup("MINUS SIGN"), self,
            toolTip="Remove selected label.",
            triggered=self.on_remove_label,
            enabled=False,
            shortcut=QKeySequence(QKeySequence.Delete))

        button_size = gui.toolButtonSizeHint()
        button_size = QSize(button_size, button_size)

        button = QToolButton(self)
        button.setFixedSize(button_size)
        button.setDefaultAction(self.add_label_action)
        hlayout.addWidget(button)

        button = QToolButton(self)
        button.setFixedSize(button_size)
        button.setDefaultAction(self.remove_label_action)
        hlayout.addWidget(button)
        hlayout.addStretch(10)
        vlayout.addLayout(hlayout)

        self.main_form.addRow("Labels:", vlayout)

    def set_data(self, var):
        """Set the variable to edit.
        """
        self.clear()
        self.var = var

        if var is not None:
            self.name_edit.setText(var.name)
            self.labels_model.set_dict(dict(var.attributes))
            self.add_label_action.setEnabled(True)
        else:
            self.add_label_action.setEnabled(False)
            self.remove_label_action.setEnabled(False)

    def get_data(self):
        """Retrieve the modified variable.
        """
        name = str(self.name_edit.text()).strip()
        labels = self.labels_model.get_dict()

        # Is the variable actually changed.
        if self.var is not None and not self.is_same():
            var = type(self.var)(name)
            var.attributes.update(labels)
            self.var = var
        else:
            var = self.var

        return var

    def is_legal(self):
        name = str(self.name_edit.text()).strip()
        return not len(name) == 0

    def is_same(self):
        """Is the current model state the same as the input.
        """
        name = str(self.name_edit.text()).strip()
        labels = self.labels_model.get_dict()
        return (self.var is not None and name == self.var.name and
                labels == self.var.attributes)

    def clear(self):
        """Clear the editor state.
        """
        self.var = None
        self.name_edit.setText("")
        self.labels_model.set_dict({})

    def maybe_commit(self):
        if not self.is_same() and self.is_legal():
            self.commit()

    def commit(self):
        """Emit a ``variable_changed()`` signal.
        """
        self.variable_changed.emit()

    @Slot()
    def on_name_changed(self):
        self.maybe_commit()

    @Slot()
    def on_labels_changed(self, *args):
        self.maybe_commit()

    @Slot()
    def on_add_label(self):
        self.labels_model.appendRow([QStandardItem(""), QStandardItem("")])
        row = self.labels_model.rowCount() - 1
        index = self.labels_model.index(row, 0)
        self.labels_edit.edit(index)

    @Slot()
    def on_remove_label(self):
        rows = self.labels_edit.selectionModel().selectedRows()
        if rows:
            row = rows[0]
            self.labels_model.removeRow(row.row())

    @Slot()
    def on_label_selection_changed(self):
        selected = self.labels_edit.selectionModel().selectedRows()
        self.remove_label_action.setEnabled(bool(len(selected)))


class DiscreteVariableEditor(VariableEditor):
    """An editor widget for editing a discrete variable.

    Extends the :class:`VariableEditor` to enable editing of
    variables values.

    """
    def setup_gui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.main_form = QFormLayout()
        self.main_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.addLayout(self.main_form)

        self._setup_gui_name()
        self._setup_gui_values()
        self._setup_gui_labels()

    def _setup_gui_values(self):
        self.values_edit = QListView()
        self.values_edit.setEditTriggers(QTreeView.CurrentChanged)
        self.values_model = itemmodels.PyListModel(flags=Qt.ItemIsSelectable | \
                                        Qt.ItemIsEnabled | Qt.ItemIsEditable)
        self.values_edit.setModel(self.values_model)

        self.values_model.dataChanged.connect(self.on_values_changed)
        self.main_form.addRow("Values:", self.values_edit)

    def set_data(self, var):
        """Set the variable to edit
        """
        VariableEditor.set_data(self, var)
        self.values_model[:] = list(var.values) if var is not None else []

    def get_data(self):
        """Retrieve the modified variable
        """
        name = str(self.name_edit.text()).strip()
        labels = self.labels_model.get_dict()
        values = map(str, self.values_model)

        if self.var is not None and not self.is_same():
            var = type(self.var)(name, values=values)
            var.attributes.update(labels)
            self.var = var
        else:
            var = self.var

        return var

    def is_same(self):
        """Is the current model state the same as the input.
        """
        values = list(map(str, self.values_model))
        return (VariableEditor.is_same(self) and self.var is not None and
                self.var.values == values)

    def clear(self):
        """Clear the model state.
        """
        VariableEditor.clear(self)
        self.values_model.clear()

    @Slot()
    def on_values_changed(self):
        self.maybe_commit()


class ContinuousVariableEditor(VariableEditor):
    # TODO: enable editing of number_of_decimals, scientific format ...
    pass


class OWEditDomain(widget.OWWidget):
    name = "Edit Domain"
    description = "Rename features and their values."
    icon = "icons/EditDomain.svg"
    priority = 3125

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        data = Output("Data", Orange.data.Table)

    settingsHandler = settings.DomainContextHandler()

    domain_change_hints = settings.ContextSetting({})
    selected_index = settings.ContextSetting({})

    autocommit = settings.Setting(True)

    def __init__(self):
        super().__init__()

        self.data = None
        self.input_vars = ()
        self._invalidated = False

        box = gui.vBox(self.controlArea, "Domain Features")

        self.domain_model = itemmodels.VariableListModel()
        self.domain_view = QListView(
            selectionMode=QListView.SingleSelection,
            uniformItemSizes=True
        )
        self.domain_view.setModel(self.domain_model)
        self.domain_view.selectionModel().selectionChanged.connect(
            self._on_selection_changed)
        box.layout().addWidget(self.domain_view)

        box = gui.hBox(self.controlArea)
        gui.button(box, self, "Reset Selected", callback=self.reset_selected)
        gui.button(box, self, "Reset All", callback=self.reset_all)

        gui.auto_commit(self.controlArea, self, "autocommit", "Apply")

        box = gui.vBox(self.mainArea, "Edit")
        self.editor_stack = QStackedWidget()

        self.editor_stack.addWidget(DiscreteVariableEditor())
        self.editor_stack.addWidget(ContinuousVariableEditor())
        self.editor_stack.addWidget(VariableEditor())

        box.layout().addWidget(self.editor_stack)

        self.Error.add_message(
            "duplicate_var_name",
            "A variable name is duplicated.")

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        """Set input dataset."""
        self.closeContext()
        self.clear()
        self.data = data

        if self.data is not None:
            self._initialize()
            self.openContext(self.data)
            self._restore()

        self.unconditional_commit()

    def clear(self):
        """Clear the widget state."""
        self.data = None
        self.domain_model[:] = []
        self.input_vars = []
        self.domain_change_hints = {}
        self.selected_index = -1

    def reset_selected(self):
        """Reset the currently selected variable to its original state."""
        ind = self.selected_var_index()
        if ind >= 0:
            var = self.input_vars[ind]
            desc = variable_description(var, skip_attributes=True)
            if desc in self.domain_change_hints:
                del self.domain_change_hints[desc]

            self.domain_model[ind] = var
            self.editor_stack.currentWidget().set_data(var)
            self._invalidate()

    def reset_all(self):
        """Reset all variables to their original state."""
        self.domain_change_hints = {}
        if self.data is not None:
            # To invalidate stored hints
            self.domain_model[:] = self.input_vars
            itemmodels.select_row(self.domain_view, self.selected_index)
            self._invalidate()

    def selected_var_index(self):
        """Return the selected row in 'Domain Features' view."""
        rows = self.domain_view.selectedIndexes()
        assert len(rows) <= 1
        return rows[0].row() if rows else -1

    def _initialize(self):
        domain = self.data.domain
        self.input_vars = domain.variables + domain.metas
        self.domain_model[:] = list(self.input_vars)

    def _restore(self):
        # Restore the variable states from saved settings.
        def transform(var):
            vdesc = variable_description(var, skip_attributes=True)
            if vdesc in self.domain_change_hints:
                return variable_from_description(
                    self.domain_change_hints[vdesc],
                    compute_value=Orange.preprocess.transformation.Identity(var))
            else:
                return var

        self.domain_model[:] = map(transform, self.input_vars)

        # Restore the variable selection if possible
        index = self.selected_index
        if index >= len(self.input_vars):
            index = 0 if len(self.input_vars) else -1
        if index >= 0:
            itemmodels.select_row(self.domain_view, index)

    def _on_selection_changed(self):
        self.selected_index = self.selected_var_index()
        self.open_editor(self.selected_index)

    def open_editor(self, index):
        self.clear_editor()
        if index < 0:
            return

        var = self.domain_model[index]

        editor_index = 2
        if var.is_discrete:
            editor_index = 0
        elif var.is_continuous:
            editor_index = 1
        editor = self.editor_stack.widget(editor_index)
        self.editor_stack.setCurrentWidget(editor)

        editor.set_data(var)
        editor.variable_changed.connect(self._on_variable_changed)

    def clear_editor(self):
        current = self.editor_stack.currentWidget()
        try:
            current.variable_changed.disconnect(self._on_variable_changed)
        except Exception:
            pass
        current.set_data(None)

    def _on_variable_changed(self):
        """User edited the current variable in editor."""
        assert 0 <= self.selected_index <= len(self.domain_model)
        editor = self.editor_stack.currentWidget()

        # Replace the variable in the 'Domain Features' view/model
        old_var = self.input_vars[self.selected_index]
        new_var = editor.get_data().copy(
            compute_value=Orange.preprocess.transformation.Identity(old_var)
        )
        self.domain_model[self.selected_index] = new_var


        # Store the transformation hint.
        old_var_desc = variable_description(old_var, skip_attributes=True)
        self.domain_change_hints[old_var_desc] = variable_description(new_var)

        self._invalidate()

    def _invalidate(self):
        self.commit()

    def commit(self):
        """Send the changed data to output."""
        new_data = None
        var_names = [vn.name for vn in self.domain_model]
        self.Error.duplicate_var_name.clear()
        if self.data is not None:
            if len(var_names) == len(set(var_names)):
                input_domain = self.data.domain
                n_attrs = len(input_domain.attributes)
                n_class_vars = len(input_domain.class_vars)
                all_new_vars = list(self.domain_model)
                attrs = all_new_vars[: n_attrs]
                class_vars = all_new_vars[n_attrs: n_attrs + n_class_vars]
                new_metas = all_new_vars[n_attrs + n_class_vars:]
                new_domain = Orange.data.Domain(attrs, class_vars, new_metas)
                new_data = self.data.transform(new_domain)
            else:
                self.Error.duplicate_var_name()

        self.Outputs.data.send(new_data)

    def sizeHint(self):
        sh = super().sizeHint()
        return sh.expandedTo(QSize(660, 550))

    def send_report(self):
        if self.data is not None:
            self.report_raw("", EditDomainReport(
                old_domain=chain(self.data.domain.variables, self.data.domain.metas),
                new_domain=self.domain_model).to_html())
        else:
            self.report_data(None)


class EditDomainReport:
    """Report creator for changes made in the OWEditDomain widget.

    Parameters
    ----------
    old_domain : list of Variable
        input variables
    new_domain : list of Variable
        variables with applied changes
    """

    VARIABLE_HTML = "<li style='font-weight: bold; padding-top:{};'>{}</li>".format
    INDENTED_ITEM = "<div style='padding-left: 1em'>{}</div>".format

    def __init__(self, old_domain, new_domain):
        self.old_domain = old_domain
        self.new_domain = new_domain

    def to_html(self):
        """Collect changes to variable names, values and labels
        and render them to html.
        """
        all_changes = []
        for old_var, new_var in zip(self.old_domain, self.new_domain):
            changes = list(chain.from_iterable([
                self._section("Values", self._value_changes(old_var, new_var)),
                self._section("Labels", self._label_changes(old_var, new_var))
            ]))

            padding_top = ".5em" if all_changes else "0"
            if old_var.name != new_var.name:
                all_changes.append(self.VARIABLE_HTML(
                    padding_top, "{} → {}".format(old_var.name, new_var.name)))
            elif changes:
                all_changes.append(self.VARIABLE_HTML(padding_top, old_var.name))
            all_changes.extend(changes)
        return "<ul>{}</ul>".format("".join(all_changes)) if all_changes else "No changes"

    def _section(self, name, changes):
        """Generator that adds section name if there were any changes made."""
        changes = list(changes)
        if changes:
            yield "<div>{}:</div>".format(name)
            yield from changes

    def _value_changes(self, old_variable, new_variable):
        """Generator of all renamed values"""
        if not old_variable.is_discrete:
            return
        for old_value, new_value in zip(old_variable.values, new_variable.values):
            if old_value != new_value:
                yield self.INDENTED_ITEM("{} → {}".format(old_value, new_value))

    def _label_changes(self, old_variable, new_variable):
        """Generator of all added, removed and modified labels"""
        old_labels = old_variable.attributes
        new_labels = new_variable.attributes

        for name, value in new_labels.items():
            if name not in old_labels:
                yield self.INDENTED_ITEM("<i>{}</i>: {}&nbsp;&nbsp;&nbsp;<i>(new)</i>"
                                         .format(name, value))

        for name, value in old_labels.items():
            if name not in new_labels:
                yield self.INDENTED_ITEM("<strike><i>{}</i>: {}</strike>"
                                         .format(name, value))

        for name in old_labels:
            if name in new_labels and new_labels[name] != old_labels[name]:
                yield self.INDENTED_ITEM("<i>{}</i>: {} → {}"
                                         .format(name, old_labels[name], new_labels[name]))


def main():
    from AnyQt.QtWidgets import QApplication
    app = QApplication([])
    w = OWEditDomain()
    data = Orange.data.Table("iris")
    w.set_data(data)
    w.show()
    w.raise_()

    return app.exec_()


if __name__ == "__main__":
    main()
