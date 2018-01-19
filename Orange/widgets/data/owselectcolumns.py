from functools import partial

from AnyQt.QtWidgets import QWidget, QGridLayout
from AnyQt.QtCore import Qt, QSortFilterProxyModel, QItemSelection, QItemSelectionModel

from Orange.util import deprecated
from Orange.widgets import gui, widget
from Orange.widgets.data.contexthandlers import \
    SelectAttributesDomainContextHandler
from Orange.widgets.settings import ContextSetting, Setting
from Orange.widgets.utils.listfilter import VariablesListItemView, slices, variables_filter
from Orange.widgets.widget import Input, Output
from Orange.data.table import Table
from Orange.widgets.utils import vartype
from Orange.widgets.utils.itemmodels import VariableListModel
import Orange


def source_model(view):
    """ Return the source model for the Qt Item View if it uses
    the QSortFilterProxyModel.
    """
    if isinstance(view.model(), QSortFilterProxyModel):
        return view.model().sourceModel()
    else:
        return view.model()


def source_indexes(indexes, view):
    """ Map model indexes through a views QSortFilterProxyModel
    """
    model = view.model()
    if isinstance(model, QSortFilterProxyModel):
        return list(map(model.mapToSource, indexes))
    else:
        return indexes


# owloadcorpus in orange3-text used this
@deprecated('Orange.widgets.utils.itemmodels.VariableListModel')
def VariablesListItemModel(*args, **kwargs):
    return VariableListModel(*args, enable_dnd=True, **kwargs)


class ClassVarListItemModel(VariableListModel):
    def dropMimeData(self, mime, action, row, column, parent):
        """ Ensure only one variable can be dropped onto the view.
        """
        vars = mime.property('_items')
        if vars is None:
            return False
        if action == Qt.IgnoreAction:
            return True
        return VariableListModel.dropMimeData(
            self, mime, action, row, column, parent)


class ClassVariableItemView(VariablesListItemView):
    def __init__(self, parent=None, acceptedType=Orange.data.Variable):
        VariablesListItemView.__init__(self, parent, acceptedType)
        self.setDropIndicatorShown(False)

    def acceptsDropEvent(self, event):
        """
        Reimplemented

        Ensure only one variable is in the model.
        """
        accepts = super().acceptsDropEvent(event)
        mime = event.mimeData()
        vars = mime.property('_items')
        if vars is None:
            return False

        return accepts


class OWSelectAttributes(widget.OWWidget):
    name = "Select Columns"
    description = "Select columns from the data table and assign them to " \
                  "data features, classes or meta variables."
    icon = "icons/SelectColumns.svg"
    priority = 100

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)
        features = Output("Features", widget.AttributeList, dynamic=False)

    want_main_area = False
    want_control_area = True

    settingsHandler = SelectAttributesDomainContextHandler()
    domain_role_hints = ContextSetting({})
    auto_commit = Setting(True)

    def __init__(self):
        super().__init__()
        self.controlArea = QWidget(self.controlArea)
        self.layout().addWidget(self.controlArea)
        layout = QGridLayout()
        self.controlArea.setLayout(layout)
        layout.setContentsMargins(4, 4, 4, 4)
        box = gui.vBox(self.controlArea, "Available Variables",
                       addToLayout=False)

        self.available_attrs = VariableListModel(enable_dnd=True)
        filter_edit, self.available_attrs_view = variables_filter(
            parent=self, model=self.available_attrs)
        box.layout().addWidget(filter_edit)

        def dropcompleted(action):
            if action == Qt.MoveAction:
                self.commit()

        self.available_attrs_view.selectionModel().selectionChanged.connect(
            partial(self.update_interface_state, self.available_attrs_view))
        self.available_attrs_view.selectionModel().selectionChanged.connect(
            partial(self.update_interface_state, self.available_attrs_view))
        self.available_attrs_view.dragDropActionDidComplete.connect(dropcompleted)

        box.layout().addWidget(self.available_attrs_view)
        layout.addWidget(box, 0, 0, 3, 1)

        box = gui.vBox(self.controlArea, "Features", addToLayout=False)
        self.used_attrs = VariableListModel(enable_dnd=True)
        self.used_attrs_view = VariablesListItemView(
            acceptedType=(Orange.data.DiscreteVariable,
                          Orange.data.ContinuousVariable))

        self.used_attrs_view.setModel(self.used_attrs)
        self.used_attrs_view.selectionModel().selectionChanged.connect(
            partial(self.update_interface_state, self.used_attrs_view))
        self.used_attrs_view.dragDropActionDidComplete.connect(dropcompleted)
        box.layout().addWidget(self.used_attrs_view)
        layout.addWidget(box, 0, 2, 1, 1)

        box = gui.vBox(self.controlArea, "Target Variable", addToLayout=False)
        self.class_attrs = ClassVarListItemModel(enable_dnd=True)
        self.class_attrs_view = ClassVariableItemView(
            acceptedType=(Orange.data.DiscreteVariable,
                          Orange.data.ContinuousVariable))
        self.class_attrs_view.setModel(self.class_attrs)
        self.class_attrs_view.selectionModel().selectionChanged.connect(
            partial(self.update_interface_state, self.class_attrs_view))
        self.class_attrs_view.dragDropActionDidComplete.connect(dropcompleted)
        self.class_attrs_view.setMaximumHeight(72)
        box.layout().addWidget(self.class_attrs_view)
        layout.addWidget(box, 1, 2, 1, 1)

        box = gui.vBox(self.controlArea, "Meta Attributes", addToLayout=False)
        self.meta_attrs = VariableListModel(enable_dnd=True)
        self.meta_attrs_view = VariablesListItemView(
            acceptedType=Orange.data.Variable)
        self.meta_attrs_view.setModel(self.meta_attrs)
        self.meta_attrs_view.selectionModel().selectionChanged.connect(
            partial(self.update_interface_state, self.meta_attrs_view))
        self.meta_attrs_view.dragDropActionDidComplete.connect(dropcompleted)
        box.layout().addWidget(self.meta_attrs_view)
        layout.addWidget(box, 2, 2, 1, 1)

        bbox = gui.vBox(self.controlArea, addToLayout=False, margin=0)
        layout.addWidget(bbox, 0, 1, 1, 1)

        self.up_attr_button = gui.button(bbox, self, "Up",
                                         callback=partial(self.move_up, self.used_attrs_view))
        self.move_attr_button = gui.button(bbox, self, ">",
                                           callback=partial(self.move_selected,
                                                            self.used_attrs_view)
                                          )
        self.down_attr_button = gui.button(bbox, self, "Down",
                                           callback=partial(self.move_down, self.used_attrs_view))

        bbox = gui.vBox(self.controlArea, addToLayout=False, margin=0)
        layout.addWidget(bbox, 1, 1, 1, 1)

        self.up_class_button = gui.button(bbox, self, "Up",
                                          callback=partial(self.move_up, self.class_attrs_view))
        self.move_class_button = gui.button(bbox, self, ">",
                                            callback=partial(self.move_selected,
                                                             self.class_attrs_view,
                                                             exclusive=False)
                                           )
        self.down_class_button = gui.button(bbox, self, "Down",
                                            callback=partial(self.move_down, self.class_attrs_view))

        bbox = gui.vBox(self.controlArea, addToLayout=False, margin=0)
        layout.addWidget(bbox, 2, 1, 1, 1)
        self.up_meta_button = gui.button(bbox, self, "Up",
                                         callback=partial(self.move_up, self.meta_attrs_view))
        self.move_meta_button = gui.button(bbox, self, ">",
                                           callback=partial(self.move_selected,
                                                            self.meta_attrs_view)
                                          )
        self.down_meta_button = gui.button(bbox, self, "Down",
                                           callback=partial(self.move_down, self.meta_attrs_view))

        autobox = gui.auto_commit(None, self, "auto_commit", "Send")
        layout.addWidget(autobox, 3, 0, 1, 3)
        reset = gui.button(None, self, "Reset", callback=self.reset, width=120)
        autobox.layout().insertWidget(0, reset)
        autobox.layout().insertStretch(1, 20)

        layout.setRowStretch(0, 4)
        layout.setRowStretch(1, 0)
        layout.setRowStretch(2, 2)
        layout.setHorizontalSpacing(0)
        self.controlArea.setLayout(layout)

        self.data = None
        self.output_data = None
        self.original_completer_items = []

        self.resize(500, 600)

    @Inputs.data
    def set_data(self, data=None):
        self.update_domain_role_hints()
        self.closeContext()
        self.data = data
        if data is not None:
            self.openContext(data)
            all_vars = data.domain.variables + data.domain.metas

            var_sig = lambda attr: (attr.name, vartype(attr))

            domain_hints = {var_sig(attr): ("attribute", i)
                            for i, attr in enumerate(data.domain.attributes)}

            domain_hints.update({var_sig(attr): ("meta", i)
                                 for i, attr in enumerate(data.domain.metas)})

            if data.domain.class_vars:
                domain_hints.update(
                    {var_sig(attr): ("class", i)
                     for i, attr in enumerate(data.domain.class_vars)})

            # update the hints from context settings
            domain_hints.update(self.domain_role_hints)

            attrs_for_role = lambda role: [
                (domain_hints[var_sig(attr)][1], attr)
                for attr in all_vars if domain_hints[var_sig(attr)][0] == role]

            attributes = [
                attr for place, attr in sorted(attrs_for_role("attribute"),
                                               key=lambda a: a[0])]
            classes = [
                attr for place, attr in sorted(attrs_for_role("class"),
                                               key=lambda a: a[0])]
            metas = [
                attr for place, attr in sorted(attrs_for_role("meta"),
                                               key=lambda a: a[0])]
            available = [
                attr for place, attr in sorted(attrs_for_role("available"),
                                               key=lambda a: a[0])]

            self.used_attrs[:] = attributes
            self.class_attrs[:] = classes
            self.meta_attrs[:] = metas
            self.available_attrs[:] = available
        else:
            self.used_attrs[:] = []
            self.class_attrs[:] = []
            self.meta_attrs[:] = []
            self.available_attrs[:] = []

        self.unconditional_commit()

    def update_domain_role_hints(self):
        """ Update the domain hints to be stored in the widgets settings.
        """
        hints_from_model = lambda role, model: [
            ((attr.name, vartype(attr)), (role, i))
            for i, attr in enumerate(model)]
        hints = dict(hints_from_model("available", self.available_attrs))
        hints.update(hints_from_model("attribute", self.used_attrs))
        hints.update(hints_from_model("class", self.class_attrs))
        hints.update(hints_from_model("meta", self.meta_attrs))
        self.domain_role_hints = hints

    def selected_rows(self, view):
        """ Return the selected rows in the view.
        """
        rows = view.selectionModel().selectedRows()
        model = view.model()
        if isinstance(model, QSortFilterProxyModel):
            rows = [model.mapToSource(r) for r in rows]
        return [r.row() for r in rows]

    def move_rows(self, view, rows, offset):
        model = view.model()
        newrows = [min(max(0, row + offset), len(model) - 1) for row in rows]

        for row, newrow in sorted(zip(rows, newrows), reverse=offset > 0):
            model[row], model[newrow] = model[newrow], model[row]

        selection = QItemSelection()
        for nrow in newrows:
            index = model.index(nrow, 0)
            selection.select(index, index)
        view.selectionModel().select(
            selection, QItemSelectionModel.ClearAndSelect)

        self.commit()

    def move_up(self, view):
        selected = self.selected_rows(view)
        self.move_rows(view, selected, -1)

    def move_down(self, view):
        selected = self.selected_rows(view)
        self.move_rows(view, selected, 1)

    def move_selected(self, view, exclusive=False):
        if self.selected_rows(view):
            self.move_selected_from_to(view, self.available_attrs_view)
        elif self.selected_rows(self.available_attrs_view):
            self.move_selected_from_to(self.available_attrs_view, view,
                                       exclusive)

    def move_selected_from_to(self, src, dst, exclusive=False):
        self.move_from_to(src, dst, self.selected_rows(src), exclusive)

    def move_from_to(self, src, dst, rows, exclusive=False):
        src_model = source_model(src)
        attrs = [src_model[r] for r in rows]

        for s1, s2 in reversed(list(slices(rows))):
            del src_model[s1:s2]

        dst_model = source_model(dst)

        dst_model.extend(attrs)

        self.commit()

    def update_interface_state(self, focus=None, selected=None, deselected=None):
        for view in [self.available_attrs_view, self.used_attrs_view,
                     self.class_attrs_view, self.meta_attrs_view]:
            if view is not focus and not view.hasFocus() and self.selected_rows(view):
                view.selectionModel().clear()

        def selected_vars(view):
            model = source_model(view)
            return [model[i] for i in self.selected_rows(view)]

        available_selected = selected_vars(self.available_attrs_view)
        attrs_selected = selected_vars(self.used_attrs_view)
        class_selected = selected_vars(self.class_attrs_view)
        meta_selected = selected_vars(self.meta_attrs_view)

        available_types = set(map(type, available_selected))
        all_primitive = all(var.is_primitive()
                            for var in available_types)

        move_attr_enabled = (available_selected and all_primitive) or \
                            attrs_selected

        self.move_attr_button.setEnabled(bool(move_attr_enabled))
        if move_attr_enabled:
            self.move_attr_button.setText(">" if available_selected else "<")

        move_class_enabled = (all_primitive and available_selected) or class_selected

        self.move_class_button.setEnabled(bool(move_class_enabled))
        if move_class_enabled:
            self.move_class_button.setText(">" if available_selected else "<")
        move_meta_enabled = available_selected or meta_selected

        self.move_meta_button.setEnabled(bool(move_meta_enabled))
        if move_meta_enabled:
            self.move_meta_button.setText(">" if available_selected else "<")

    def commit(self):
        self.update_domain_role_hints()
        if self.data is not None:
            attributes = list(self.used_attrs)
            class_var = list(self.class_attrs)
            metas = list(self.meta_attrs)

            domain = Orange.data.Domain(attributes, class_var, metas)
            newdata = self.data.transform(domain)
            self.output_data = newdata
            self.Outputs.data.send(newdata)
            self.Outputs.features.send(widget.AttributeList(attributes))
        else:
            self.output_data = None
            self.Outputs.data.send(None)
            self.Outputs.features.send(None)

    def reset(self):
        if self.data is not None:
            self.available_attrs[:] = []
            self.used_attrs[:] = self.data.domain.attributes
            self.class_attrs[:] = self.data.domain.class_vars
            self.meta_attrs[:] = self.data.domain.metas
            self.update_domain_role_hints()
            self.commit()

    def send_report(self):
        if not self.data or not self.output_data:
            return
        in_domain, out_domain = self.data.domain, self.output_data.domain
        self.report_domain("Input data", self.data.domain)
        if (in_domain.attributes, in_domain.class_vars, in_domain.metas) == (
                out_domain.attributes, out_domain.class_vars, out_domain.metas):
            self.report_paragraph("Output data", "No changes.")
        else:
            self.report_domain("Output data", self.output_data.domain)
            diff = list(set(in_domain.variables + in_domain.metas) -
                        set(out_domain.variables + out_domain.metas))
            if diff:
                text = "%i (%s)" % (len(diff), ", ".join(x.name for x in diff))
                self.report_items((("Removed", text),))


if __name__ == "__main__":  # pragma: no cover
    OWSelectAttributes.test_run(Orange.data.Table("brown-selected"))
