from itertools import chain
from PyQt4 import QtGui, Qt
from Orange.widgets import widget, gui
from Orange.widgets.settings import *
from Orange.widgets.utils import vartype
from Orange.data.table import Table
from Orange.data import DiscreteVariable, ContinuousVariable, StringVariable
import Orange.data.filter as data_filter


class OWSelectData(widget.OWWidget):
    name = "Select Data"
    id = "Orange.widgets.data.file"
    description = "Selection of data based on values of variables"
    icon = "icons/SelectData.svg"
    priority = 100
    category = "Data"
    author = "Peter Juvan, Janez Demšar"
    author_email = "janez.demsar(@at@)fri.uni-lj.si"
    inputs = [("Data", Table, "set_data")]
    outputs = [("Matching Data", Table), ("Unmatched Data", Table)]

    want_main_area = False

    settingsHandler = DomainContextHandler(
        match_values=DomainContextHandler.MATCH_VALUES_ALL)
    conditions = ContextSetting([])
    update_on_change = Setting(True)
    purge_attributes = Setting(True)
    purge_classes = Setting(True)

    operator_names = {
        ContinuousVariable: ["equals", "is not",
                             "is below", "is at most",
                             "is greater than", "is at least",
                             "is between", "is outside",
                             "is defined"],
        DiscreteVariable: ["is", "is not", "is one of", "is not one of",
                           "is defined"],
        StringVariable: ["equals", "is not",
                         "is before", "is equal or before",
                         "is after", "is equal or after",
                         "is between", "is outside",
                         "begins with", "ends with",
                         "is defined"]}

    def __init__(self):
        super().__init__()

        self.old_purge_classes = True

        self.conditions = []
        self.last_output_conditions = None

        box = gui.widgetBox(self.controlArea, 'Conditions', stretch=100)
        self.cond_list = QtGui.QTableWidget(box)
        box.layout().addWidget(self.cond_list)
        self.cond_list.setShowGrid(False)
        self.cond_list.setSelectionMode(QtGui.QTableWidget.NoSelection)
        self.cond_list.setColumnCount(3)
        self.cond_list.setRowCount(0)
        self.cond_list.verticalHeader().hide()
        self.cond_list.horizontalHeader().hide()
        self.cond_list.resizeColumnToContents(0)
        self.cond_list.horizontalHeader().setResizeMode(
            QtGui.QHeaderView.Stretch)
        self.cond_list.viewport().setBackgroundRole(QtGui.QPalette.Window)

        box2 = gui.widgetBox(box, orientation="horizontal")
        self.add_button = gui.button(box2, self, "Add condition",
                                     callback=self.add_row)
        self.add_all_button = gui.button(box2, self, "Add all variables",
                                         callback=self.add_all)
        self.remove_all_button = gui.button(box2, self, "Remove all",
                                            callback=self.remove_all)
        gui.rubber(box2)

        info = gui.widgetBox(self.controlArea, '', orientation="horizontal")
        box_data_in = gui.widgetBox(info, 'Data In')
#        self.data_in_rows = gui.widgetLabel(box_data_in, " ")
        self.data_in_variables = gui.widgetLabel(box_data_in, " ")
        gui.rubber(box_data_in)

        box_data_out = gui.widgetBox(info, 'Data Out')
        self.data_out_rows = gui.widgetLabel(box_data_out, " ")
#        self.dataOutAttributesLabel = gui.widgetLabel(box_data_out, " ")
        gui.rubber(box_data_out)

        box = gui.widgetBox(self.controlArea, orientation="horizontal")
        boxSettings = gui.widgetBox(box, 'Purging')
        cb = gui.checkBox(boxSettings, self, "purge_attributes",
                          "Remove unused values/attributes",
                          callback=self.on_purge_change)
        self.purgeClassesCB = gui.checkBox(
            gui.indentedBox(boxSettings, sep=gui.checkButtonOffsetHint(cb)),
            self, "purge_classes", "Remove unused classes",
            callback=self.on_purge_change)
        boxCommit = gui.widgetBox(box, 'Commit')
        gui.checkBox(boxCommit, self, "update_on_change", "Commit on change")
        gui.button(boxCommit, self, "Commit", self.output_data, default=True)

        self.set_data(None)
        self.resize(600, 400)


    def add_row(self, attr=None, condition_type=None, condition_value=None):
        model = self.cond_list.model()
        row = model.rowCount()
        model.insertRow(row)

        attr_combo = QtGui.QComboBox()
        attr_combo.row = row
        for var in chain(self.data.domain.variables, self.data.domain.metas):
            attr_combo.addItem(*gui.attributeItem(var))
        attr_combo.setCurrentIndex(attr or 0)
        self.cond_list.setCellWidget(row, 0, attr_combo)

        self.remove_all_button.setDisabled(False)
        self.set_new_operators(attr_combo, attr is not None,
                               condition_type, condition_value)
        attr_combo.currentIndexChanged.connect(
            lambda _: self.set_new_operators(attr_combo, False))

        self.cond_list.resizeRowToContents(row)

    def add_all(self):
        if self.cond_list.rowCount():
            Mb = QtGui.QMessageBox
            if Mb.question(
                    self, "Remove existing filters",
                    "This will replace the existing filters with "
                    "filters for all variables.", Mb.Ok | Mb.Cancel) != Mb.Ok:
                return
            self.remove_all()
        domain = self.data.domain
        for i in range(len(domain.variables) + len(domain.metas)):
            self.add_row(i)


    def remove_all(self):
        self.remove_all_rows()
        self.conditions_changed()


    def remove_all_rows(self):
        self.cond_list.clear()
        self.cond_list.setRowCount(0)
        self.remove_all_button.setDisabled(True)


    def set_new_operators(self, attr_combo, adding_all,
                          selected_index=None, selected_values=None):
        oper_combo = QtGui.QComboBox()
        oper_combo.row = attr_combo.row
        oper_combo.attr_combo = attr_combo
        var = self.data.domain[attr_combo.currentText()]
        oper_combo.addItems(self.operator_names[type(var)])
        oper_combo.setCurrentIndex(selected_index or 0)
        self.set_new_values(oper_combo, adding_all, selected_values)
        self.cond_list.setCellWidget(oper_combo.row, 1, oper_combo)
        oper_combo.currentIndexChanged.connect(
            lambda _: self.set_new_values(oper_combo, False))

    @staticmethod
    def _get_lineedit_contents(box):
        return [child.text() for child in getattr(box, "controls", [box])
                if isinstance(child, QtGui.QLineEdit)]

    @staticmethod
    def _get_value_contents(box):
        cont = []
        for child in getattr(box, "controls", [box]):
            if isinstance(child, QtGui.QLineEdit):
                cont.append(child.text())
            elif isinstance(child, QtGui.QComboBox):
                cont.append(child.currentIndex())
        return tuple(cont)

    class QDoubleValidatorEmpty(QtGui.QDoubleValidator):
        def validate(self, input_, pos):
            if not input_:
                return (QtGui.QDoubleValidator.Acceptable, input_, pos)
            else:
                return super().validate(input_, pos)

    def set_new_values(self, oper_combo, adding_all, selected_values=None):
        def remove_children():
            for child in box.children()[1:]:
                box.layout().removeWidget(child)
                child.setParent(None)

        def add_textual(contents):
            le = gui.lineEdit(box, self, None)
            if contents:
                le.setText(contents)
            le.setMaximumWidth(60)
            le.setAlignment(Qt.Qt.AlignRight)
            le.editingFinished.connect(self.conditions_changed)
            return le

        def add_numeric(contents):
            le = add_textual(contents)
            le.setValidator(OWSelectData.QDoubleValidatorEmpty())
            return le

        var = self.data.domain[oper_combo.attr_combo.currentText()]
        box = self.cond_list.cellWidget(oper_combo.row, 2)
        if selected_values is not None:
            lc = list(selected_values) + ["", ""]
            lc = [str(x) for x in lc[:2]]
        else:
            lc = ["", ""]
        if box and vartype(var) == box.var_type:
            lc = self._get_lineedit_contents(box) + lc
        oper = oper_combo.currentIndex()

        if oper == oper_combo.count() - 1:
            self.cond_list.removeCellWidget(oper_combo.row, 2)
        elif isinstance(var, DiscreteVariable):
            combo = QtGui.QComboBox()
            combo.addItems([""] + var.values)
            if lc[0]:
                combo.setCurrentIndex(int(lc[0]))
            else:
                combo.setCurrentIndex(0)
            combo.var_type = vartype(var)
            self.cond_list.setCellWidget(oper_combo.row, 2, combo)
            combo.currentIndexChanged.connect(self.conditions_changed)
        else:
            box = gui.widgetBox(self, orientation="horizontal",
                                addToLayout=False)
            box.var_type = vartype(var)
            self.cond_list.setCellWidget(oper_combo.row, 2, box)
            if isinstance(var, ContinuousVariable):
                box.controls = [add_numeric(lc[0])]
                if oper > 5:
                    gui.widgetLabel(box, " and ")
                    box.controls.append(add_numeric(lc[1]))
                gui.rubber(box)
            elif isinstance(var, StringVariable):
                box.controls = [add_textual(lc[0])]
                if oper in [6, 7]:
                    gui.widgetLabel(box, " and ")
                    box.controls.append(add_textual(lc[1]))
            else:
                box.controls = []
        if not adding_all:
            self.conditions_changed()


    def set_data(self, data):
        self.closeContext()
        self.data = data
        self.remove_all_rows()
        self.add_button.setDisabled(data is None)
        domain = data and data.domain
        self.add_all_button.setDisabled(
            data is None or
            len(domain.variables) + len(domain.metas) > 100)
        if not data:
            return
        self.openContext(data)
        if not self.conditions and len(domain.variables):
            self.add_row()
        self.update_info(data, self.data_in_variables)
        for attr, cond_type, cond_value in self.conditions:
            attrs = [a.name for a in domain.variables + domain.metas]
            if attr in attrs:
                self.add_row(attrs.index(attr), cond_type, cond_value)


    def on_purge_change(self):
        if self.purge_attributes:
            if not self.purgeClassesCB.isEnabled():
                self.purgeClassesCB.setEnabled(True)
                self.purge_classes = self.old_purge_classes
        else:
            if self.purgeClassesCB.isEnabled():
                self.purgeClassesCB.setEnabled(False)
                self.old_purge_classes = self.purge_classes
                self.purge_classes = False
        self.conditions_changed()

    def conditions_changed(self):
        try:
            self.conditions = []
            self.conditions = [
                (self.cond_list.cellWidget(row, 0).currentText(),
                 self.cond_list.cellWidget(row, 1).currentIndex(),
                 self._get_value_contents(self.cond_list.cellWidget(row, 2)))
                for row in range(self.cond_list.rowCount())]
            if self.update_on_change and (
                    self.last_output_conditions is None or
                    self.last_output_conditions != self.conditions):
                self.output_data()
        except AttributeError:
            # Attribute error appears if the signal is triggered when the
            # controls are being constructed
            pass


    def output_data(self):
        matching_output = self.data
        non_matching_output = None
        if self.data:
            domain = self.data.domain
            filters = data_filter.Values()
            for attr_name, oper, values in self.conditions:
                attr_index = domain.index(attr_name)
                attr = domain[attr_index]
                if isinstance(attr, ContinuousVariable):
                    if any(not v for v in values):
                        continue
                    filter = data_filter.FilterContinuous(
                        attr_index, oper, *[float(v) for v in values])
                elif isinstance(attr, StringVariable):
                    if any(v for v in values):
                        continue
                    filter = data_filter.FilterString(
                        attr_index, oper, *[str(v) for v in values])
                else:
                    if oper in [2, 3]:
                        raise NotImplementedError(
                            "subset filters for discrete attributes are not "
                            "implemented yet")
                    elif oper == 4:
                        f_values = None
                    else:
                        if not values or not values[0]:
                            continue
                        if oper == 0:
                            f_values = {values[0] - 1}
                        else:
                            f_values = set(range(len(attr.values)))
                            f_values.remove(values[0] - 1)
                    filter = data_filter.FilterDiscrete(attr_index, f_values)
                filters.conditions.append(filter)

            matching_output = filters(self.data)
            filters.negate = True
            non_matching_output = filters(self.data)

            if hasattr(self.data, "name"):
                matching_output.name = self.data.name
                non_matching_output.name = self.data.name

            """
            if self.purge_attributes or self.purge_classes:
                remover = orange.RemoveUnusedValues(removeOneValued=True)

                newDomain = remover(matching_output, 0, True, self.purge_classes)
                if newDomain != matching_output.domain:
                    matching_output = orange.ExampleTable(newDomain, matching_output)

                newDomain = remover(non_matching_output, 0, True, self.purge_classes)
                if newDomain != non_matching_output.domain:
                    nonmatchingOutput = orange.ExampleTable(newDomain, non_matching_output)
            """
        self.send("Matching Data", matching_output)
        self.send("Unmatched Data", non_matching_output)

#        self.update_info(matching_output,
#                         self.data_out_variables, self.data_out_rows)


    def update_info(self, data, lab1):
        def sp(s, capitalize=True):
            return s and s or ("No" if capitalize else "no"), "s" * (s != 1)

        if not data:
            lab1.setText("")
        else:
            lab1.setText("%s row%s, %s variable%s" % (sp(len(data)) +
            sp(len(data.domain.variables) + len(data.domain.metas))))



    def sendReport(self):
        self.reportSettings("Output", [("Remove unused values/attributes", self.purge_attributes),
                                       ("Remove unused classes", self.purge_classes)])
        text = "<table>\n<th>Attribute</th><th>Condition</th><th>Value</th>/n"
        for i, cond in enumerate(self.conditions):
            if cond.type == "OR":
                text += "<tr><td span=3>\"OR\"</td></tr>\n"
            else:
                text += "<tr><td>%s</td><td>%s</td><td>%s</td></tr>\n" % (cond.varName, repr(cond.operator), cond.val1)

        text += "</table>"
        import OWReport
        self.reportSection("Conditions")
        self.reportRaw(OWReport.reportTable(self.cond_list))
#        self.reportTable("Conditions", self.criteriaTable)


def test():
    app = QtGui.QApplication([])
    w = OWSelectData()
    w.set_data(Table("iris"))
    w.show()
    app.exec_()

if __name__ == "__main__":
    test()
