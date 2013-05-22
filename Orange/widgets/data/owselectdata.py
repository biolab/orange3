from itertools import chain
from PyQt4 import QtGui, Qt
from Orange.widgets import widget, gui
from Orange.widgets.settings import *
from Orange.data.table import Table
from Orange.data import DiscreteVariable, ContinuousVariable, StringVariable
import Orange.data.filter as data_filter


class OWSelectData(widget.OWWidget):
    _name = "Select Data"
    _id = "Orange.widgets.data.file"
    _description = "Selection of data based on values of variables"
    _icon = "icons/SelectData.svg"
    _priority = 100
    _category = "Data"
    _author = "Peter Juvan, Janez Demšar"
    _author_email = "janez.demsar(@at@)fri.uni-lj.si"
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

    def __init__(self, parent=None, signalManager=None):
        super().__init__(parent, signalManager)

        self.old_purge_classes = True

        self.conditions = []

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


    def add_row(self, attr=None):
        model = self.cond_list.model()
        row = model.rowCount()
        model.insertRow(row)

        attr_combo = QtGui.QComboBox()
        attr_combo.row = row
        attr_combo.setStyleSheet("border: none")
        for var in chain(self.data.domain.variables, self.data.domain.metas):
            attr_combo.addItem(*gui.attributeItem(var))
        attr_combo.setCurrentIndex(attr or 0)
        self.cond_list.setCellWidget(row, 0, attr_combo)

        self.remove_all_button.setDisabled(False)
        self.set_new_operators(attr_combo, attr is not None)
        attr_combo.currentIndexChanged.connect(
            lambda _: self.set_new_operators(attr_combo, False))
        self.conditions.append([])


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
        self.output_data_if()


    def remove_all_rows(self):
        self.cond_list.clear()
        self.cond_list.setRowCount(0)
        self.remove_all_button.setDisabled(True)


    def set_new_operators(self, attr_combo, adding_all):
        oper_combo = QtGui.QComboBox()
        oper_combo.row = attr_combo.row
        oper_combo.attr_combo = attr_combo
        oper_combo.setStyleSheet("border: none")
        var = self.data.domain[attr_combo.currentText()]
        oper_combo.addItems(self.operator_names[type(var)])
        oper_combo.setCurrentIndex(0)
        self.set_new_values(oper_combo, adding_all)
        self.cond_list.setCellWidget(oper_combo.row, 1, oper_combo)
        oper_combo.currentIndexChanged.connect(
            lambda _: self.set_new_values(oper_combo, False))

    @staticmethod
    def _get_lineedit_contents(box):
        return [child.text() for child in box.children()
                if isinstance(child, QtGui.QLineEdit)]


    def set_new_values(self, oper_combo, adding_all):
        def remove_children():
            for child in box.children()[1:]:
                box.layout().removeWidget(child)
                child.setParent(None)

        def add_textual(contents):
            le = gui.lineEdit(box, self, None)
            if contents:
                le.setText(contents)
            le.setMaximumWidth(60)
            le.setStyleSheet("""margin-left: 5px;
                                border: none;
                                border-bottom: 1px solid black;
                                """)
            le.setAlignment(Qt.Qt.AlignRight)
            le.editingFinished.connect(self.output_data_if)
            return le

        def add_numeric(contents):
            le = add_textual(contents)
            le.setValidator(QtGui.QDoubleValidator())
            return le

        var = self.data.domain[oper_combo.attr_combo.currentText()]
        box = self.cond_list.cellWidget(oper_combo.row, 2)
        lc = ["", ""]
        if box and var.var_type == box.var_type:
            lc = self._get_lineedit_contents(box) + lc
        oper = oper_combo.currentIndex()

        if oper == oper_combo.count() - 1:
            self.cond_list.removeCellWidget(oper_combo.row, 2)
        elif isinstance(var, DiscreteVariable):
            combo = QtGui.QComboBox()
            combo.addItems([""] + var.values)
            combo.setCurrentIndex(0)
            combo.setStyleSheet("border: none")
            combo.var_type = var.var_type
            self.cond_list.setCellWidget(oper_combo.row, 2, combo)
            combo.currentIndexChanged.connect(self.output_data_if)
        else:
            box = gui.widgetBox(self, orientation="horizontal",
                                addToLayout=False)
            box.var_type = var.var_type
            self.cond_list.setCellWidget(oper_combo.row, 2, box)
            if isinstance(var, ContinuousVariable):
                add_numeric(lc[0])
                if oper > 5:
                    gui.widgetLabel(box, " and ")
                    add_numeric(lc[1])
                gui.rubber(box)
            elif isinstance(var, StringVariable):
                add_textual(lc[0])
                if oper in [6, 7]:
                    gui.widgetLabel(box, " and ")
                    add_textual(lc[1])

        self.conditions[oper_combo.row] = [var.name, oper_combo.currentIndex(), ]
        if not adding_all:
            self.output_data()


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
        self.update_info(data, self.data_in_variables)
        self.output_data_if()


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
        self.output_data_if()


    def output_data_if(self):
        if self.update_on_change:
            self.output_data()


    def output_data(self):
        matching_output = self.data
        non_matching_output = None
        if self.data:
            filters = data_filter.Values()
            clist = self.cond_list
            for row in range(clist.rowCount()):
                var = clist.cellWidget(row, 0).currentText()
                var = self.data.domain[var]

            hasClass = self.data.domain.class_var is not None

            matching_output = filter(self.data, 1)
            matching_output.name = self.data.name
            non_matching_output = filter(self.data, 1, negate=1)
            non_matching_output.name = self.data.name

            if self.purge_attributes or self.purge_classes:
                remover = orange.RemoveUnusedValues(removeOneValued=True)

                newDomain = remover(matching_output, 0, True, self.purge_classes)
                if newDomain != matching_output.domain:
                    matching_output = orange.ExampleTable(newDomain, matching_output)

                newDomain = remover(non_matching_output, 0, True, self.purge_classes)
                if newDomain != non_matching_output.domain:
                    nonmatchingOutput = orange.ExampleTable(newDomain, non_matching_output)

        self.send("Matching Data", matching_output)
        self.send("Unmatched Data", non_matching_output)

        self.update_info(matching_output,
                         self.data_out_variables, self.data_out_rows)


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
