from functools import partial
from itertools import chain
from PyQt4 import QtCore
from PyQt4 import QtGui
from PyQt4.QtCore import Qt
from Orange.widgets import widget, gui, widget
from Orange.widgets.settings import *
from Orange.data.table import Table
from Orange.data import DiscreteVariable, ContinuousVariable, StringVariable


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

    update_on_change = Setting(True)
    purge_attributes = Setting(True)
    purge_classes = Setting(True)
    case_sensitive = Setting(False)

    settingsHandler = DomainContextHandler(match_values=DomainContextHandler.MATCH_VALUES_ALL)
    conditions = ContextSetting([])

    operator_names = {
        DiscreteVariable("equals", "")
    }
    operatorsD = staticmethod(["equals","in"])
    operatorsC = staticmethod(["=","<","<=",">",">=","between","outside"])
    operatorsS = staticmethod(["=","<","<=",">",">=","contains","begins with","ends with","between","outside"])
    operatorDef = staticmethod("is defined")

    def __init__(self, parent = None, signalManager = None, name = "Select data"):
        super().__init__(parent, signalManager)

        self.name2var = {}   # key: variable name, item: orange.Variable
        self.current_var = None
        self.negate_condition = False
        self.current_operator_dict = {ContinuousVariable: Operator(Operator.operatorsC[0], ContinuousVariable),
                                      DiscreteVariable: Operator(Operator.operatorsD[0], DiscreteVariable),
                                      StringVariable: Operator(Operator.operatorsS[0], StringVariable)}
        self.old_purge_classes = True

        self.loaded_var_names = []
        self.loaded_conditions = []

        box = gui.widgetBox(self.controlArea, 'Conditions')
        self.condition_list = QtGui.QTableWidget(box)
        self.condition_list.setShowGrid(False)
        self.condition_list.setSelectionMode(QTableWidget.SingleSelection)
        self.condition_list.setColumnCount(2)
        self.condition_list.setRowCount(1)
        self.condition_list.verticalHeader().hide()
        self.condition_list.horizontalHeader().hide()
        self.condition_list.resizeColumnToContents(0)
        self.condition_list.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.condition_list.horizontalHeader().setResizeMode(1, QtGui.QHeaderView.Stretch)
        self.addRow()

        # enabled  attribute condition value [and value] [case sensitive] [remove]

        info = gui.widgetBox(self.controlArea, '', orientation="horizontal")
        boxDataIn = gui.widgetBox(info, 'Data In')
        self.dataInExamplesLabel = gui.widgetLabel(boxDataIn, "num rows")
        self.dataInAttributesLabel = gui.widgetLabel(boxDataIn, "num variables")
        gui.rubber(boxDataIn)

        boxDataOut = gui.widgetBox(self, 'Data Out')
        self.dataOutExamplesLabel = gui.widgetLabel(boxDataOut, "num rows")
        self.dataOutAttributesLabel = gui.widgetLabel(boxDataOut, "num variables")
        gui.rubber(boxDataOut)

        boxSettings = gui.widgetBox(self, 'Commit')
        cb = gui.checkBox(boxSettings, self, "purge_attributes", "Remove unused values/attributes", box=None,
                          callback=self.on_purge_change)
        self.purgeClassesCB = gui.checkBox(gui.indentedBox(boxSettings, sep=gui.checkButtonOffsetHint(cb)),
                                           self, "purge_classes", "Remove unused classes",
                                           callback=self.on_purge_change)
        gui.checkBox(boxSettings, self, "update_on_change", "Commit on change", box=None)
        gui.button(boxSettings, self, "Commit", self.setOutput, default=True)

        self.icons = self.createAttributeIconDict()
        self.set_data(None)
        self.resize(500,661)


    class RowControls:
        def __init__(self, check_box, condition_box, rule_box, control_box,
                     attribute_combo, operator_combo):
            self.check_box = check_box
            self.condition_box = condition_box
            self.rule_box = rule_box
            self.control_box = control_box
            self.attribute_combo = attribute_combo
            self.operator_combo = operator_combo

    def addRow(self):
        row = 42
        self.condition_list.insertRow(row)
        cb = gui.checkBox(self.condition_list, self, "", "")
        self.condition_list.setWidget(row, 1, cb)

        condition_box = gui.widgetBox(self.condition_list, orientation="horizontal")
        rule_box = gui.widgetLabel(condition_box, "(undefined condition)")
        control_box = gui.widgetBox(condition_box, orientation="horizontal")

        attr_combo = gui.comboBox(control_box, self, None)
        attr_combo.addItems(self.all_variables)
        self.condition_list.setWidget(row, 2, attr_combo)

        oper_combo = QtGui.QListBox()
        self.condition_list.setWidget(row, 3, oper_combo)

        tr = QtGui.QIcon() # fix: trash

        row_controls = self.RowControls(cb, condition_box, rule_box, control_box,
                                   attr_combo, oper_combo)
        cb.changed.connect(partial(self.update_output, row_controls))
        rule_box.on_hover.connect(partial(self.rule_box_hover, row_controls)) # fix!
        control_box.on_mouse_out.connect(partial(self.control_box_out, row_controls))
        attr_combo.currentRowChanged.connect(partial(self.attribute_changed, row_controls))
        tr.clicked.connect(partial(self.remove_condition, row_controls))

        rule_box.hide()

    def rule_box_hover(self, row_controls):
        row_controls.rule_box.hide()
        row_controls.control_box.show()

    def control_box_out(self, row_controls):
        row_controls.control_box.hide()
        row_controls.rule_box.hide()

    def update_output(self, *_):
        pass

    def attribute_changed(self, row_controls, *_):
        var_name = row_controls.attribute_combo.currentValue()
        var = self.data.domain[var_name]
        prev_operator = row_controls.operator_combo.currentValue()
        operators = self.operator_names[type(var)]
        if row_controls.checkbox.enabled(): # Fix the method name!
            self.update_output()


    def set_data(self, data):
        self.closeContext()
        self.data = data
        self.bas = getCached(data, orange.DomainBasicAttrStat, (data,))

        self.name2var = {}
        self.conditions = []

        if self.data:
            self.all_variables = ["(select variable)"] + [x.name for x in chain(data.domain, data.domain.metas)]
            optmetas = self.data.domain.getmetas(True).values()
            optmetas.sort(lambda x,y: cmp(x.name, y.name))
            self.varList = self.data.domain.variables.native() + self.data.domain.getmetas(False).values() + optmetas
            for v in self.varList:
                self.name2var[v.name] = v
            self.setLbAttr()
            self.boxButtons.setEnabled(True)
        else:
            self.varList = []
            self.current_var = None

            self.lbAttr.clear()
            self.leSelect.clear()
            self.boxButtons.setEnabled(False)

        self.openContext("", data)
        self.synchronizeTable()
        self.condition_list.setCurrentCell(-1,1)

        self.updateOperatorStack()
        self.updateValuesStack()
        self.updateInfoIn(self.data)
        self.setOutput()


    def setLbAttr(self):
        self.lbAttr.clear()
        if not self.attr_search_text:
            for v in self.varList:
                self.lbAttr.addItem(QListWidgetItem(self.icons[v.varType], v.name))
        else:
            flen = len(self.attr_search_text)
            for v in self.varList:
                if v.name[:flen].lower() == self.attr_search_text.lower():
                    self.lbAttr.addItem(QListWidgetItem(self.icons[v.varType], v.name))

        if self.lbAttr.count():
            self.lbAttr.item(0).setSelected(True)
        else:
            self.lbAttrChange()


    def setOutputIf(self):
        if self.update_on_change:
            self.setOutput()

    def setOutput(self):
        matchingOutput = self.data
        nonMatchingOutput = None
        hasClass = False
        if self.data:
            hasClass = bool(self.data.domain.classVar)
            filterList = self.getFilterList(self.data.domain, self.conditions, enabledOnly=True)
            if len(filterList)>0:
                filter = orange.Filter_disjunction([orange.Filter_conjunction(l) for l in filterList])
            else:
                filter = orange.Filter_conjunction([]) # a filter that does nothing
            matchingOutput = filter(self.data, 1)
            matchingOutput.name = self.data.name
            nonMatchingOutput = filter(self.data, 1, negate=1)
            nonMatchingOutput.name = self.data.name

            if self.purge_attributes or self.purge_classes:
                remover = orange.RemoveUnusedValues(removeOneValued=True)

                newDomain = remover(matchingOutput, 0, True, self.purge_classes)
                if newDomain != matchingOutput.domain:
                    matchingOutput = orange.ExampleTable(newDomain, matchingOutput)

                newDomain = remover(nonMatchingOutput, 0, True, self.purge_classes)
                if newDomain != nonMatchingOutput.domain:
                    nonmatchingOutput = orange.ExampleTable(newDomain, nonMatchingOutput)

        self.send("Matching Data", matchingOutput)
        self.send("Unmatched Data", nonMatchingOutput)

        self.updateInfoOut(matchingOutput)


    def getFilterList(self, domain, conditions, enabledOnly):
        """Returns list of lists of orange filters, e.g. [[f1,f2],[f3]].
        OR is always enabled (with no respect to cond.enabled)
        """
        fdList = [[]]
        for cond in conditions:
            if cond.type == "OR":
                fdList.append([])
            elif cond.enabled or not enabledOnly:
                fdList[-1].append(cond.operator.getFilter(domain, cond.varName, cond.val1, cond.val2, cond.negated, cond.caseSensitive))
        return fdList


    def lbAttrChange(self):
        if self.lbAttr.selectedItems() == []: return
        text = str(self.lbAttr.selectedItems()[0].text())
        prevVar = self.current_var
        if prevVar:
            prevVarType = prevVar.varType
            prevVarName = prevVar.name
        else:
            prevVarType = None
            prevVarName = None
        try:
            self.current_var = self.data.domain[text]
        except:
            self.current_var = None
        if self.current_var:
            currVarType = self.current_var.varType
            currVarName = self.current_var.name
        else:
            currVarType = None
            currVarName = None
        if currVarType != prevVarType:
            self.updateOperatorStack()
        if currVarName != prevVarName:
            self.updateValuesStack()


    def lbOperatorsChange(self):
        """Updates value stack, only if necessary.
        """
        if self.current_var:
            varType = self.current_var.varType
            selItems = self.lbOperatorsDict[varType].selectedItems()
            if not selItems: return
            self.current_operator_dict[varType] = Operator(str(selItems[0].text()), varType)
            self.updateValuesStack()


    def lbValsChange(self):
        """Updates list of selected discrete values (self.currentVals).
        """
        self.current_vals = []
        for i in range(0, self.lbVals.count()):
            if self.lbVals.item(i).isSelected():
                self.current_vals.append(str(self.lbVals.item(i).text()))


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

        self.setOutputIf()


    def OnNewCondition(self):
        cond = self.getConditionFromSelection()
        if not cond:
            return

        where = min(self.condition_list.currentRow() + 1, self.condition_list.rowCount())
        self.conditions.insert(where, cond)
        self.synchronizeTable()
        self.condition_list.setCurrentCell(where, 1)
        self.setOutputIf()
        self.leSelect.clear()


    def OnUpdateCondition(self):
        row = self.condition_list.currentRow()
        if row < 0:
            return
        cond = self.getConditionFromSelection()
        if not cond:
            return
        self.conditions[row] = cond
        self.synchronizeTable()
        self.setOutputIf()
        self.leSelect.clear()


    def OnRemoveCondition(self):
        """Removes current condition table row, shifts rows up, updates conditions and sends out new data.
        """
        # update self.Conditions
        currRow = self.condition_list.currentRow()
        if currRow < 0:
            return
        self.conditions.pop(currRow)
        self.synchronizeTable()
        self.condition_list.setCurrentCell(min(currRow, self.condition_list.rowCount()-1), 1)
        self.setOutputIf()


    def OnDisjunction(self):
        """Updates conditions and condition table, sends out new data.
        """
        # update self.Conditions
        where = min(self.condition_list.currentRow() + 1, self.condition_list.rowCount())
        self.conditions.insert(where, Condition(True, "OR"))
        self.synchronizeTable()
        self.condition_list.setCurrentCell(where, 1)
        self.setOutputIf()


    def btnMoveUpClicked(self):
        """Moves the selected condition one row up.
        """
        currRow = self.condition_list.currentRow()
        numRows = self.condition_list.rowCount()
        if currRow < 1 or currRow >= numRows:
            return
        self.conditions = self.conditions[:currRow-1] + [self.conditions[currRow], self.conditions[currRow-1]] + self.conditions[currRow+1:]
        self.synchronizeTable()
        self.condition_list.setCurrentCell(max(0, currRow-1), 1)
        self.updateMoveButtons()
        self.setOutputIf()


    def btnMoveDownClicked(self):
        """Moves the selected condition one row down.
        """
        currRow = self.condition_list.currentRow()
        numRows = self.condition_list.rowCount()
        if currRow < 0 or currRow >= numRows-1:
            return
        self.conditions = self.conditions[:currRow] + [self.conditions[currRow+1], self.conditions[currRow]] + self.conditions[currRow+2:]
        self.synchronizeTable()
        self.condition_list.setCurrentCell(min(currRow+1, self.condition_list.rowCount()-1), 1)
        self.updateMoveButtons()
        self.setOutputIf()


    def currentCriteriaChange(self, row, col):
        """Handles current row change in criteria table;
        select attribute and operator, and set values according to the selected condition.
        """
        if row < 0:
            return
        cond = self.conditions[row]
        if cond.type != "OR":
            # attribute
            lbItems = self.lbAttr.findItems(cond.varName, Qt.MatchExactly)
            if lbItems != []:
                self.lbAttr.setCurrentItem(lbItems[0])
            # not
            self.cbNot.setChecked(cond.negated)
            # operator
            for vt,lb in self.lbOperatorsDict.items():
                if vt == self.name2var[cond.varName].varType:
                    lb.show()
                else:
                    lb.hide()
            lbItems = self.lbOperatorsDict[self.name2var[cond.varName].varType].findItems(str(cond.operator), Qt.MatchExactly)
            if lbItems != []:
                self.lbOperatorsDict[self.name2var[cond.varName].varType].setCurrentItem(lbItems[0])
            # values
            self.valuesStack.setCurrentWidget(self.boxIndices[self.name2var[cond.varName].varType])
            if self.name2var[cond.varName].varType == orange.VarTypes.Continuous:
                self.leNum1.setText(str(cond.val1))
                if cond.operator.isInterval:
                    self.leNum2.setText(str(cond.val2))
            elif self.name2var[cond.varName].varType == orange.VarTypes.String:
                self.leStr1.setText(str(cond.val1))
                if cond.operator.isInterval:
                    self.leStr2.setText(str(cond.val2))
                self.cbCaseSensitive.setChecked(cond.caseSensitive)
            elif self.name2var[cond.varName].varType == orange.VarTypes.Discrete:
                self.lbVals.clearSelection()
                for val in cond.val1:
                    lbItems = self.lbVals.findItems(val, Qt.MatchExactly)
                    for item in lbItems:
                        item.setSelected(1)
        self.updateMoveButtons()


    def criteriaActiveChange(self, condition, active):
        """Handles clicks on criteria table checkboxes, send out new data.
        """
        condition.enabled = active
        # update the numbers of examples that matches "OR" filter
        self.updateFilteredDataLens(condition)
        # send out new data
        if self.update_on_change:
            self.setOutput()


    ############################################################################################################################################################
    ## Interface state management - updates interface elements based on selection in list boxes ################################################################
    ############################################################################################################################################################

    def updateMoveButtons(self):
        """enable/disable Move Up/Down buttons
        """
        row = self.condition_list.currentRow()
        numRows = self.condition_list.rowCount()
        if row > 0:
            self.btnMoveUp.setEnabled(True)
        else:
            self.btnMoveUp.setEnabled(False)
        if row < numRows-1:
            self.btnMoveDown.setEnabled(True)
        else:
            self.btnMoveDown.setEnabled(False)


    def updateOperatorStack(self):
        """Raises listbox with appropriate operators.
        """
        if self.current_var:
            varType = self.current_var.varType
            self.btnNew.setEnabled(True)
        else:
            varType = 0
            self.btnNew.setEnabled(False)
        for vt,lb in self.lbOperatorsDict.items():
            if vt == varType:
                lb.show()
                try:
                    lb.setCurrentRow(self.data.domain.isOptionalMeta(self.current_var) and lb.count() - 1)
                except:
                    lb.setCurrentRow(0)
            else:
                lb.hide()


    def updateValuesStack(self):
        """Raises appropriate widget for values from stack,
        fills listBox for discrete attributes,
        shows statistics for continuous attributes.
        """
        if self.current_var:
            varType = self.current_var.varType
        else:
            varType = 0
        currentOper = self.current_operator_dict.get(varType,None)
        if currentOper:
            # raise widget
            self.valuesStack.setCurrentWidget(self.boxIndices[currentOper.varType])
            if currentOper.varType==orange.VarTypes.Discrete:
                # store selected discrete values, refill values list box, set single/multi selection mode, restore selected item(s)
                selectedItemNames = []
                for i in range(self.lbVals.count()):
                    if self.lbVals.item(i).isSelected():
                        selectedItemNames.append(str(self.lbVals.item(i).text()))
                self.lbVals.clear()
                curVarValues = []
                for value in self.current_var:
                    curVarValues.append(str(value))
                curVarValues.sort()
                for value in curVarValues:
                    self.lbVals.addItem(str(value))
                if currentOper.isInterval:
                    self.lbVals.setSelectionMode(QListWidget.MultiSelection)
                else:
                    self.lbVals.setSelectionMode(QListWidget.SingleSelection)
                isSelected = False
                for name in selectedItemNames:
                    items = self.lbVals.findItems(name, Qt.MatchExactly)
                    for item in items:
                        item.setSelected(1)
                        isSelected = True
                        if not currentOper.isInterval:
                            break
                if not isSelected:
                    if self.lbVals.count() > 0:
                        self.lbVals.item(0).setSelected(True)
                    else:
                        self.current_vals = []
            elif currentOper.varType==orange.VarTypes.Continuous:
                # show / hide "and" label and 2nd line edit box
                if currentOper.isInterval:
                    self.lblAndCon.show()
                    self.leNum2.show()
                else:
                    self.lblAndCon.hide()
                    self.leNum2.hide()
                # display attribute statistics
                if self.current_var in self.data.domain.variables:
                    basstat = self.bas[self.current_var]
                else:
                    basstat = orange.BasicAttrStat(self.current_var, self.data)

                if basstat.n:
                    min, avg, max = ["%.3f" % x for x in (basstat.min, basstat.avg, basstat.max)]
                    self.num1, self.num2 = basstat.min, basstat.max
                else:
                    min = avg = max = "-"
                    self.num1 = self.num2 = 0

                self.lblMin.setText("Min: %s" % min)
                self.lblAvg.setText("Avg: %s" % avg)
                self.lblMax.setText("Max: %s" % max)
                self.lblDefined.setText("Defined for %i example(s)" % basstat.n)

            elif currentOper.varType==orange.VarTypes.String:
                # show / hide "and" label and 2nd line edit box
                if currentOper.isInterval:
                    self.lblAndStr.show()
                    self.leStr2.show()
                else:
                    self.lblAndStr.hide()
                    self.leStr2.hide()
        else:
            self.valuesStack.setCurrentWidget(self.boxIndices[0])


    def getConditionFromSelection(self):
        """Returns a condition according to the currently selected attribute / operator / values.
        """
        if self.current_var:
            if self.current_var.varType == orange.VarTypes.Continuous:
                try:
                    val1 = float(self.num1)
                    val2 = float(self.num2)
                except ValueError:
                    return
            elif self.current_var.varType == orange.VarTypes.String:
                val1 = self.str1
                val2 = self.str2
            elif self.current_var.varType == orange.VarTypes.Discrete:
                val1 = self.current_vals
                if not val1:
                    return
                val2 = None
            if not self.current_operator_dict[self.current_var.varType].isInterval:
                val2 = None
            return Condition(True, "AND", self.current_var.name, self.current_operator_dict[self.current_var.varType], self.negate_condition, val1, val2, self.case_sensitive)


    def synchronizeTable(self):
#        for row in range(len(self.Conditions), self.criteriaTable.rowCount()):
#            self.criteriaTable.clearCellWidget(row,0)
#            self.criteriaTable.clearCell(row,1)

        currentRow = self.condition_list.currentRow()
        self.condition_list.clearContents()
        self.condition_list.setRowCount(len(self.conditions))

        for row, cond in enumerate(self.conditions):
            if cond.type == "OR":
                cw = QLabel("", self)
            else:
                cw = QCheckBox(str(len(cond.operator.getFilter(self.data.domain, cond.varName, cond.val1, cond.val2, cond.negated, cond.caseSensitive)(self.data))), self)
#                cw.setChecked(cond.enabled)
                self.connect(cw, SIGNAL("toggled(bool)"), lambda val, cond=cond: self.criteriaActiveChange(cond, val))

            self.condition_list.setCellWidget(row, 0, cw)
# This is a fix for Qt bug (4.3). When Qt is fixed, the setChecked above should suffice
# but now it unchecks the checkbox as it is inserted
            if cond.type != "OR":
                cw.setChecked(cond.enabled)

            # column 1
            if cond.type == "OR":
                txt = "OR"
            else:
                soper = str(cond.operator)
                if cond.negated and soper in Operator.negations:
                    txt = "'%s' %s " % (cond.varName, Operator.negations[soper])
                else:
                    txt = (cond.negated and "NOT " or "") + "'%s' %s " % (cond.varName, soper)
                if cond.operator != Operator.operatorDef:
                    if cond.operator.varType == orange.VarTypes.Discrete:
                        if cond.operator.isInterval:
                            if len(cond.val1) > 0:
                                txt += "["
                                for name in cond.val1:
                                    txt += "%s, " % name
                                txt = txt[0:-2] + "]"
                            else:
                                txt += "[]"
                        else:
                            txt += cond.val1[0]
                    elif cond.operator.varType == orange.VarTypes.String:
                        if cond.caseSensitive:
                            cs = " (C)"
                        else:
                            cs = ""
                        if cond.operator.isInterval:
                            txt += "'%s'%s and '%s'%s" % (cond.val1, cs, cond.val2, cs)
                        else:
                            txt += "'%s'%s" % (cond.val1, cs)
                    elif cond.operator.varType == orange.VarTypes.Continuous:
                        if cond.operator.isInterval:
                            txt += str(cond.val1) + " and " + str(cond.val2)
                        else:
                            txt += str(cond.val1)

            OWGUI.tableItem(self.condition_list, row, 1, txt)

        self.condition_list.setCurrentCell(max(currentRow, len(self.conditions) - 1), 0)
        self.condition_list.resizeRowsToContents()
        self.updateFilteredDataLens()

        en = len(self.conditions)
        self.btnUpdate.setEnabled(en)
        self.btnRemove.setEnabled(en)
        self.updateMoveButtons()


    def updateFilteredDataLens(self, cond=None):
        """Updates the number of examples that match individual conditions in criteria table.
        If cond is given, updates the given row and the corresponding OR row;
        if cond==None, updates the number of examples in OR rows.
        """
        if cond:
            condIdx = self.conditions.index(cond)
            # idx1: the first non-OR condition above the clicked condition
            # idx2: the first OR condition below the clicked condition
            idx1 = 0
            idx2 = len(self.conditions)
            for i in range(condIdx,idx1-1,-1):
                if self.conditions[i].type == "OR":
                    idx1 = i+1
                    break
            for i in range(condIdx+1,idx2):
                if self.conditions[i].type == "OR":
                    idx2 = i
                    break
            fdListAll = self.getFilterList(self.data.domain, self.conditions[idx1:idx2], enabledOnly=False)
            fdListEnabled = self.getFilterList(self.data.domain, self.conditions[idx1:idx2], enabledOnly=True)
            # if we click on the row which has a preceeding OR: update OR at index idx1-1
            if idx1 > 0:
                self.condition_list.cellWidget(idx1-1,0).setText(str(len(orange.Filter_conjunction(fdListEnabled[0])(self.data))))
            # update the clicked row
            self.condition_list.cellWidget(condIdx,0).setText(str(len(fdListAll[0][condIdx-idx1](self.data))))

        elif len(self.conditions) > 0:
            # update all "OR" rows
            fdList = self.getFilterList(self.data.domain, self.conditions, enabledOnly=True)
            idx = 1
            for row,cond in enumerate(self.conditions):
                if cond.type == "OR":
                    self.condition_list.cellWidget(row,0).setText(str(len(orange.Filter_conjunction(fdList[idx])(self.data))))
                    idx += 1


    def updateInfoIn(self, data):
        """Updates data in info box.
        """
        if data:
            varList = data.domain.variables.native() + data.domain.getmetas().values()
            self.dataInAttributesLabel.setText("%s attribute%s" % self.sp(varList))
            self.dataInExamplesLabel.setText("%s example%s" % self.sp(data))
        else:
            self.dataInExamplesLabel.setText("No examples.")
            self.dataInAttributesLabel.setText("No attributes.")


    def updateInfoOut(self, data):
        """Updates data out info box.
        """
        if data:
            varList = data.domain.variables.native() + data.domain.getmetas().values()
            self.dataOutAttributesLabel.setText("%s attribute%s" % self.sp(varList))
            self.dataOutExamplesLabel.setText("%s example%s" % self.sp(data))
        else:
            self.dataOutExamplesLabel.setText("No examples.")
            self.dataOutAttributesLabel.setText("No attributes.")


    ############################################################################################################################################################
    ## Utility functions #######################################################################################################################################
    ############################################################################################################################################################

    def sp(self, l, capitalize=True):
        """Input: list; returns tupple (str(len(l)), "s"/"")
        """
        n = len(l)
        if n == 0:
            if capitalize:
                return "No", "s"
            else:
                return "no", "s"
        elif n == 1:
            return str(n), ''
        else:
            return str(n), 's'


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
        self.reportRaw(OWReport.reportTable(self.condition_list))
#        self.reportTable("Conditions", self.criteriaTable)


class Condition:
    def __init__(self, enabled, type, attribute = None, operator = None, negate = False, value1 = None, value2 = None, caseSensitive = False):
        self.enabled = enabled                  # True/False
        self.type = type                        # "AND"/"OR"
        self.varName = attribute                # orange.Variable
        self.operator = operator                # Operator
        self.negated = negate                   # True/False
        self.val1 = value1                      # string/float
        self.val2 = value2                      # string/float
        self.caseSensitive = caseSensitive      # True/False

"""
class Operator:
    operatorsD = staticmethod(["equals","in"])
    operatorsC = staticmethod(["=","<","<=",">",">=","between","outside"])
    operatorsS = staticmethod(["=","<","<=",">",">=","contains","begins with","ends with","between","outside"])
    operatorDef = staticmethod("is defined")
    getOperators = staticmethod(lambda: Operator.operatorsD + Operator.operatorsS + [Operator.operatorDef])

    negations = {"equals": "does not equal", "in": "is not in",
                 "between": "not between", "outside": "not outside",
                 "contains": "does not contain", "begins with": "does not begin with", "ends with": "does not end with",
                 "is defined": "is undefined"}

#"Equal", "NotEqual", "Less", "LessEqual", "Greater", "GreaterEqual",
#     "Between", "Outside"

    _operFilter = {"=":orange.Filter_values.Equal,
                   "<":orange.Filter_values.Less,
                   "<=":orange.Filter_values.LessEqual,
                   ">":orange.Filter_values.Greater,
                   ">=":orange.Filter_values.GreaterEqual,
                   "between":orange.Filter_values.Between,
                   "outside":orange.Filter_values.Outside,
                   "contains":orange.Filter_values.Contains,
                   "begins with":orange.Filter_values.BeginsWith,
                   "ends with":orange.Filter_values.EndsWith}

    def __init__(self, operator, varType):
        ""Members: operator, varType, isInterval.
        ""
        assert operator in Operator.getOperators(), "Unknown operator: %s" % str(operator)
        self.operator = operator
        self.varType = varType
        self.isInterval = False
        if operator in Operator.operatorsC and Operator.operatorsC.index(operator) > 4 \
           or operator in Operator.operatorsD and Operator.operatorsD.index(operator) > 0 \
           or operator in Operator.operatorsS and Operator.operatorsS.index(operator) > 7:
            self.isInterval = True

    def __eq__(self, other):
        assert other in Operator.getOperators()
        return  self.operator == other

    def __ne__(self, other):
        assert other in Operator.getOperators()
        return self.operator != other

    def __repr__(self):
        return str(self.operator)

    def __strr__(self):
        return str(self.operator)

    def getFilter(self, domain, variable, value1, value2, negate, caseSensitive):
        ""Returns orange filter.
        ""
        if self.operator == Operator.operatorDef:
            try:
                id = domain.index(variable)
            except:
                error("Error: unknown attribute (%s)." % variable)

            if id >= 0:
                f = orange.Filter_isDefined(domain=domain)
                for v in domain.variables:
                    f.check[v] = 0
                f.check[variable] = 1
            else: # variable is a meta
                    f = orange.Filter_hasMeta(id = domain.index(variable))
        elif self.operator in Operator.operatorsD:
            f = orange.Filter_values(domain=domain)
            f[variable] = value1
        else:
            f = orange.Filter_values(domain=domain)
            if value2:
                f[variable] = (Operator._operFilter[str(self.operator)], value1, value2)
            else:
                f[variable] = (Operator._operFilter[str(self.operator)], value1)
            if self.varType == orange.VarTypes.String:
                f[variable].caseSensitive = caseSensitive
        f.negate = negate
        return f
"""


if __name__=="__main__":
    import sys
    #data = orange.ExampleTable('dicty_800_genes_from_table07.tab')
    data = orange.ExampleTable('../../doc/datasets/adult_sample.tab')
#    data = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\iris.tab")
    # add meta attribute
    #data.domain.addmeta(orange.newmetaid(), orange.StringVariable("workclass_name"))

    a=QApplication(sys.argv)
    ow=OWSelectData()
    ow.show()
    ow.set_data(data)
    a.exec_()

