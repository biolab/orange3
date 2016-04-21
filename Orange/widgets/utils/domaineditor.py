from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import QTableView, QSizePolicy

from Orange.data import DiscreteVariable, ContinuousVariable, StringVariable
from Orange.widgets import gui
from Orange.widgets.data.owcolor import HorizontalGridDelegate
from Orange.widgets.utils.itemmodels import TableModel


class VarTableModel(QtCore.QAbstractTableModel):
    places = "feature", "class", "meta", "skip"
    typenames = "nominal", "numeric", "string"
    vartypes = DiscreteVariable, ContinuousVariable, StringVariable
    name2type = dict(zip(typenames, vartypes))
    type2name = dict(zip(vartypes, typenames))

    def __init__(self, variables):
        super().__init__()
        self.variables = self.original = variables

    def set_domain(self, domain):
        def may_be_numeric(var):
            if var.is_continuous:
                return True
            if var.is_discrete:
                try:
                    sum(float(x) for x in var.values)
                    return True
                except ValueError:
                    return False
            return False

        self.modelAboutToBeReset.emit()
        self.variables[:] = self.original = [
            [var.name, type(var), place,
             ", ".join(var.values) if var.is_discrete else "",
             may_be_numeric(var)]
            for place, vars in enumerate(
                (domain.attributes, domain.class_vars, domain.metas))
            for var in vars
        ]
        self.modelReset.emit()

    def rowCount(self, parent):
        return 0 if parent.isValid() else len(self.variables)

    def columnCount(self, parent):
        return 0 if parent.isValid() else 4

    def data(self, index, role):
        row, col = index.row(), index.column()
        val = self.variables[row][col]
        if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
            if col == 1:
                return self.type2name[val]
            if col == 2:
                return self.places[val]
            else:
                return val
        if role == QtCore.Qt.DecorationRole:
            if col == 1:
                return gui.attributeIconDict[self.vartypes.index(val) + 1]
        if role == QtCore.Qt.ForegroundRole:
            if self.variables[row][2] == 3 and col != 2:
                return QtGui.QColor(160, 160, 160)
        if role == QtCore.Qt.BackgroundRole:
            place = self.variables[row][2]
            return TableModel.ColorForRole.get(place, None)

    def setData(self, index, value, role):
        row, col = index.row(), index.column()
        row_data = self.variables[row]
        if role == QtCore.Qt.EditRole:
            if col == 0:
                row_data[col] = value
            elif col == 1:
                vartype = self.name2type[value]
                row_data[col] = vartype
                if not vartype.is_primitive() and row_data[2] < 2:
                    row_data[2] = 2
            elif col == 2:
                row_data[col] = self.places.index(value)
            else:
                return False
            # Settings may change background colors
            self.dataChanged.emit(index.sibling(row, 0), index.sibling(row, 3))
            return True

    def flags(self, index):
        return super().flags(index) | QtCore.Qt.ItemIsEditable


class ComboDelegate(HorizontalGridDelegate):
    def __init__(self, view, items):
        super().__init__()
        self.view = view
        self.items = items

    def createEditor(self, parent, option, index):
        # This ugly hack closes the combo when the user selects an item
        class Combo(QtGui.QComboBox):
            def __init__(self, *args):
                super().__init__(*args)
                self.popup_shown = False
                self.highlighted_text = None

            def highlight(self, index):
                self.highlighted_text = index

            def showPopup(self, *args):
                super().showPopup(*args)
                self.popup_shown = True

            def hidePopup(me):
                if me.popup_shown:
                    self.view.model().setData(
                            index, me.highlighted_text, QtCore.Qt.EditRole)
                    self.popup_shown = False
                super().hidePopup()
                self.view.closeEditor(me, self.NoHint)

        combo = Combo(parent)
        combo.highlighted[str].connect(combo.highlight)
        return combo


class VarTypeDelegate(ComboDelegate):
    def setEditorData(self, combo, index):
        combo.clear()
        no_numeric = not self.view.model().variables[index.row()][4]
        ind = self.items.index(index.data())
        combo.addItems(self.items[:1] + self.items[1 + no_numeric:])
        combo.setCurrentIndex(ind - (no_numeric and ind > 1))


class PlaceDelegate(ComboDelegate):
    def setEditorData(self, combo, index):
        combo.clear()
        to_meta = not self.view.model().variables[index.row()][1].is_primitive()
        combo.addItems(self.items[2 * to_meta:])
        combo.setCurrentIndex(self.items.index(index.data()) - 2 * to_meta)


class DomainEditor(QTableView):
    def __init__(self, variables):
        super().__init__()
        self.setModel(VarTableModel(variables))
        self.setSelectionMode(QTableView.NoSelection)
        self.horizontalHeader().hide()
        self.horizontalHeader().setStretchLastSection(True)
        self.setShowGrid(False)
        self.setEditTriggers(
            QTableView.SelectedClicked | QTableView.DoubleClicked)
        self.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        # setItemDelegate(ForColumn) apparently does not take ownership
        # of delegates, and coredumps if a a references is not stored somewhere.
        # We thus store delegates as attributes
        self.grid_delegate = HorizontalGridDelegate()
        self.setItemDelegate(self.grid_delegate)
        self.vartype_delegate = VarTypeDelegate(self, VarTableModel.typenames)
        self.setItemDelegateForColumn(1, self.vartype_delegate)
        self.place_delegate = PlaceDelegate(self, VarTableModel.places)
        self.setItemDelegateForColumn(2, self.place_delegate)
