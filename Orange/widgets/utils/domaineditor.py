from AnyQt.QtCore import Qt, QAbstractTableModel
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import QComboBox, QTableView, QSizePolicy

from Orange.data import DiscreteVariable, ContinuousVariable, StringVariable, \
    TimeVariable
from Orange.widgets import gui
from Orange.widgets.gui import HorizontalGridDelegate
from Orange.widgets.utils.itemmodels import TableModel


class Column:
    name = 0
    tpe = 1
    place = 2
    values = 3
    not_valid = 4


class Place:
    feature = 0
    class_var = 1
    meta = 2
    skip = 3


class VarTableModel(QAbstractTableModel):
    DISCRETE_VALUE_DISPLAY_LIMIT = 20

    places = "feature", "target", "meta", "skip"
    typenames = "nominal", "numeric", "string", "datetime"
    vartypes = DiscreteVariable, ContinuousVariable, StringVariable, TimeVariable
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

        def discrete_value_display(value_list):
            result = ", ".join(str(v) for v in value_list[:VarTableModel.DISCRETE_VALUE_DISPLAY_LIMIT])
            if len(value_list) > VarTableModel.DISCRETE_VALUE_DISPLAY_LIMIT:
                result += ", ..."
            return result

        self.modelAboutToBeReset.emit()
        if domain is None:
            self.variables.clear()
        else:
            self.variables[:] = self.original = [
                [var.name, type(var), place,
                 discrete_value_display(var.values) if var.is_discrete else "",
                 may_be_numeric(var)]
                for place, vars in enumerate(
                    (domain.attributes, domain.class_vars, domain.metas))
                for var in vars
            ]
        self.modelReset.emit()

    def reset(self):
        self.modelAboutToBeReset.emit()
        self.variables[:] = []
        self.modelReset.emit()

    def rowCount(self, parent):
        return 0 if parent.isValid() else len(self.variables)

    def columnCount(self, parent):
        return 0 if parent.isValid() else Column.not_valid

    def data(self, index, role):
        row, col = index.row(), index.column()
        val = self.variables[row][col]
        if role == Qt.DisplayRole or role == Qt.EditRole:
            if col == Column.tpe:
                return self.type2name[val]
            if col == Column.place:
                return self.places[val]
            else:
                return val
        if role == Qt.DecorationRole:
            if col == Column.tpe:
                return gui.attributeIconDict[self.vartypes.index(val) + 1]
        if role == Qt.ForegroundRole:
            if self.variables[row][Column.place] == Place.skip \
                    and col != Column.place:
                return QColor(160, 160, 160)
        if role == Qt.BackgroundRole:
            place = self.variables[row][Column.place]
            mapping = [Place.meta, Place.feature, Place.class_var, None]
            return TableModel.ColorForRole.get(mapping[place], None)

    def setData(self, index, value, role):
        row, col = index.row(), index.column()
        row_data = self.variables[row]
        if role == Qt.EditRole:
            if col == Column.name:
                row_data[col] = value
            elif col == Column.tpe:
                vartype = self.name2type[value]
                row_data[col] = vartype
                if not vartype.is_primitive() and \
                                row_data[Column.place] < Place.meta:
                    row_data[Column.place] = Place.meta
            elif col == Column.place:
                row_data[col] = self.places.index(value)
            else:
                return False
            # Settings may change background colors
            self.dataChanged.emit(index.sibling(row, 0), index.sibling(row, 3))
            return True

    def flags(self, index):
        if index.column() == Column.values:
            return super().flags(index)
        return super().flags(index) | Qt.ItemIsEditable


class ComboDelegate(HorizontalGridDelegate):
    def __init__(self, view, items):
        super().__init__()
        self.view = view
        self.items = items

    def createEditor(self, parent, option, index):
        # This ugly hack closes the combo when the user selects an item
        class Combo(QComboBox):
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
                            index, me.highlighted_text, Qt.EditRole)
                    self.popup_shown = False
                super().hidePopup()
                self.view.closeEditor(me, self.NoHint)

        combo = Combo(parent)
        combo.highlighted[str].connect(combo.highlight)
        return combo


class VarTypeDelegate(ComboDelegate):
    def setEditorData(self, combo, index):
        combo.clear()
        no_numeric = not self.view.model().variables[
            index.row()][Column.not_valid]
        ind = self.items.index(index.data())
        combo.addItems(self.items[:1] + self.items[1 + no_numeric:])
        combo.setCurrentIndex(ind - (no_numeric and ind > 1))


class PlaceDelegate(ComboDelegate):
    def setEditorData(self, combo, index):
        combo.clear()
        to_meta = not self.view.model().variables[
            index.row()][Column.tpe].is_primitive()
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
        self.setItemDelegateForColumn(Column.tpe, self.vartype_delegate)
        self.place_delegate = PlaceDelegate(self, VarTableModel.places)
        self.setItemDelegateForColumn(Column.place, self.place_delegate)
