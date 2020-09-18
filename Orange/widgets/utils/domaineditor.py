from itertools import chain
from copy import deepcopy

import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import Qt, QAbstractTableModel
from AnyQt.QtGui import QColor, QFont
from AnyQt.QtWidgets import QComboBox, QTableView, QSizePolicy

from Orange.data import DiscreteVariable, ContinuousVariable, StringVariable, \
    TimeVariable, Domain
from Orange.misc.collections import natural_sorted
from Orange.data.util import get_unique_names_duplicates
from Orange.statistics.util import unique
from Orange.widgets import gui
from Orange.widgets.gui import HorizontalGridDelegate
from Orange.widgets.settings import ContextSetting
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
    typenames = "categorical", "numeric", "text", "datetime"
    vartypes = DiscreteVariable, ContinuousVariable, StringVariable, TimeVariable
    name2type = dict(zip(typenames, vartypes))
    type2name = dict(zip(vartypes, typenames))

    def __init__(self, variables, *args):
        super().__init__(*args)
        self.variables = variables
        self.orig_variables = None
        self.set_orig_variables(variables)

    def set_orig_variables(self, variables):
        self.orig_variables = deepcopy(variables)

    def reset_variables(self):
        self.set_variables(deepcopy(self.orig_variables))

    def set_variables(self, variables):
        self.modelAboutToBeReset.emit()
        self.variables = variables
        self.modelReset.emit()

    def rowCount(self, parent):
        return 0 if parent.isValid() else len(self.variables)

    @staticmethod
    def columnCount(parent):
        return 0 if parent.isValid() else Column.not_valid

    def data(self, index, role):
        row, col = index.row(), index.column()
        val = self.variables[row][col]
        if role in (Qt.DisplayRole, Qt.EditRole):
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
        if role == Qt.FontRole:
            if self.variables[row] != self.orig_variables[row]:
                font = QFont()
                font.setBold(True)
                return font
        return None

    def setData(self, index, value, role=Qt.EditRole):
        row, col = index.row(), index.column()
        row_data = self.variables[row]
        if role == Qt.EditRole:
            if col == Column.name and not (value.isspace() or value == ""):
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
        return False

    def headerData(self, i, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole and i < 4:
            return ("Name", "Type", "Role", "Values")[i]
        if role == Qt.TextAlignmentRole:
            return Qt.AlignLeft
        return super().headerData(i, orientation, role)

    def flags(self, index):
        if index.column() == Column.values:
            return super().flags(index)
        return super().flags(index) | Qt.ItemIsEditable


class ComboDelegate(HorizontalGridDelegate):
    def __init__(self, view, items):
        super().__init__()
        self.view = view
        self.items = items

    def createEditor(self, parent, _option, index):
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

            # Here, we need `self` from the closure
            # pylint: disable=no-self-argument,attribute-defined-outside-init
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
        if no_numeric:
            # Do not allow selection of numeric and datetime
            items = [i for i in self.items if i not in ("numeric", "datetime")]
        else:
            items = self.items

        ind = items.index(index.data())
        combo.addItems(items)
        combo.setCurrentIndex(ind)
        combo.showPopup()


class PlaceDelegate(ComboDelegate):
    def setEditorData(self, combo, index):
        combo.clear()
        to_meta = not self.view.model().variables[
            index.row()][Column.tpe].is_primitive()
        combo.addItems(self.items[2 * to_meta:])
        combo.setCurrentIndex(self.items.index(index.data()) - 2 * to_meta)
        combo.showPopup()


class DomainEditor(QTableView):
    """Component for editing of variable types.

    Parameters
    ----------
    widget : parent widget
    """

    variables = ContextSetting([])

    def __init__(self, widget):
        super().__init__()
        widget.settingsHandler.initialize(self)
        widget.contextAboutToBeOpened.connect(lambda args: self.set_domain(args[0]))
        widget.contextOpened.connect(lambda: self.model().set_variables(self.variables))
        widget.contextClosed.connect(lambda: self.model().set_variables([]))

        self.setModel(VarTableModel(self.variables, self))

        self.setSelectionMode(QTableView.NoSelection)
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

    @staticmethod
    def _is_missing(x):
        return str(x) in ("nan", "")

    @staticmethod
    def _iter_vals(x):
        """Iterate over values of sparse or dense arrays."""
        for i in range(x.shape[0]):
            yield x[i, 0]

    @staticmethod
    def _to_column(x, to_sparse, dtype=None):
        """Transform list of values to sparse/dense column array."""
        x = np.asarray(x, dtype=dtype).reshape(-1, 1)
        if to_sparse:
            x = sp.csc_matrix(x)
        return x

    @staticmethod
    def _merge(cols, force_dense=False):
        if len(cols) == 0:
            return None

        all_dense = not any(sp.issparse(c) for c in cols)
        if all_dense:
            return np.hstack(cols)
        if force_dense:
            return np.hstack([c.toarray() if sp.issparse(c) else c for c in cols])
        sparse_cols = [c if sp.issparse(c) else sp.csc_matrix(c) for c in cols]
        return sp.hstack(sparse_cols).tocsr()

    def get_domain(self, domain, data, deduplicate=False):
        """
        Create domain (and dataset) from changes made in the widget.

        Returns
        -------

        Args:
            domain (Domain): original domain
            data (Table): original data
            deduplicate (bool): if True, variable names are deduplicated and
               the result contains an additional list with names of renamed
               variables

        Returns:
            (new_domain, [attribute_columns, class_var_columns, meta_columns])
            or
            (new_domain, [attribute_columns, class_var_columns, meta_columns], renamed)
        """
        # Allow type-checking with type() instead of isinstance() for exact comparison
        # pylint: disable=unidiomatic-typecheck
        variables = self.model().variables
        places = [[], [], []]  # attributes, class_vars, metas
        cols = [[], [], []]  # Xcols, Ycols, Mcols

        def numbers_are_round(var, col_data):
            if type(var) == ContinuousVariable:
                data = np.asarray(col_data.data)  # Works for dense and sparse
                data = data[~np.isnan(data)]
                return (data == data.astype(int)).all()
            return False

        # Exit early with original domain if the user didn't actually change anything
        if all((name == orig_var.name and tpe == type(orig_var) and place == orig_plc)
               for (name, tpe, place, _, _), (orig_var, orig_plc) in
               zip(variables,
                   chain(((at, Place.feature) for at in domain.attributes),
                         ((cl, Place.class_var) for cl in domain.class_vars),
                         ((mt, Place.meta) for mt in domain.metas)))):
            if deduplicate:
                return domain, [data.X, data.Y, data.metas], []
            else:
                return domain, [data.X, data.Y, data.metas]

        relevant_names = [var[0] for var in variables if var[2] != Place.skip]
        if deduplicate:
            renamed_iter = iter(get_unique_names_duplicates(relevant_names))
        else:
            renamed_iter = iter(relevant_names)
        renamed = []
        for (name, tpe, place, _, may_be_numeric), (orig_var, orig_plc) in \
                zip(variables,
                        chain([(at, Place.feature) for at in domain.attributes],
                              [(cl, Place.class_var) for cl in domain.class_vars],
                              [(mt, Place.meta) for mt in domain.metas])):
            if place == Place.skip:
                continue

            new_name = next(renamed_iter)
            if new_name != name and name not in renamed:
                renamed.append(name)

            col_data = self._get_column(data, orig_var, orig_plc)
            is_sparse = sp.issparse(col_data)

            if new_name == orig_var.name and tpe == type(orig_var):
                var = orig_var
            elif tpe == type(orig_var):
                var = orig_var.copy(name=new_name)
            elif tpe == DiscreteVariable:
                values = natural_sorted(
                    list(str(i) for i in unique(col_data)
                         if not self._is_missing(i))
                )
                round_numbers = numbers_are_round(orig_var, col_data)
                col_data = [np.nan if self._is_missing(x) else values.index(str(x))
                            for x in self._iter_vals(col_data)]
                if round_numbers:
                    values = [str(int(float(v))) for v in values]
                var = tpe(new_name, values)
                col_data = self._to_column(col_data, is_sparse)
            elif tpe == StringVariable:
                var = tpe.make(new_name)
                if type(orig_var) in [DiscreteVariable, TimeVariable]:
                    col_data = [orig_var.repr_val(x) if not np.isnan(x) else ""
                                for x in self._iter_vals(col_data)]
                elif type(orig_var) == ContinuousVariable:
                    round_numbers = numbers_are_round(orig_var, col_data)
                    col_data = ['' if np.isnan(x) else
                                str(int(x)) if round_numbers else
                                orig_var.repr_val(x)
                                for x in self._iter_vals(col_data)]
                # don't obey sparsity for StringVariable since they are
                # in metas which are transformed to dense below
                col_data = self._to_column(col_data, False, dtype=object)
            elif tpe == ContinuousVariable and type(orig_var) == DiscreteVariable:
                var = tpe.make(new_name)
                if may_be_numeric:
                    col_data = [np.nan if self._is_missing(x) else float(orig_var.values[int(x)])
                                for x in self._iter_vals(col_data)]
                col_data = self._to_column(col_data, is_sparse)
            else:
                var = tpe(new_name)
            places[place].append(var)
            cols[place].append(col_data)

        # merge columns for X, Y and metas
        feats = cols[Place.feature]
        X = self._merge(feats) if feats else np.empty((len(data), 0))
        Y = self._merge(cols[Place.class_var], force_dense=True)
        m = self._merge(cols[Place.meta], force_dense=True)
        domain = Domain(*places)
        if deduplicate:
            return domain, [X, Y, m], renamed
        else:
            return domain, [X, Y, m]

    @staticmethod
    def _get_column(data, source_var, source_place):
        """ Extract column from data and preserve sparsity. """
        if source_place == Place.meta:
            col_data = data[:, source_var].metas
        elif source_place == Place.class_var:
            col_data = data[:, source_var].Y.reshape(-1, 1)
        else:
            col_data = data[:, source_var].X
        return col_data

    def set_domain(self, domain):
        self.variables = self.parse_domain(domain)
        self.model().set_orig_variables(self.variables)

    def reset_domain(self):
        self.model().reset_variables()
        self.variables = self.model().variables

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            index = self.indexAt(event.pos())
            self.edit(index)
            event.accept()
        else:
            super().mousePressEvent(event)

    @staticmethod
    def parse_domain(domain):
        """Convert domain into variable representation used by
        the VarTableModel.

        Parameters
        ----------
        domain : the domain to convert

        Returns
        -------
        list of [variable_name, var_type, place, values, can_be_numeric] lists.

        """
        if domain is None:
            return []

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
            result = ", ".join(str(v)
                               for v in value_list[:VarTableModel.DISCRETE_VALUE_DISPLAY_LIMIT])
            if len(value_list) > VarTableModel.DISCRETE_VALUE_DISPLAY_LIMIT:
                result += ", ..."
            return result

        return [
            [var.name, type(var), place,
             discrete_value_display(var.values) if var.is_discrete else "",
             may_be_numeric(var)]
            for place, vars in enumerate(
                (domain.attributes, domain.class_vars, domain.metas))
            for var in vars
        ]
