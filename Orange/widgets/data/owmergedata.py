from collections import namedtuple
from itertools import chain, product

import numpy as np

from AnyQt.QtCore import Qt, QModelIndex, pyqtSignal as Signal
from AnyQt.QtWidgets import QWidget, QLabel, QComboBox, QPushButton, \
    QVBoxLayout, QHBoxLayout

import Orange
from Orange.data import StringVariable, ContinuousVariable, Variable
from Orange.data.util import hstack
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output, Msg

INSTANCEID = "Instance id"
INDEX = "Row index"


class ConditionBox(QWidget):
    vars_changed = Signal(list)

    RowItems = namedtuple(
        "RowItems",
        ("pre_label", "left_combo", "in_label", "right_combo",
         "remove_button", "add_button"))

    def __init__(self, parent, model_left, model_right, pre_label, in_label):
        super().__init__(parent)
        self.model_left = model_left
        self.model_right = model_right
        self.pre_label, self.in_label = pre_label, in_label
        self.rows = []
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        self.setMouseTracking(True)

    def add_row(self):
        def sync_combos():
            combo = self.sender()
            index = combo.currentIndex()
            model = combo.model()

            other = ({row_items.left_combo, row_items.right_combo}
                     - {combo}).pop()
            other_index = other.currentIndex()
            other_model = other.model()

            if 0 <= index < len(combo.model()):
                var = model[index]
                if isinstance(var, str):
                    other.setCurrentText(model[index])
                elif isinstance(other_model[other_index], str):
                    for other_var in other_model:
                        if isinstance(other_var, Variable) \
                                and var.name == other_var.name \
                                and (type(other_var) is type(var)
                                     or (not var.is_continuous
                                         and not other_var.is_continuous)):
                            other.setCurrentText(var.name)
                            break

            self.emit_list()

        def get_combo(model):
            combo = QComboBox(self)
            combo.setModel(model)
            # We use signal activated because it is triggered only on user
            # interaction, not programmatically.
            combo.activated.connect(sync_combos)
            return combo

        def get_button(label, callback):
            button = QPushButton(label, self)
            button.setFlat(True)
            button.setFixedWidth(12)
            button.clicked.connect(callback)
            return button

        row = self.layout().count()
        row_items = self.RowItems(
            QLabel("and" if row else self.pre_label),
            get_combo(self.model_left),
            QLabel(self.in_label),
            get_combo(self.model_right),
            get_button("×", self.on_remove_row),
            get_button("+", self.on_add_row)
        )
        layout = QHBoxLayout()
        layout.setSpacing(10)
        self.layout().addLayout(layout)
        layout.addStretch(10)
        for item in row_items:
            layout.addWidget(item)
        self.rows.append(row_items)
        self._reset_buttons()

    def remove_row(self, row):
        self.layout().takeAt(self.rows.index(row))
        self.rows.remove(row)
        for item in row:
            item.deleteLater()
        self._reset_buttons()

    def on_add_row(self, _):
        self.add_row()
        self.emit_list()

    def on_remove_row(self):
        if len(self.rows) == 1:
            return
        button = self.sender()
        for row in self.rows:
            if button is row.remove_button:
                self.remove_row(row)
                break
        self.emit_list()

    def _reset_buttons(self):
        def endis(button, enable, text):
            button.setEnabled(enable)
            button.setText(text if enable else "")

        self.rows[0].pre_label.setText(self.pre_label)
        single_row = len(self.rows) == 1
        endis(self.rows[0].remove_button, not single_row, "×")
        endis(self.rows[-1].add_button, True, "+")
        if not single_row:
            endis(self.rows[-2].add_button, False, "")

    def current_state(self):
        def get_var(model, combo):
            return model[combo.currentIndex()]

        return [(get_var(self.model_left, row.left_combo),
                 get_var(self.model_right, row.right_combo))
                for row in self.rows]

    def set_state(self, values):
        while len(self.rows) > len(values):
            self.remove_row(self.rows[0])
        while len(self.rows) < len(values):
            self.add_row()
        for (val_left, val_right), row in zip(values, self.rows):
            row.left_combo.setCurrentIndex(self.model_left.indexOf(val_left))
            row.right_combo.setCurrentIndex(self.model_right.indexOf(val_right))

    def emit_list(self):
        self.vars_changed.emit(self.current_state())


class DomainModelWithTooltips(DomainModel):
    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.ToolTipRole \
                and isinstance(index, QModelIndex) and index.isValid():
            if index.row() == 0:
                return "Match rows sequentially"
            if index.row() == 1:
                return "Re-match rows from tables obtained from the same " \
                       "source,\n" \
                       "e.g. data from the same file that was split within " \
                       "the workflow."
        return super().data(index, role)


class OWMergeData(widget.OWWidget):
    name = "Merge Data"
    description = "Merge datasets based on the values of selected features."
    icon = "icons/MergeData.svg"
    priority = 1110
    keywords = ["join"]

    class Inputs:
        data = Input("Data", Orange.data.Table, default=True, replaces=["Data A"])
        extra_data = Input("Extra Data", Orange.data.Table, replaces=["Data B"])

    class Outputs:
        data = Output("Data",
                      Orange.data.Table,
                      replaces=["Merged Data A+B", "Merged Data B+A", "Merged Data"])

    LeftJoin, InnerJoin, OuterJoin = range(3)
    OptionNames = ("Append columns from Extra data",
                   "Find matching pairs of rows",
                   "Concatenate tables")
    OptionDescriptions = (
        "The first table may contain, for instance, city names,\n"
        "and the second would be a list of cities and their coordinates.\n"
        "Columns with coordinates would then be appended to the output.",

        "Input tables contain different features describing the same data "
        "instances.\n"
        "Output contains matched instances. Rows without matches are removed.",

        "Input tables contain different features describing the same data "
        "instances.\n"
        "Output contains all instances. Data from merged instances is "
        "merged into single rows."
    )

    UserAdviceMessages = [
        widget.Message(
            "Confused about merging options?\nSee the tooltips!",
            "merging_types")]

    attr_pairs = Setting(None, schema_only=True)
    merging = Setting(LeftJoin)
    auto_apply = Setting(True)
    settings_version = 1

    want_main_area = False
    resizing_enabled = False

    class Warning(widget.OWWidget.Warning):
        duplicate_names = Msg("Duplicate variable names in output.")

    class Error(widget.OWWidget.Error):
        matching_numeric_with_nonnum = Msg(
            "Numeric and non-numeric columns ({} and {}) cannot be matched.")
        matching_index_with_sth = Msg("Row index cannot be matched with {}.")
        matching_id_with_sth = Msg("Instance cannot be matched with {}.")
        nonunique_left = Msg(
            "Some combinations of values on the left appear in multiple rows.\n"
            "For this type of merging, every possible combination of values "
            "on the left should appear at most once.")
        nonunique_right = Msg(
            "Some combinations of values on the right appear in multiple rows."
            "\n"
            "Every possible combination of values on the right should appear "
            "at most once.")

    def __init__(self):
        super().__init__()

        self.data = None
        self.extra_data = None

        content = [
            INDEX, INSTANCEID,
            DomainModel.ATTRIBUTES, DomainModel.CLASSES, DomainModel.METAS]
        self.model = DomainModelWithTooltips(content)
        self.extra_model = DomainModelWithTooltips(content)

        box = gui.hBox(self.controlArea, box=None)
        self.infoBoxData = gui.label(
            box, self, self.dataInfoText(None), box="Data")
        self.infoBoxExtraData = gui.label(
            box, self, self.dataInfoText(None), box="Extra Data")

        grp = gui.radioButtons(
            self.controlArea, self, "merging", box="Merging",
            btnLabels=self.OptionNames, tooltips=self.OptionDescriptions,
            callback=self.change_merging)
        grp.layout().setSpacing(8)

        self.attr_boxes = ConditionBox(
            self, self.model, self.extra_model, "", "matches")
        self.attr_boxes.add_row()
        box = gui.vBox(self.controlArea, box="Row matching")
        box.layout().addWidget(self.attr_boxes)

        gui.auto_apply(self.controlArea, self, box=False)
        # connect after wrapping self.commit with gui.auto_commit!
        self.attr_boxes.vars_changed.connect(self.commit)
        self.settingsAboutToBePacked.connect(self.store_combo_state)

    def change_merging(self):
        self.commit()

    @staticmethod
    def _try_set_combo(combo, var):
        if var in combo.model():
            combo.setCurrentIndex(combo.model().indexOf(var))
        else:
            combo.setCurrentIndex(0)

    def _find_best_match(self):
        def get_unique_str_metas_names(model_):
            return [m for m in model_ if isinstance(m, StringVariable)]

        def best_match(model, extra_model):
            attr, extra_attr, n_max_intersect = None, None, 0
            str_metas = get_unique_str_metas_names(model)
            extra_str_metas = get_unique_str_metas_names(extra_model)
            for m_a, m_b in product(str_metas, extra_str_metas):
                col = self.data[:, m_a].metas
                extra_col = self.extra_data[:, m_b].metas
                if col.size and extra_col.size \
                        and isinstance(col[0][0], str) \
                        and isinstance(extra_col[0][0], str):
                    n_inter = len(np.intersect1d(col, extra_col))
                    if n_inter > n_max_intersect:
                        n_max_intersect, attr, extra_attr = n_inter, m_a, m_b
            return attr, extra_attr

        if self.data and self.extra_data:
            box = self.attr_boxes
            state = box.current_state()
            if len(state) == 1 \
                    and not any(isinstance(v, Variable) for v in state[0]):
                l_var, r_var = best_match(box.model_left, box.model_right)
                if l_var is not None:
                    self._try_set_combo(box.rows[0].left_combo, l_var)
                    self._try_set_combo(box.rows[0].right_combo, r_var)

    @Inputs.data
    @check_sql_input
    def setData(self, data):
        self.data = data
        prev_settings = self.attr_boxes.current_state()
        self.model.set_domain(data and data.domain)
        self._restore_combo_current_items(0, prev_settings)
        self.infoBoxData.setText(self.dataInfoText(data))

    @Inputs.extra_data
    @check_sql_input
    def setExtraData(self, data):
        self.extra_data = data
        prev_settings = self.attr_boxes.current_state()
        self.extra_model.set_domain(data and data.domain)
        self._restore_combo_current_items(1, prev_settings)
        self.infoBoxExtraData.setText(self.dataInfoText(data))

    def _restore_combo_current_items(self, side, prev_settings):
        for row, pair in zip(self.attr_boxes.rows, prev_settings):
            self._try_set_combo(
                [row.left_combo, row.right_combo][side], pair[side])

    def store_combo_state(self):
        self.attr_pairs = (
            self.data is not None,
            self.extra_data is not None,
            [[[INDEX, INSTANCEID].index(x) if isinstance(x, str) else x.name
              for x in row]
             for row in self.attr_boxes.current_state()])

    def handleNewSignals(self):
        if self.attr_pairs \
                and self.attr_pairs[:2] == (self.data is not None,
                                            self.extra_data is not None):
            def unpack(x, data):
                if isinstance(x, int):
                    return [INDEX, INSTANCEID][x]
                if data is not None and x in data.domain:
                    return data.domain[x]
                else:
                    return INDEX

            state = [
                [unpack(row[0], self.data), unpack(row[1], self.extra_data)]
                for row in self.attr_pairs[2]]
            self.attr_boxes.set_state(state)
            # This is schema-only setting, so it should be single-shot
            # More complicated alternative: make it a context setting
            self.attr_pairs = None
        self._find_best_match()
        box = self.attr_boxes

        state = box.current_state()
        for i in range(len(box.rows) - 1, 0, -1):
            if state[i] in state[:i]:
                box.remove_row(box.rows[i])

        self.unconditional_commit()

    @staticmethod
    def dataInfoText(data):
        if data is None:
            return "No data."
        else:
            return \
                f"{data.name}\n" \
                f"{len(data)} instances\n" \
                f"{len(data.domain) + len(data.domain.metas)} variables"

    def commit(self):
        self.Error.clear()
        self.Warning.duplicate_names.clear()
        if not self.data or not self.extra_data:
            merged_data = None
        else:
            merged_data = self.merge()
            if merged_data:
                merged_domain = merged_data.domain
                var_names = [var.name for var in chain(merged_domain.variables,
                                                       merged_domain.metas)]
                if len(set(var_names)) != len(var_names):
                    self.Warning.duplicate_names()
        self.Outputs.data.send(merged_data)

    def send_report(self):
        # pylint: disable=invalid-sequence-index
        self.report_items((
            ("Merging", self.OptionNames[self.merging]),
            ("Match",
             ", ".join(
                 f"{self._get_col_name(left)} with {self._get_col_name(right)}"
                 for left, right in self.attr_boxes.current_state()))))

    def merge(self):
        # pylint: disable=invalid-sequence-index
        pairs = self.attr_boxes.current_state()
        if not self._check_pair_types(pairs):
            return None
        left_vars, right_vars = zip(*pairs)
        left_mask = np.full(len(self.data), True)
        left = np.vstack(tuple(self._values(self.data, var, left_mask)
                               for var in left_vars)).T
        right_mask = np.full(len(self.extra_data), True)
        right = np.vstack(tuple(self._values(self.extra_data, var, right_mask)
                                for var in right_vars)).T
        if not self._check_uniqueness(left, left_mask, right, right_mask):
            return None
        method = self._merge_methods[self.merging]
        lefti, righti, rightu = method(self, left, left_mask, right, right_mask)
        reduced_extra_data = self._compute_reduced_extra_data(right_vars)
        return self._join_table_by_indices(
            reduced_extra_data, lefti, righti, rightu)

    def _check_pair_types(self, pairs):
        for left, right in pairs:
            if isinstance(left, ContinuousVariable) \
                    != isinstance(right, ContinuousVariable):
                self.Error.matching_numeric_with_nonnum(left, right)
                return False
            if INDEX in (left, right) and left != right:
                self.Error.matching_index_with_sth(
                    self._get_col_name(({left, right} - {INDEX}).pop()))
                return False
            if INSTANCEID in (left, right) and left != right:
                self.Error.matching_id_with_sth(
                    self._get_col_name(({left, right} - {INSTANCEID}).pop()))
                return False
        return True

    @staticmethod
    def _get_col_name(obj):
        return f"'{obj.name}'" if isinstance(obj, Variable) else obj.lower()

    def _check_uniqueness(self, left, left_mask, right, right_mask):
        ok = True
        masked_right = right[right_mask]
        if len(set(map(tuple, masked_right))) != len(masked_right):
            self.Error.nonunique_right()
            ok = False
        if self.merging != self.LeftJoin:
            masked_left = left[left_mask]
            if len(set(map(tuple, masked_left))) != len(masked_left):
                self.Error.nonunique_left()
                ok = False
        return ok

    def _compute_reduced_extra_data(self, extra_vars):
        """Prepare a table with extra columns that will appear in the merged
        table"""
        domain = self.data.domain
        extra_domain = self.extra_data.domain
        all_vars = set(chain(domain.variables, domain.metas))

        if self.merging != self.OuterJoin:
            all_vars |= set(extra_vars)
        extra_vars = chain(extra_domain.variables, extra_domain.metas)
        return self.extra_data[:, [var for var in extra_vars
                                   if var not in all_vars]]

    @staticmethod
    def _values(data, var, mask):
        """Return an iterotor over keys for rows of the table."""
        if var == INDEX:
            return np.arange(len(data))
        if var == INSTANCEID:
            return np.fromiter(
                (inst.id for inst in data), count=len(data), dtype=np.int)
        col = data.get_column_view(var)[0]
        if var.is_primitive():
            col = col.astype(float, copy=False)
            nans = np.isnan(col)
            mask *= ~nans
            if var.is_discrete:
                col = col.astype(int)
                col[nans] = len(var.values)
                col = np.array(var.values + [np.nan])[col]
        else:
            col = col.copy()
            defined = col.astype(bool)
            mask *= defined
            col[~mask] = np.nan
        return col

    def _left_join_indices(self, left, left_mask, right, right_mask):
        """Compute a two-row array of indices:
        - the first row contains indices for the primary table,
        - the second row contains the matching rows in the extra table or -1"""
        data = self.data
        # Don't match nans. This is needed since numpy may change nan to string
        # nan, so nan's will match each other
        indices = np.arange(len(right))
        indices[~right_mask] = -1
        if right.shape[1] == 1:
            # The more common case can be handled faster
            right_map = dict(zip(right.flatten(), indices))
            righti = (right_map.get(val, -1) for val in left.flatten())
        else:
            right_map = dict(zip(map(tuple, right), indices))
            righti = (right_map.get(tuple(val), -1) for val in left)
        righti = np.fromiter(righti, dtype=np.int64, count=len(data))
        lefti = np.arange(len(data), dtype=np.int64)
        righti[lefti[~left_mask]] = -1
        return lefti, righti, None

    def _inner_join_indices(self, left, left_mask, right, right_mask):
        """Use _augment_indices to compute the array of indices,
        then remove those with no match in the second table"""
        lefti, righti, _ = \
            self._left_join_indices(left, left_mask, right, right_mask)
        mask = righti != [-1]
        return lefti[mask], righti[mask], None

    def _outer_join_indices(self, left, left_mask, right, right_mask):
        """Use _augment_indices to compute the array of indices,
        then add rows in the second table without a match in the first"""
        lefti, righti, _ = \
            self._left_join_indices(left, left_mask, right, right_mask)
        unused = np.full(len(right), True)
        unused[righti] = False
        if len(right) - 1 not in righti:
            # righti can include -1, which sets the last element as used
            unused[-1] = True
        return lefti, righti, np.nonzero(unused)[0]

    _merge_methods = [
        _left_join_indices, _inner_join_indices, _outer_join_indices]

    def _join_table_by_indices(self, reduced_extra, lefti, righti, rightu):
        """Join (horizontally) self.data and reduced_extra, taking the pairs
        of rows given in indices"""
        if not lefti.size:
            return None
        domain = Orange.data.Domain(
            *(getattr(self.data.domain, x) + getattr(reduced_extra.domain, x)
              for x in ("attributes", "class_vars", "metas")))
        X = self._join_array_by_indices(self.data.X, reduced_extra.X, lefti, righti)
        Y = self._join_array_by_indices(
            np.c_[self.data.Y], np.c_[reduced_extra.Y], lefti, righti)
        string_cols = [i for i, var in enumerate(domain.metas) if var.is_string]
        metas = self._join_array_by_indices(
            self.data.metas, reduced_extra.metas, lefti, righti, string_cols)
        if rightu is not None:
            extras = self.extra_data[rightu].transform(domain)
            X = np.vstack((X, extras.X))
            Y = np.vstack((Y, extras.Y))
            metas = np.vstack((metas, extras.metas))
        table = Orange.data.Table.from_numpy(domain, X, Y, metas)
        table.name = getattr(self.data, 'name', '')
        table.attributes = getattr(self.data, 'attributes', {})
        if rightu is not None:
            table.ids = np.hstack(
                (self.data.ids, self.extra_data.ids[rightu]))
        else:
            table.ids = self.data.ids[lefti]

        return table

    @staticmethod
    def _join_array_by_indices(left, right, lefti, righti, string_cols=None):
        """Join (horizontally) two arrays, taking pairs of rows given in indices
        """
        def prepare(arr, inds, str_cols):
            try:
                newarr = arr[inds]
            except IndexError:
                newarr = np.full_like(arr, np.nan)
            else:
                empty = np.full(arr.shape[1], np.nan)
                if str_cols:
                    assert arr.dtype == object
                    empty = empty.astype(object)
                    empty[str_cols] = ''
                newarr[inds == -1] = empty
            return newarr

        left_width = left.shape[1]
        str_left = [i for i in string_cols or () if i < left_width]
        str_right = [i - left_width for i in string_cols or () if i >= left_width]
        res = hstack((prepare(left, lefti, str_left),
                      prepare(right, righti, str_right)))
        return res

    @staticmethod
    def migrate_settings(settings, version=None):
        def mig_value(x):
            if x == "Position (index)":
                return INDEX
            if x == "Source position (index)":
                return INSTANCEID
            return x

        if not version:
            operations = ("augment", "merge", "combine")
            oper = operations[settings["merging"]]
            settings["attr_pairs"] = (
                True, True,
                [(mig_value(settings[f"attr_{oper}_data"]),
                  mig_value(settings[f"attr_{oper}_extra"]))])
            for oper in operations:
                del settings[f"attr_{oper}_data"]
                del settings[f"attr_{oper}_extra"]


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWMergeData).run(
        setData=Orange.data.Table("tests/data-gender-region"),
        setExtraData=Orange.data.Table("tests/data-regions"))
