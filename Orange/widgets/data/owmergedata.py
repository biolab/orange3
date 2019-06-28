from collections import namedtuple
from itertools import chain, product

from AnyQt.QtCore import pyqtSignal as Signal
from AnyQt.QtWidgets import QApplication, QStyle, QWidget, \
    QLabel, QComboBox, QPushButton, QVBoxLayout, QHBoxLayout

import numpy as np

import Orange
from Orange.data import StringVariable, ContinuousVariable, Variable
from Orange.data.util import hstack
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output


INSTANCEID = "Source position (index)"
INDEX = "Position (index)"


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
        self.layout().setSpacing(0)
        self.setMouseTracking(True)

    def add_row(self, value_left=None, value_right=None):
        def get_combo(model, value):
            combo = QComboBox(self)
            combo.setModel(model)
            if value is not None:
                combo.setCurrentIndex(model.indexOf(value))
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
            get_combo(self.model_left, value_left),
            QLabel(self.in_label),
            get_combo(self.model_right, value_right),
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
        self.rows[0].pre_label.setText(self.pre_label)
        self.rows[0].remove_button.setText("× "[len(self.rows) == 1])
        self.rows[-1].add_button.setText("+")
        if len(self.rows) > 1:
            self.rows[-2].add_button.setText(" ")

    def current_state(self):
        def get_var(model, combo):
            index = combo.currentIndex()
            if 0 <= index < len(model):
                return model[index]
            else:
                return None

        return [(get_var(self.model_left, row.left_combo),
                 get_var(self.model_right, row.right_combo))
                for row in self.rows]

    def set_state(self, values):
        while len(self.rows) > len(values):
            self.remove_row()
        while len(self.rows) < len(values):
            self.add_row()
        for (val_left, val_right), row in zip(values, self.rows):
            row.left_combo.setCurrentIndex(self.model_left.indexOf(val_left))
            row.right_combo.setCurrentIndex(self.model_right.indexOf(val_right))

    def emit_list(self):
        self.vars_changed.emit(self.current_state())


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

    # TODO: Context migration
    attr_pairs = settings.Setting('', schema_only=True)
    merging = settings.Setting(LeftJoin)
    auto_apply = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False

    class Warning(widget.OWWidget.Warning):
        duplicate_names = widget.Msg("Duplicate variable names in output.")
        non_unique_variables = widget.Msg(
            "Columns with non-unique values can't be used for matching.\n"
            "The second variable must uniquely define the row in the second "
            "table. Variables with non-unique values ({}) "
            "are thus unsuitable as keys for merging."
        )

    def __init__(self):
        super().__init__()

        self.data = None
        self.extra_data = None

        self.model = itemmodels.VariableListModel()
        self.model_unique_with_id = itemmodels.VariableListModel()
        self.extra_model_unique = itemmodels.VariableListModel()
        self.extra_model_unique_with_id = itemmodels.VariableListModel()

        box = gui.hBox(self.controlArea, box=None)
        self.infoBoxData = gui.label(
            box, self, self.dataInfoText(None), box="Data")
        self.infoBoxExtraData = gui.label(
            box, self, self.dataInfoText(None), box="Extra Data")

        grp = gui.radioButtonsInBox(
            self.controlArea, self, "merging", box="Merging",
            callback=self.change_merging)
        self.attr_boxes = []

        radio_width = \
            QApplication.style().pixelMetric(QStyle.PM_ExclusiveIndicatorWidth)

        def add_option(label, pre_label, in_label, model, extra_model):
            gui.appendRadioButton(grp, label)
            vbox = gui.vBox(grp)
            box = ConditionBox(vbox, model, extra_model, pre_label, in_label)
            box.add_row()
            vbox.layout().addWidget(box)
            self.attr_boxes.append(box)
            box.vars_changed.connect(self._invalidate)

        add_option("Append columns from Extra Data", "by matching", "with",
                   self.model, self.extra_model_unique)
        add_option("Find matching rows", "where", "equals",
                   self.model_unique_with_id, self.extra_model_unique_with_id)
        add_option("Concatenate tables, merge rows", "where", "equals",
                   self.model_unique_with_id, self.extra_model_unique_with_id)
        self.set_merging()
        gui.auto_commit(self.controlArea, self, "auto_apply", "&Apply",
                        box=False)

        self.settingsAboutToBePacked.connect(self.store_combo_state)

    def store_combo_state(self):
        self.attr_pairs = self.boxes[self.merging].current_state()

    def set_merging(self):
        # pylint: disable=invalid-sequence-index
        # all boxes should be hidden before one is shown, otherwise widget's
        # layout changes height
        for box in self.attr_boxes:
            box.hide()
        self.attr_boxes[self.merging].show()

    def change_merging(self):
        self.set_merging()
        self.commit()

    def _set_unique_model(self, data, model):
        self.Warning.non_unique_variables.clear()
        if data is None:
            model[:] = []
            return
        m = [INDEX]
        non_unique = []
        for attr in chain(data.domain.variables, data.domain.metas):
            col = data.get_column_view(attr)[0]
            if attr.is_primitive():
                col = col.astype(float)
                col = col[~np.isnan(col)]
            else:
                col = col[~(col == "")]
            if len(np.unique(col)) == len(col):
                m.append(attr)
            else:
                non_unique.append(attr.name)
        if non_unique:
            self.Warning.non_unique_variables(", ".join(non_unique))
        model[:] = m

    @staticmethod
    def _set_model(data, model):
        if data is None:
            model[:] = []
            return
        model[:] = list(chain([INDEX], data.domain.variables, data.domain.metas))

    def _add_instanceid_to_models(self):
        needs_id = self.data is not None and self.extra_data is not None and \
            len(np.intersect1d(self.data.ids, self.extra_data.ids))
        for model in (self.model_unique_with_id,
                      self.extra_model_unique_with_id):
            has_id = INSTANCEID in model
            if needs_id and not has_id:
                model.insert(0, INSTANCEID)
            elif not needs_id and has_id:
                model.remove(INSTANCEID)

    def _restore_combo_current_items(self, side, prev_settings):
        for box, box_vars in zip(self.attr_boxes, prev_settings):
            for row, vars in zip(box.rows, box_vars):
                self._try_set_combo(
                    [row.left_combo, row.right_combo][side], vars[side])

    @staticmethod
    def _try_set_combo(combo, var):
        if var in combo.model():
            combo.setCurrentIndex(combo.model().indexOf(var))
        else:
            combo.setCurrentIndex(0)

    def _clean_repeated_combos(self):
        # This remove rows with combos with default values after domain change
        for box in self.attr_boxes:
            state = box.current_state()
            for i in range(len(box.rows) - 1, 0, -1):
                if state[i] in state[:i]:
                    box.remove_row(box.rows[i])

    def _find_best_match(self):
        def get_unique_str_metas_names(model_):
            return [m for m in model_ if isinstance(m, StringVariable)]

        def best_match(model, extra_model):
            attr, extra_attr, n_max_intersect = INDEX, INDEX, 0
            str_metas = get_unique_str_metas_names(model)
            extra_str_metas = get_unique_str_metas_names(extra_model)
            for m_a, m_b in product(str_metas, extra_str_metas):
                n_inter = len(np.intersect1d(self.data[:, m_a].metas,
                                             self.extra_data[:, m_b].metas))
                if n_inter > n_max_intersect:
                    n_max_intersect, attr, extra_attr = n_inter, m_a, m_b
            return attr, extra_attr

        if self.data and self.extra_data:
            for box in self.attr_boxes:
                state = box.current_state()
                if len(state) == 1 \
                        and not any(isinstance(v, Variable) for v in state[0]):
                    l_var, r_var = best_match(box.model_left, box.model_right)
                    self._try_set_combo(box.rows[0].left_combo, l_var)
                    self._try_set_combo(box.rows[0].right_combo, r_var)

    @Inputs.data
    @check_sql_input
    def setData(self, data):
        self.data = data
        prev_settings = [box.current_state() for box in self.attr_boxes]
        self._set_model(data, self.model)
        self._set_unique_model(data, self.model_unique_with_id)
        self._add_instanceid_to_models()
        self._restore_combo_current_items(0, prev_settings)
        self.infoBoxData.setText(self.dataInfoText(data))

    @Inputs.extra_data
    @check_sql_input
    def setExtraData(self, data):
        self.extra_data = data
        prev_settings = [box.current_state() for box in self.attr_boxes]
        self._set_unique_model(data, self.extra_model_unique)
        self._set_unique_model(data, self.extra_model_unique_with_id)
        self._add_instanceid_to_models()
        self._restore_combo_current_items(1, prev_settings)
        self.infoBoxExtraData.setText(self.dataInfoText(data))

    def handleNewSignals(self):
        if self.attr_pairs:
            self.boxes[self.merging].set_current_state(self.attr_pairs)
            # This is schema-only setting, so it should be single-shot
            # More complicated alternative: make it a context setting
            self.attr_pairs = []
        self._find_best_match()
        self._clean_repeated_combos()
        self._invalidate()

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
        attr = (self.attr_augment_data, self.attr_merge_data,
                self.attr_combine_data)
        extra_attr = (self.attr_augment_extra, self.attr_merge_extra,
                      self.attr_combine_extra)
        merging_types = ("Append columns from Extra Data", "Find matching rows",
                         "Concatenate tables, merge rows")
        self.report_items((
            ("Merging", merging_types[self.merging]),
            ("Data attribute", attr[self.merging]),
            ("Extra data attribute", extra_attr[self.merging])))

    def merge(self):
        # pylint: disable=invalid-sequence-index
        pairs = self.attr_boxes[self.merging].current_state()
        vars, extra_vars = zip(*pairs)
        as_string = [not all(isinstance(v, ContinuousVariable) for v in pair)
                     for pair in pairs]
        merge_method = [
            self._augment_indices, self._merge_indices, self._combine_indices][
            self.merging]

        extra_map = self._get_keymap(self.extra_data, extra_vars, as_string)
        match_indices = merge_method(vars, extra_map, as_string)
        reduced_extra_data = self._compute_reduced_extra_data(extra_vars)
        return self._join_table_by_indices(reduced_extra_data, match_indices)

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
    def _values(data, var, as_string):
        """Return an iterotor over keys for rows of the table."""
        if var == INSTANCEID:
            return (inst.id for inst in data)
        if var == INDEX:
            return range(len(data))
        col = data.get_column_view(var)[0]
        if not as_string:
            return col
        if var.is_primitive():
            return (var.str_val(val) if not np.isnan(val) else np.nan
                    for val in col)
        else:
            return (str(val) if val else np.nan for val in col)

    @classmethod
    def _get_keymap(cls, data, vars, as_strings):
        """Return a generator of pairs (key, index) by enumerating and
        switching the values for rows (method `_values`).
        """
        vals_combinations = zip(*(cls._values(data, var, as_string)
                                  for var, as_string in zip(vars, as_strings)))
        return ((val, i) for i, val in enumerate(vals_combinations))

    @classmethod
    def _get_values(cls, data, vars, as_strings):
        return zip(*(cls._values(data, var, as_string)
                     for var, as_string in zip(vars, as_strings)))

    def _augment_indices(self, vars, extra_map, as_strings):
        """Compute a two-row array of indices:
        - the first row contains indices for the primary table,
        - the second row contains the matching rows in the extra table or -1"""
        data = self.data
        # Don't match nans. This is needed since numpy supports using nan as
        # keys. If numpy fixes this, the below conditions will always be false,
        # so we're OK again.
        extra_map = {val: i for val, i in extra_map if all(x == x for x in val)}
        keys = (extra_map.get(val, -1)
                for val in self._get_values(data, vars, as_strings))
        return np.vstack((np.arange(len(data), dtype=np.int64),
                          np.fromiter(keys, dtype=np.int64, count=len(data))))

    def _merge_indices(self, vars, extra_map, as_strings):
        """Use _augment_indices to compute the array of indices,
        then remove those with no match in the second table"""
        augmented = self._augment_indices(vars, extra_map, as_strings)
        return augmented[:, augmented[1] != -1]

    def _combine_indices(self, vars, extra_map, as_strings):
        """Use _augment_indices to compute the array of indices,
        then add rows in the second table without a match in the first"""
        extra_map = list(extra_map)  # we need it twice; tee would make a copy
        # dict instead of set because we have pairs; we'll need only keys
        key_map = {val
                   for val, _ in self._get_keymap(self.data, vars, as_strings)
                   if all(x == x for x in val)}
        keys = np.fromiter((j for key, j in extra_map if key not in key_map),
                           dtype=np.int64)
        right_indices = np.vstack((np.full(len(keys), -1, np.int64), keys))
        return np.hstack(
            (self._augment_indices(vars, extra_map, as_strings),
             right_indices))

    def _join_table_by_indices(self, reduced_extra, indices):
        """Join (horizontally) self.data and reduced_extra, taking the pairs
        of rows given in indices"""
        if not indices.size:
            return None
        domain = Orange.data.Domain(
            *(getattr(self.data.domain, x) + getattr(reduced_extra.domain, x)
              for x in ("attributes", "class_vars", "metas")))
        X = self._join_array_by_indices(self.data.X, reduced_extra.X, indices)
        Y = self._join_array_by_indices(
            np.c_[self.data.Y], np.c_[reduced_extra.Y], indices)
        string_cols = [i for i, var in enumerate(domain.metas) if var.is_string]
        metas = self._join_array_by_indices(
            self.data.metas, reduced_extra.metas, indices, string_cols)
        table = Orange.data.Table.from_numpy(domain, X, Y, metas)
        table.name = getattr(self.data, 'name', '')
        table.attributes = getattr(self.data, 'attributes', {})
        table.ids = self.data.ids
        return table

    @staticmethod
    def _join_array_by_indices(left, right, indices, string_cols=None):
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
        res = hstack((prepare(left, indices[0], str_left),
                      prepare(right, indices[1], str_right)))
        return res


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWMergeData).run(
        setData=Orange.data.Table("tests/data-gender-region"),
        setExtraData=Orange.data.Table("tests/data-regions"))
