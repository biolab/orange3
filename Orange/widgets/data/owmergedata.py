from enum import IntEnum
import math
from collections import defaultdict
from itertools import chain, product

from AnyQt.QtWidgets import QApplication, QStyle, QSizePolicy

import numpy as np

import Orange
from Orange.data import StringVariable
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.sql import check_sql_input

INSTANCEID = "Source position (index)"
INDEX = "Position (index)"


class MergeType(IntEnum):
    LEFT_JOIN, INNER_JOIN, OUTER_JOIN = 0, 1, 2


class OWMergeData(widget.OWWidget):
    name = "Merge Data"
    description = "Merge data sets based on the values of selected data features."
    icon = "icons/MergeData.svg"
    priority = 1110

    inputs = [widget.InputSignal("Data", Orange.data.Table, "setData",
                                 widget.Default, replaces=["Data A"]),
              widget.InputSignal("Extra Data", Orange.data.Table,
                                 "setExtraData", replaces=["Data B"])]
    outputs = [widget.OutputSignal(
        "Data", Orange.data.Table,
        replaces=["Merged Data A+B", "Merged Data B+A", "Merged Data"])]

    attr_augment_data = settings.Setting('', schema_only=True)
    attr_augment_extra = settings.Setting('', schema_only=True)
    attr_merge_data = settings.Setting('', schema_only=True)
    attr_merge_extra = settings.Setting('', schema_only=True)
    attr_combine_data = settings.Setting('', schema_only=True)
    attr_combine_extra = settings.Setting('', schema_only=True)
    merging = settings.Setting(0)

    want_main_area = False

    class Warning(widget.OWWidget.Warning):
        duplicate_names = widget.Msg("Duplicate variable names.")

    def __init__(self):
        super().__init__()

        self.data = None
        self.extra_data = None
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

        def add_option(label, pre_label, between_label,
                       merge_type, model, extra_model):
            gui.appendRadioButton(grp, label)
            vbox = gui.vBox(grp)
            box = gui.hBox(vbox)
            box.layout().addSpacing(radio_width)
            self.attr_boxes.append(box)
            gui.widgetLabel(box, pre_label)
            model[:] = [getattr(self, 'attr_{}_data'.format(merge_type))]
            extra_model[:] = [getattr(self, 'attr_{}_extra'.format(merge_type))]
            cb = gui.comboBox(box, self, 'attr_{}_data'.format(merge_type),
                              callback=self._invalidate, model=model)
            cb.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
            gui.widgetLabel(box, between_label)
            cb = gui.comboBox(box, self, 'attr_{}_extra'.format(merge_type),
                              callback=self._invalidate, model=extra_model)
            cb.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
            vbox.layout().addSpacing(6)

        add_option("Append columns from Extra Data",
                   "by matching", "with", "augment",
                   self.model, self.extra_model_unique)
        add_option("Find matching rows", "where",
                   "equals", "merge",
                   self.model_unique_with_id, self.extra_model_unique_with_id)
        add_option("Concatenate tables, merge rows",
                   "where", "equals", "combine",
                   self.model_unique_with_id, self.extra_model_unique_with_id)
        self.set_merging()

    def set_merging(self):
        # all boxes should be hidden before one is shown, otherwise widget's
        # layout changes height
        for box in self.attr_boxes:
            box.hide()
        self.attr_boxes[int(self.merging)].show()

    def change_merging(self):
        self.set_merging()
        self._invalidate()

    @staticmethod
    def _set_unique_model(data, model):
        if data is None:
            model[:] = []
            return
        m = [INDEX]
        for attr in data.domain:
            col = data.get_column_view(attr)[0]
            col = col[~np.isnan(col)]
            if len(np.unique(col)) == len(col):
                m.append(attr)
        # TODO: sparse data...
        for attr in data.domain.metas:
            col = data.get_column_view(attr)[0]
            if attr.is_primitive():
                col = col.astype(float)
                col = col[~np.isnan(col)]
            else:
                col = col[~(col == "")]
            if len(np.unique(col)) == len(col):
                m.append(attr)
        model[:] = m

    @staticmethod
    def _set_model(data, model):
        if data is None:
            model[:] = []
            return
        model[:] = list(chain([INDEX], data.domain, data.domain.metas))

    def _add_instanceid_to_models(self):
        needs_id = self.data is not None and self.extra_data is not None and \
                        len(np.intersect1d(self.data.ids, self.extra_data.ids))
        for model in (self.model_unique_with_id,
                      self.extra_model_unique_with_id):
            has_id = model and model[0] == INSTANCEID
            if needs_id and not has_id:
                model.insert(0, INSTANCEID)
            elif not needs_id and has_id:
                del model[0]

    def _init_combo_current_items(self, variables, models):
        for var, model in zip(variables, models):
            value = getattr(self, var)
            if len(model) > 0:
                setattr(self, var, value if value in model else INDEX)

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

        def set_attrs(attr_name, attr_extra_name, attr, extra_attr):
            if getattr(self, attr_name) == INDEX and \
                    getattr(self, attr_extra_name) == INDEX:
                setattr(self, attr_name, attr)
                setattr(self, attr_extra_name, extra_attr)

        if self.data and self.extra_data:
            attrs = best_match(self.model, self.extra_model_unique)
            set_attrs("attr_augment_data", "attr_augment_extra", *attrs)
            attrs = best_match(self.model_unique_with_id,
                               self.extra_model_unique_with_id)
            set_attrs("attr_merge_data", "attr_merge_extra", *attrs)
            set_attrs("attr_combine_data", "attr_combine_extra", *attrs)

    @check_sql_input
    def setData(self, data):
        self.data = data
        self._set_model(data, self.model)
        self._set_unique_model(data, self.model_unique_with_id)
        self._add_instanceid_to_models()
        self._init_combo_current_items(
            ("attr_augment_data", "attr_merge_data", "attr_combine_data"),
            (self.model, self.model_unique_with_id, self.model_unique_with_id))
        self.infoBoxData.setText(self.dataInfoText(data))
        self._find_best_match()

    @check_sql_input
    def setExtraData(self, data):
        self.extra_data = data
        self._set_unique_model(data, self.extra_model_unique)
        self._set_unique_model(data, self.extra_model_unique_with_id)
        self._add_instanceid_to_models()
        self._init_combo_current_items(
            ("attr_augment_extra", "attr_merge_extra", "attr_combine_extra"),
            (self.extra_model_unique, self.extra_model_unique_with_id,
             self.extra_model_unique_with_id))
        self.infoBoxExtraData.setText(self.dataInfoText(data))
        self._find_best_match()

    def handleNewSignals(self):
        self._invalidate()

    def dataInfoText(self, data):
        if data is None:
            return "No data."
        else:
            return "{}\n{} instances\n{} variables".format(
                data.name, len(data), len(data.domain) + len(data.domain.metas))

    def commit(self):
        self.Warning.duplicate_names.clear()
        merged_data = None
        if self.data is not None and self.extra_data is not None:
            if self.merging == MergeType.LEFT_JOIN:
                var_data = self.attr_augment_data
                var_extra_data = self.attr_augment_extra
            elif self.merging == MergeType.INNER_JOIN:
                var_data = self.attr_merge_data
                var_extra_data = self.attr_merge_extra
            else:
                var_data = self.attr_combine_data
                var_extra_data = self.attr_combine_extra
            merged_data = merge(self.data, var_data, self.extra_data,
                                var_extra_data, self.merging)
            if merged_data:
                var_names = [var.name for var in merged_data.domain.variables +
                             merged_data.domain.metas]
                if len(np.unique(var_names)) != len(var_names):
                    self.Warning.duplicate_names()
        self.send("Data", merged_data)

    def _invalidate(self):
        self.commit()

    def send_report(self):
        attr = (self.attr_augment_data, self.attr_merge_data,
                self.attr_combine_data)
        extra_attr = (self.attr_augment_extra, self.attr_merge_extra,
                      self.attr_combine_extra)
        merging_types = ("Append columns from Extra Data", "Find matching rows",
                         "Concatenate tables, merge rows")
        self.report_items((
            ("Merging", merging_types[int(self.merging)]),
            ("Data attribute", attr[int(self.merging)]),
            ("Extra data attribute", extra_attr[int(self.merging)])))


def merge(A, varA, B, varB, merge_type):
    indices = join_indices(A, B, varA, varB, merge_type)
    seen_set = set()

    def seen(val):
        return (val in seen_set or bool(seen_set.add(val))) and val is not None

    merge_indices = [(i, j) for i, j in indices if not seen(i)]

    all_vars = set(A.domain.variables + A.domain.metas)
    if merge_type != MergeType.OUTER_JOIN:
        all_vars.add(varB)

    iter_vars_B = chain(
        enumerate(B.domain.variables),
        ((-i, m) for i, m in enumerate(B.domain.metas, start=1)))
    reduced_B = B[:, [i for i, var in iter_vars_B if var not in all_vars]]

    return join_table_by_indices(A, reduced_B, merge_indices)


def group_table_indices(table, key_var, exclude_unknown=False):
    """
    Group table indices based on values of selected columns (`key_vars`).

    Return a dictionary mapping all unique value combinations (keys)
    into a list of indices in the table where they are present.

    :param Orange.data.Table table:
    :param Orange.data.FeatureDescriptor] key_var:
    :param bool exclude_unknown:

    """
    groups = defaultdict(list)
    for i, inst in enumerate(table):
        key = inst.id if key_var == INSTANCEID else i if \
            key_var == INDEX else inst[key_var]
        if exclude_unknown and math.isnan(key):
            continue
        groups[str(key)].append(i)
    return groups


def join_indices(table1, table2, var1, var2, join_type):
    key_map1 = group_table_indices(table1, var1, True)
    key_map2 = group_table_indices(table2, var2, True)
    indices = []

    def get_key(var):
        return str(inst.id if var == INSTANCEID
                   else i if var == INDEX else inst[var])

    for i, inst in enumerate(table1):
        key = get_key(var1)
        if key in key_map1 and key in key_map2:
            for j in key_map2[key]:
                indices.append((i, j))
        elif join_type != MergeType.INNER_JOIN:
            indices.append((i, None))

    if join_type != MergeType.OUTER_JOIN:
        return indices

    for i, inst in enumerate(table2):
        key = get_key(var2)
        if not (key in key_map1 and key in key_map2):
            indices.append((None, i))
    return indices


def join_table_by_indices(left, right, indices):
    if not indices:
        return None
    domain = Orange.data.Domain(
        left.domain.attributes + right.domain.attributes,
        left.domain.class_vars + right.domain.class_vars,
        left.domain.metas + right.domain.metas
    )
    X = join_array_by_indices(left.X, right.X, indices)
    Y = join_array_by_indices(np.c_[left.Y], np.c_[right.Y], indices)
    metas = join_array_by_indices(left.metas, right.metas, indices)
    for col, var in enumerate(domain.metas):
        if var.is_string:
            for row in range(metas.shape[0]):
                cell = metas[row, col]
                if isinstance(cell, float) and np.isnan(cell):
                    metas[row, col] = ""

    return Orange.data.Table.from_numpy(domain, X, Y, metas)


def join_array_by_indices(left, right, indices, masked=float("nan")):
    left_masked = [masked] * left.shape[1]
    right_masked = [masked] * right.shape[1]

    leftparts = []
    rightparts = []
    for i, j in indices:
        if i is not None:
            leftparts.append(left[i])
        else:
            leftparts.append(left_masked)
        if j is not None:
            rightparts.append(right[j])
        else:
            rightparts.append(right_masked)

    def hstack_blocks(blocks):
        return np.hstack(list(map(np.vstack, blocks)))

    if left.shape[1] and left.dtype == object:
        leftparts = np.array(leftparts).astype(object)
    if right.shape[1] and right.dtype == object:
        rightparts = np.array(rightparts).astype(object)
    return hstack_blocks((leftparts, rightparts))


def test():
    app = QApplication([])
    w = OWMergeData()
    data = Orange.data.Table("tests/data-gender-region")
    extra_data = Orange.data.Table("tests/data-regions")
    w.setData(data)
    w.setExtraData(extra_data)
    w.handleNewSignals()
    w.show()
    app.exec_()


if __name__ == "__main__":
    test()
