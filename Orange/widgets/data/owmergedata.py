import math
import itertools
from collections import defaultdict
from itertools import chain

from AnyQt.QtWidgets import QWidget, QGridLayout, QApplication, QStyle
from AnyQt.QtCore import Qt
import numpy as np

import Orange
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.sql import check_sql_input


class OWMergeData(widget.OWWidget):
    name = "Merge Data"
    description = "Merge data sets based on the values of selected data features."
    icon = "icons/MergeData.svg"
    priority = 1110

    inputs = [("Data", Orange.data.Table, "setData", widget.Default),
              ("Extra Data", Orange.data.Table, "setExtraData")]
    outputs = [widget.OutputSignal(
        "Data", Orange.data.Table,
        replaces=["Merged Data A+B", "Merged Data B+A"])]

    attr_augment_data = settings.Setting('', schema_only=True)
    attr_augment_extra = settings.Setting('', schema_only=True)
    attr_merge_data = settings.Setting('', schema_only=True)
    attr_merge_extra = settings.Setting('', schema_only=True)
    attr_combine_data = settings.Setting('', schema_only=True)
    attr_combine_extra = settings.Setting('', schema_only=True)
    merging = settings.Setting(0)


    want_main_area = False

    def __init__(self):
        super().__init__()

        self.data = None
        self.extra_data = None
        self.extra_data = None

        self.model = itemmodels.VariableListModel()
        self.model_unique = itemmodels.VariableListModel()
        self.extra_model_unique = itemmodels.VariableListModel()

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
            gui.comboBox(box, self, 'attr_{}_data'.format(merge_type),
                         callback=self._invalidate, model=model)
            gui.widgetLabel(box, between_label)
            gui.comboBox(box, self, 'attr_{}_extra'.format(merge_type),
                         callback=self._invalidate, model=extra_model)
            vbox.layout().addSpacing(6)

        add_option("Append columns from Extra Data",
                   "by matching", "with", "augment",
                   self.model, self.extra_model_unique)
        add_option("Find matching rows", "where",
                   "equals", "merge",
                   self.model_unique, self.extra_model_unique)
        add_option("Concatenate tables, merge rows",
                   "where", "equals", "combine",
                   self.model_unique, self.extra_model_unique)
        self.set_merging()

    def set_merging(self):
        for i, box in enumerate(self.attr_boxes):
            if self.merging == i:
                box.show()
            else:
                box.hide()

    def change_merging(self):
        self.set_merging()

    @staticmethod
    def _set_unique_model(data, model):
        m = ["Position (index)"]
        for attr in data.domain:
            col = data.get_column_view(attr)[0]
            print(col)
            col = col[~np.isnan(col)]
            if len(np.unique(col)) == len(col):
                m.append(attr)
        # TODO: handle unknowns in metas, sparse data...
        for attr in data.domain.metas:
            col = data.get_column_view(attr)[0]
            if len(np.unique(col)) == len(col):
                m.append(attr)
        model[:] = m

    @staticmethod
    def _set_model(data, model):
        model[:] = list(chain(data.domain, data.domain.metas))

    @check_sql_input
    def setData(self, data):
        self.data = data
        self._set_model(data, self.model)
        self._set_unique_model(data, self.model_unique)
        self.infoBoxData.setText(self.dataInfoText(data))
        if len(self.model_unique) > 1:
            attr_merge_data = self.model_unique[1]
            attr_combine_data = self.model_unique[1]

    @check_sql_input
    def setExtraData(self, data):
        self.extra_data = data
        self._set_unique_model(data, self.extra_model_unique)
        self.infoBoxExtraData.setText(self.dataInfoText(data))
        if len(self.extra_model_unique) > 1:
            attr_augment_extra = attr_merge_extra = attr_combine_extra = \
                self.extra_model_unique[1]

    def handleNewSignals(self):
        self._invalidate()

    def dataInfoText(self, data):
        if data is None:
            return "No data."
        else:
            return "{}\n{} instances\n{} variables".format(
                data.name, len(data), len(data.domain) + len(data.domain.metas))

    def commit(self):
        return

    def _invalidate(self):
        self.commit()

    def send_report(self):
        attr_a = None
        attr_b = None
        if self.dataA is not None:
            attr_a = self.attr_a
            if attr_a in self.dataA.domain:
                attr_a = self.dataA.domain[attr_a]
        if self.dataB is not None:
            attr_b = self.attr_b
            if attr_b in self.dataB.domain:
                attr_b = self.dataB.domain[attr_b]
        self.report_items((
            ("Attribute A", attr_a),
            ("Attribute B", attr_b),
        ))


def merge(A, varA, B, varB, inner=True):
    join_indices = inner_join_indices(A, B, varA, varB) if inner else \
        outer_join_indices(A, B, varA, varB)
    seen_set = set()

    def seen(val):
        return (val in seen_set or bool(seen_set.add(val))) and val is not None

    merge_indices = [(i, j) for i, j in join_indices if not seen(i)]

    all_vars = set(A.domain.variables + A.domain.metas)
    if inner:
        all_vars.add(varB)

    iter_vars_B = itertools.chain(
        enumerate(B.domain.variables),
        ((-i, m) for i, m in enumerate(B.domain.metas, start=1))
    )
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


def inner_join_indices(table1, table2, var1, var2):
    key_map1 = group_table_indices(table1, var1, True)
    key_map2 = group_table_indices(table2, var2, True)
    indices = []
    for i, inst in enumerate(table1):
        key = str(inst.id if var1 == INSTANCEID
                  else i if var1 == INDEX else inst[var1])
        if key in key_map1 and key in key_map2:
            for j in key_map2[key]:
                indices.append((i, j))
    return indices


def outer_join_indices(table1, table2, var1, var2):
    key_map1 = group_table_indices(table1, var1, True)
    key_map2 = group_table_indices(table2, var2, True)
    indices = []

    def get_key(var):
        # local function due to better performance
        return str(inst.id if var == INSTANCEID
                   else i if var == INDEX else inst[var])

    for i, inst in enumerate(table1):
        key = get_key(var1)
        if key in key_map1 and key in key_map2:
            for j in key_map2[key]:
                indices.append((i, j))
        else:
            indices.append((i, None))
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

    return Orange.data.Table.from_np(domain, X, Y, metas)


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
    from AnyQt.QtWidgets import QApplication
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
