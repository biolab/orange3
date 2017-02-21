import math
import itertools
from collections import defaultdict

from AnyQt.QtWidgets import QWidget, QGridLayout
from AnyQt.QtCore import Qt
import numpy

import Orange
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.sql import check_sql_input


INSTANCEID = "Source position (index)"
INDEX = "Position (index)"


class OWMergeData(widget.OWWidget):
    name = "Merge Data"
    description = "Merge data sets based on the values of selected data features."
    icon = "icons/MergeData.svg"
    priority = 1110

    inputs = [("Data A", Orange.data.Table, "setDataA", widget.Default),
              ("Data B", Orange.data.Table, "setDataB")]
    outputs = [widget.OutputSignal(
        "Merged Data", Orange.data.Table,
        replaces=["Merged Data A+B", "Merged Data B+A"])]

    attr_a = settings.Setting('', schema_only=True)
    attr_b = settings.Setting('', schema_only=True)
    inner = settings.Setting(True)

    want_main_area = False

    def __init__(self):
        super().__init__()

        # data
        self.dataA = None
        self.dataB = None

        # GUI
        box = gui.hBox(self.controlArea, "Match instances by")

        # attribute A selection
        self.attrViewA = gui.comboBox(box, self, 'attr_a', label="Data A",
                                      orientation=Qt.Vertical,
                                      sendSelectedValue=True,
                                      callback=self._invalidate)
        self.attrModelA = itemmodels.VariableListModel()
        self.attrViewA.setModel(self.attrModelA)

        # attribute  B selection
        self.attrViewB = gui.comboBox(box, self, 'attr_b', label="Data B",
                                      orientation=Qt.Vertical,
                                      sendSelectedValue=True,
                                      callback=self._invalidate)
        self.attrModelB = itemmodels.VariableListModel()
        self.attrViewB.setModel(self.attrModelB)

        # info A
        box = gui.hBox(self.controlArea, box=None)
        self.infoBoxDataA = gui.label(box, self, self.dataInfoText(None),
                                      box="Data A Info")

        # info B
        self.infoBoxDataB = gui.label(box, self, self.dataInfoText(None),
                                      box="Data B Info")

        gui.separator(self.controlArea)
        box = gui.vBox(self.controlArea, box=True)
        gui.checkBox(box, self, "inner", "Exclude instances without a match",
                     callback=self._invalidate)

    def _setAttrs(self, model, data, othermodel, otherdata):
        model[:] = allvars(data) if data is not None else []

        if data is not None and otherdata is not None and \
                len(numpy.intersect1d(data.ids, otherdata.ids)):
            for model_ in (model, othermodel):
                if len(model_) and model_[0] != INSTANCEID:
                    model_.insert(0, INSTANCEID)

    @check_sql_input
    def setDataA(self, data):
        self.dataA = data
        self._setAttrs(self.attrModelA, data, self.attrModelB, self.dataB)
        curr_index = -1
        if self.attr_a:
            curr_index = next((i for i, val in enumerate(self.attrModelA)
                               if str(val) == self.attr_a), -1)
        if curr_index != -1:
            self.attrViewA.setCurrentIndex(curr_index)
        else:
            self.attr_a = INDEX
        self.infoBoxDataA.setText(self.dataInfoText(data))

    @check_sql_input
    def setDataB(self, data):
        self.dataB = data
        self._setAttrs(self.attrModelB, data, self.attrModelA, self.dataA)
        curr_index = -1
        if self.attr_b:
            curr_index = next((i for i, val in enumerate(self.attrModelB)
                               if str(val) == self.attr_b), -1)
        if curr_index != -1:
            self.attrViewB.setCurrentIndex(curr_index)
        else:
            self.attr_b = INDEX
        self.infoBoxDataB.setText(self.dataInfoText(data))

    def handleNewSignals(self):
        self._invalidate()

    def dataInfoText(self, data):
        ninstances = 0
        nvariables = 0
        if data is not None:
            ninstances = len(data)
            nvariables = len(data.domain)

        instances = self.tr("%n instance(s)", None, ninstances)
        attributes = self.tr("%n variable(s)", None, nvariables)
        return "\n".join([instances, attributes])

    def commit(self):
        AB = None
        if (self.attr_a and self.attr_b and
                self.dataA is not None and
                self.dataB is not None):
            varA = (self.attr_a if self.attr_a in (INDEX, INSTANCEID) else
                    self.dataA.domain[self.attr_a])
            varB = (self.attr_b if self.attr_b in (INDEX, INSTANCEID) else
                    self.dataB.domain[self.attr_b])
            AB = merge(self.dataA, varA, self.dataB, varB, self.inner)
        self.send("Merged Data", AB)

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


def allvars(data):
    return (INDEX,) + data.domain.attributes + data.domain.class_vars + data.domain.metas


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
    Y = join_array_by_indices(numpy.c_[left.Y], numpy.c_[right.Y], indices)
    metas = join_array_by_indices(left.metas, right.metas, indices)
    for col, var in enumerate(domain.metas):
        if var.is_string:
            for row in range(metas.shape[0]):
                cell = metas[row, col]
                if isinstance(cell, float) and numpy.isnan(cell):
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
        return numpy.hstack(list(map(numpy.vstack, blocks)))

    if left.shape[1] and left.dtype == object:
        leftparts = numpy.array(leftparts).astype(object)
    if right.shape[1] and right.dtype == object:
        rightparts = numpy.array(rightparts).astype(object)
    return hstack_blocks((leftparts, rightparts))


def test():
    from AnyQt.QtWidgets import QApplication
    app = QApplication([])

    w = OWMergeData()
    zoo = Orange.data.Table("zoo")
    A = zoo[:, [0, 1, 2, "type", -1]]
    B = zoo[:, [3, 4, 5, "type", -1]]
    w.setDataA(A)
    w.setDataB(B)
    w.handleNewSignals()
    w.show()
    app.exec_()


if __name__ == "__main__":
    test()
