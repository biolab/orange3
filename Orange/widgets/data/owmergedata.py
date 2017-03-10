from enum import IntEnum
from itertools import chain, product

from AnyQt.QtWidgets import QApplication, QStyle, QSizePolicy

import numpy as np

import Orange
from Orange.data import StringVariable
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.sql import check_sql_input


class MergeType(IntEnum):
    LEFT_JOIN, INNER_JOIN, OUTER_JOIN = 0, 1, 2

INSTANCEID = "Source position (index)"
INDEX = "Position (index)"


class OWMergeData(widget.OWWidget):
    name = "Merge Data"
    description = "Merge data sets based on the values of selected features."
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
        # pylint: disable=invalid-sequence-index
        # all boxes should be hidden before one is shown, otherwise widget's
        # layout changes height
        for box in self.attr_boxes:
            box.hide()
        self.attr_boxes[self.merging].show()

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
        if self.data is None or self.extra_data is None:
            merged_data = None
        else:
            merged_data = self.merge()
            if merged_data:
                merged_domain = merged_data.domain
                var_names = [var.name for var in chain(merged_domain.variables,
                                                       merged_domain.metas)]
                if len(set(var_names)) != len(var_names):
                    self.Warning.duplicate_names()
        self.send("Data", merged_data)

    def _invalidate(self):
        self.commit()

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
        operation = ["augment", "merge", "combine"][self.merging]
        var_data = getattr(self, "attr_{}_data".format(operation))
        var_extra_data = getattr(self, "attr_{}_extra".format(operation))

        method = getattr(self, "_{}_indices".format(operation))
        merge_indices = method(var_data, var_extra_data)
        reduced_extra = self._compute_reduced_extra(var_extra_data)
        return self.join_table_by_indices(reduced_extra, merge_indices)

    def _compute_reduced_extra(self, var_extra_data):
        domain = self.data.domain
        extra_domain = self.extra_data.domain
        all_vars = set(chain(domain.variables, domain.metas))
        if self.merging != MergeType.OUTER_JOIN:
            all_vars.add(var_extra_data)
        iter_extra_vars = chain(
            enumerate(extra_domain.variables),
            ((-i, m) for i, m in enumerate(extra_domain.metas, start=1)))
        return self.extra_data[:, [i for i, var in iter_extra_vars
                                   if var not in all_vars]]

    @staticmethod
    def get_keymap(data, var):
        if var == INSTANCEID:
            return {inst.id: i for i, inst in enumerate(data)}
        elif var == INDEX:
            return {i: i for i in range(len(data))}
        elif var != INDEX:
            return {str(inst[var]): i for i, inst in enumerate(data)}

    def _augment_indices(self, var_data, var_extra_data):
        data = self.data
        n = len(self.data)
        extra_map = self.get_keymap(self.extra_data, var_extra_data)
        if var_data == INSTANCEID:
            keys = (extra_map.get(inst.id, -1) for inst in data)
        elif var_data == INDEX:
            keys = (extra_map.get(i, -1) for i in range(n))
        else:
            keys = (extra_map.get(str(inst[var_data]), -1) for inst in data)
        return np.vstack((np.arange(n, dtype=np.int64),
                          np.fromiter(keys, dtype=np.int64, count=n)))

    def _merge_indices(self, var_data, var_extra_data):
        augmented = self._augment_indices(var_data, var_extra_data)
        return augmented[:, augmented[1] != -1]

    def _combine_indices(self, var_data, var_extra_data):
        extra_data = self.extra_data
        if var_extra_data == INSTANCEID:
            to_add = (inst.id for inst in extra_data)
        elif var_extra_data == INDEX:
            to_add = range(len(extra_data))
        else:
            to_add = (str(inst[var_extra_data]) for inst in extra_data)
        key_map = self.get_keymap(self.data, var_data)
        keys = np.fromiter((j for j, key in enumerate(to_add)
                            if key not in key_map), dtype=np.int64)
        right_indices = np.vstack((np.full(len(keys), -1, np.int64), keys))
        return np.hstack(
            (self._augment_indices(var_data, var_extra_data), right_indices))

    def join_table_by_indices(self, reduced_extra, indices):
        if not len(indices):
            return None
        domain = Orange.data.Domain(
            *(getattr(self.data.domain, x) + getattr(reduced_extra.domain, x)
              for x in ("attributes", "class_vars", "metas")))
        X = self.join_array_by_indices(
            self.data.X, reduced_extra.X, indices)
        Y = self.join_array_by_indices(
            np.c_[self.data.Y], np.c_[reduced_extra.Y], indices)
        string_cols = [i for i, var in enumerate(domain.metas) if var.is_string]
        metas = self.join_array_by_indices(
            self.data.metas, reduced_extra.metas, indices, string_cols)
        return Orange.data.Table.from_numpy(domain, X, Y, metas)

    @staticmethod
    def join_array_by_indices(left, right, indices, string_cols=None):
        tpe = object if object in (left.dtype, right.dtype) else left.dtype
        left_width, right_width = left.shape[1], right.shape[1]
        arr = np.full((indices.shape[1], left_width + right_width), np.nan, tpe)
        if string_cols:
            arr[:, string_cols] = ""
        for indices, to_change, lookup in (
                (indices[0], arr[:, :left_width], left),
                (indices[1], arr[:, left_width:], right)):
            known = indices != -1
            to_change[known] = lookup[indices[known]]
        return arr


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
