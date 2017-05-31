from enum import IntEnum
from itertools import chain, product, tee

from AnyQt.QtWidgets import QApplication, QStyle, QSizePolicy

import numpy as np
import scipy.sparse as sp

import Orange
from Orange.data import StringVariable, ContinuousVariable
from Orange.data.util import hstack
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import Input, Output


class MergeType(IntEnum):
    LEFT_JOIN, INNER_JOIN, OUTER_JOIN = 0, 1, 2

INSTANCEID = "Source position (index)"
INDEX = "Position (index)"


class OWMergeData(widget.OWWidget):
    name = "Merge Data"
    description = "Merge data sets based on the values of selected features."
    icon = "icons/MergeData.svg"
    priority = 1110

    class Inputs:
        data = Input("Data", Orange.data.Table, default=True, replaces=["Data A"])
        extra_data = Input("Extra Data", Orange.data.Table, replaces=["Data B"])

    class Outputs:
        data = Output("Data",
                      Orange.data.Table,
                      replaces=["Merged Data A+B", "Merged Data B+A", "Merged Data"])

    attr_augment_data = settings.Setting('', schema_only=True)
    attr_augment_extra = settings.Setting('', schema_only=True)
    attr_merge_data = settings.Setting('', schema_only=True)
    attr_merge_extra = settings.Setting('', schema_only=True)
    attr_combine_data = settings.Setting('', schema_only=True)
    attr_combine_extra = settings.Setting('', schema_only=True)
    merging = settings.Setting(0)

    want_main_area = False
    resizing_enabled = False

    class Warning(widget.OWWidget.Warning):
        duplicate_names = widget.Msg("Duplicate variable names in output.")

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
            cb.setFixedWidth(190)
            gui.widgetLabel(box, between_label)
            cb = gui.comboBox(box, self, 'attr_{}_extra'.format(merge_type),
                              callback=self._invalidate, model=extra_model)
            cb.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
            cb.setFixedWidth(190)
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
        for attr in chain(data.domain.variables, data.domain.metas):
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
            has_id = INSTANCEID in model
            if needs_id and not has_id:
                model.insert(0, INSTANCEID)
            elif not needs_id and has_id:
                model.remove(INSTANCEID)

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
    @Inputs.data
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
    @Inputs.extra_data
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
        if self.data is None or len(self.data) == 0 or \
                self.extra_data is None or len(self.extra_data) == 0:
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
        merge_method = getattr(self, "_{}_indices".format(operation))

        as_string = not (isinstance(var_data, ContinuousVariable) and
                         isinstance(var_extra_data, ContinuousVariable))
        extra_map = self._get_keymap(self.extra_data, var_extra_data, as_string)
        match_indices = merge_method(var_data, extra_map, as_string)
        reduced_extra_data = self._compute_reduced_extra_data(var_extra_data)
        return self._join_table_by_indices(reduced_extra_data, match_indices)

    def _compute_reduced_extra_data(self, var_extra_data):
        """Prepare a table with extra columns that will appear in the merged
        table"""
        domain = self.data.domain
        extra_domain = self.extra_data.domain
        all_vars = set(chain(domain.variables, domain.metas))
        if self.merging != MergeType.OUTER_JOIN:
            all_vars.add(var_extra_data)
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
    def _get_keymap(cls, data, var, as_string):
        """Return a generator of pairs (key, index) by enumerating and
        switching the values for rows (method `_values`).
        """
        return ((val, i)
                for i, val in enumerate(cls._values(data, var, as_string)))

    def _augment_indices(self, var_data, extra_map, as_string):
        """Compute a two-row array of indices:
        - the first row contains indices for the primary table,
        - the second row contains the matching rows in the extra table or -1"""
        data = self.data
        extra_map = dict(extra_map)
        # Don't match nans. This is needed since numpy supports using nan as
        # keys. If numpy fixes this, the below conditions will always be false,
        # so we're OK again.
        if np.nan in extra_map:
            del extra_map[np.nan]
        keys = (extra_map.get(val, -1)
                for val in self._values(data, var_data, as_string))
        return np.vstack((np.arange(len(data), dtype=np.int64),
                          np.fromiter(keys, dtype=np.int64, count=len(data))))

    def _merge_indices(self, var_data, extra_map, as_string):
        """Use _augment_indices to compute the array of indices,
        then remove those with no match in the second table"""
        augmented = self._augment_indices(var_data, extra_map, as_string)
        return augmented[:, augmented[1] != -1]

    def _combine_indices(self, var_data, extra_map, as_string):
        """Use _augment_indices to compute the array of indices,
        then add rows in the second table without a match in the first"""
        to_add, extra_map = tee(extra_map)
        # dict instead of set because we have pairs; we'll need only keys
        key_map = dict(self._get_keymap(self.data, var_data, as_string))
        # _augment indices will skip rows where the key in the left table
        # is nan. See comment in `_augment_indices` wrt numpy and nan in dicts
        if np.nan in key_map:
            del key_map[np.nan]
        keys = np.fromiter((j for key, j in to_add if key not in key_map),
                           dtype=np.int64)
        right_indices = np.vstack((np.full(len(keys), -1, np.int64), keys))
        return np.hstack(
            (self._augment_indices(var_data, extra_map, as_string),
             right_indices))

    def _join_table_by_indices(self, reduced_extra, indices):
        """Join (horizontally) self.data and reduced_extra, taking the pairs
        of rows given in indices"""
        if not len(indices):
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
        return Orange.data.Table.from_numpy(domain, X, Y, metas)

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


def main():
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
    main()
