from collections import namedtuple
from itertools import chain, product

import numpy as np

from AnyQt.QtCore import Qt, QModelIndex, pyqtSignal as Signal
from AnyQt.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy
)

from orangewidget.utils.combobox import ComboBoxSearch

import Orange
from Orange.data import StringVariable, ContinuousVariable, Variable, Domain
from Orange.data.util import hstack, get_unique_names_duplicates
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting, ContextHandler, ContextSetting
from Orange.widgets.utils import vartype
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
        ("pre_label", "left_combo", "in_label", "right_combo", "remove_button"))

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

    def get_button(self, label, callback):
        return gui.button(
            None, self, label, callback=callback,
            addToLayout=False, autoDefault=False, width=34,
            sizePolicy=(QSizePolicy.Maximum, QSizePolicy.Maximum))

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
            combo = ComboBoxSearch(self)
            combo.setModel(model)
            # We use signal activated because it is triggered only on user
            # interaction, not programmatically.
            combo.activated.connect(sync_combos)
            return combo

        row = self.layout().count()
        row_items = self.RowItems(
            QLabel("and" if row else self.pre_label),
            get_combo(self.model_left),
            QLabel(self.in_label),
            get_combo(self.model_right),
            self.get_button("×", self.on_remove_row)
        )
        layout = QHBoxLayout()
        layout.setSpacing(10)
        self.layout().insertLayout(self.layout().count() - 1, layout)
        layout.addStretch(10)
        for item in row_items:
            layout.addWidget(item)
        self.rows.append(row_items)
        self._reset_buttons()

    def add_plus_row(self):
        layout = QHBoxLayout()
        self.layout().addLayout(layout)
        layout.addStretch(1)
        layout.addWidget(self.get_button("+", self.on_add_row))

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
        self.rows[0].remove_button.setDisabled(len(self.rows) == 1)

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


class MergeDataContextHandler(ContextHandler):
    # `widget` is used as an argument in most methods
    # pylint: disable=redefined-outer-name
    # context handlers override methods using different signatures
    # pylint: disable=arguments-differ

    def new_context(self, variables1, variables2):
        context = super().new_context()
        context.variables1 = variables1
        context.variables2 = variables2
        return context

    def open_context(self, widget, domain1, domain2):
        if domain1 is not None and domain2 is not None:
            super().open_context(widget,
                                 self._encode_domain(domain1),
                                 self._encode_domain(domain2))

    @staticmethod
    def encode_variables(variables):
        return [(v.name, 100 + vartype(v))
                if isinstance(v, Variable) else (v, 100)
                for v in variables]

    @staticmethod
    def decode_pair(widget, pair):
        left_domain = widget.data and widget.data.domain
        right_domain = widget.extra_data and widget.extra_data.domain
        return tuple(var[0] if var[0] in (INDEX, INSTANCEID) else domain[var[0]]
                     for domain, var in zip((left_domain, right_domain), pair))

    def _encode_domain(self, domain):
        if domain is None:
            return {}
        if not isinstance(domain, Domain):
            domain = domain.domain
        all_vars = chain(domain.variables, domain.metas)
        return dict(self.encode_variables(all_vars))

    def settings_from_widget(self, widget, *_args):
        context = widget.current_context
        if context is None:
            return
        context.values["attr_pairs"] = [self.encode_variables(row)
                                        for row in widget.attr_pairs]

    def settings_to_widget(self, widget, *_args):
        context = widget.current_context
        if context is None:
            return
        pairs = context.values.get("attr_pairs")
        if pairs:
            # attr_pairs is schema only setting which means it is not always
            # present. When not present leave widgets default.
            widget.attr_pairs = [
                self.decode_pair(widget, pair) for pair in pairs
            ]

    def match(self, context, variables1, variables2):
        def matches(part, variables):
            return all(var[1] == 100 and var[0] in variables
                       or variables.get(var[0], -1) == var[1] for var in part)

        if (variables1, variables2) == (context.variables1, context.variables2):
            return self.PERFECT_MATCH

        if "attr_pairs" in context.values:
            left, right = zip(*context.values["attr_pairs"])
            if matches(left, variables1) and matches(right, variables2):
                return 0.5

        return self.NO_MATCH


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

    settingsHandler = MergeDataContextHandler()
    attr_pairs = ContextSetting(None, schema_only=True)
    merging = Setting(LeftJoin)
    auto_apply = Setting(True)
    settings_version = 2

    want_main_area = False
    resizing_enabled = False

    class Warning(widget.OWWidget.Warning):
        renamed_vars = Msg("Some variables have been renamed "
                           "to avoid duplicates.\n{}")

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

        grp = gui.radioButtons(
            self.controlArea, self, "merging", box="Merging",
            btnLabels=self.OptionNames, tooltips=self.OptionDescriptions,
            callback=self.change_merging)

        self.attr_boxes = ConditionBox(
            self, self.model, self.extra_model, "", "matches")
        self.attr_boxes.add_row()
        self.attr_boxes.add_plus_row()
        box = gui.vBox(self.controlArea, box="Row matching")
        box.layout().addWidget(self.attr_boxes)

        gui.auto_apply(self.buttonsArea, self)

        self.attr_boxes.vars_changed.connect(self.commit.deferred)
        self.attr_boxes.vars_changed.connect(self.store_combo_state)
        self.settingsAboutToBePacked.connect(self.store_combo_state)

    def change_merging(self):
        self.commit.deferred()

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        self.data = data
        self.model.set_domain(data.domain if data else None)

    @Inputs.extra_data
    @check_sql_input
    def set_extra_data(self, data):
        self.extra_data = data
        self.extra_model.set_domain(data.domain if data else None)

    def store_combo_state(self):
        self.attr_pairs = self.attr_boxes.current_state()

    def handleNewSignals(self):
        self.closeContext()
        self.attr_pairs = [self._find_best_match()]
        self.openContext(self.data and self.data.domain,
                         self.extra_data and self.extra_data.domain)
        self.attr_boxes.set_state(self.attr_pairs)
        self.commit.now()

    def _find_best_match(self):
        def get_unique_str_metas_names(model_):
            return [m for m in model_ if isinstance(m, StringVariable)]

        attr, extra_attr, n_max_intersect = INDEX, INDEX, 0
        str_metas = get_unique_str_metas_names(self.model)
        extra_str_metas = get_unique_str_metas_names(self.extra_model)
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

    @gui.deferred
    def commit(self):
        self.clear_messages()
        merged = self.merge() if self.data and self.extra_data else None
        self.Outputs.data.send(merged)

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
        reduced_extra_data = \
            self._compute_reduced_extra_data(right_vars, lefti, righti, rightu)
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

    def _compute_reduced_extra_data(self,
                                    right_match_vars, lefti, righti, rightu):
        """Prepare a table with extra columns that will appear in the merged
        table"""
        domain = self.data.domain
        extra_domain = self.extra_data.domain

        def var_needed(var):
            if rightu is not None and rightu.size:
                return True
            if var in right_match_vars and self.merging != self.OuterJoin:
                return False
            if var not in domain:
                return True
            both_defined = (lefti != -1) * (righti != -1)
            left_col = \
                self.data.get_column_view(var)[0][lefti[both_defined]]
            right_col = \
                self.extra_data.get_column_view(var)[0][righti[both_defined]]
            if var.is_primitive():
                left_col = left_col.astype(float)
                right_col = right_col.astype(float)
                mask_left = np.isfinite(left_col)
                mask_right = np.isfinite(right_col)
                return not (
                    np.all(mask_left == mask_right)
                    and np.all(left_col[mask_left] == right_col[mask_right]))
            else:
                return not np.all(left_col == right_col)

        extra_vars = [
            var for var in chain(extra_domain.variables, extra_domain.metas)
            if var_needed(var)]
        return self.extra_data[:, extra_vars]

    @staticmethod
    def _values(data, var, mask):
        """Return an iterotor over keys for rows of the table."""
        if var == INDEX:
            return np.arange(len(data))
        if var == INSTANCEID:
            return np.fromiter(
                (inst.id for inst in data), count=len(data), dtype=int)
        col = data.get_column_view(var)[0]
        if var.is_primitive():
            col = col.astype(float, copy=False)
            nans = np.isnan(col)
            mask *= ~nans
            if var.is_discrete:
                col = col.astype(int)
                col[nans] = len(var.values)
                col = np.array(var.values + (np.nan, ))[col]
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
        lt_dom = self.data.domain
        xt_dom = reduced_extra.domain
        domain = self._domain_rename_duplicates(
            lt_dom.attributes + xt_dom.attributes,
            lt_dom.class_vars + xt_dom.class_vars,
            lt_dom.metas + xt_dom.metas)
        X = self._join_array_by_indices(
            self.data.X, reduced_extra.X, lefti, righti)
        Y = self._join_array_by_indices(
            np.c_[self.data.Y], np.c_[reduced_extra.Y], lefti, righti)
        string_cols = [i for i, var in enumerate(domain.metas) if var.is_string]
        metas = self._join_array_by_indices(
            self.data.metas, reduced_extra.metas, lefti, righti, string_cols)
        if rightu is not None:
            # This domain is used for transforming the extra rows for outer join
            # It must use the original - not renamed - variables from right, so
            # values are copied,
            # but new domain for the left, so renamed values are *not* copied
            right_domain = Orange.data.Domain(
                domain.attributes[:len(lt_dom.attributes)] + xt_dom.attributes,
                domain.class_vars[:len(lt_dom.class_vars)] + xt_dom.class_vars,
                domain.metas[:len(lt_dom.metas)] + xt_dom.metas)
            extras = self.extra_data[rightu].transform(right_domain)
            X = np.vstack((X, extras.X))
            extras_Y = extras.Y
            if extras_Y.ndim == 1:
                extras_Y = extras_Y.reshape(-1, 1)
            Y = np.vstack((Y, extras_Y))
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

    def _domain_rename_duplicates(self, attributes, class_vars, metas):
        """Check for duplicate variable names in domain. If any, rename
        the variables, by replacing them with new ones (names are
        appended a number). """
        attrs, cvars, mets = [], [], []
        n_attrs, n_cvars, n_metas = len(attributes), len(class_vars), len(metas)
        lists = [attrs] * n_attrs + [cvars] * n_cvars + [mets] * n_metas

        all_vars = attributes + class_vars + metas
        proposed_names = [m.name for m in all_vars]
        unique_names = get_unique_names_duplicates(proposed_names)
        duplicates = set()
        for p_name, u_name, var, c in zip(proposed_names, unique_names,
                                          all_vars, lists):
            if p_name != u_name:
                duplicates.add(p_name)
                var = var.copy(name=u_name)
            c.append(var)
        if duplicates:
            self.Warning.renamed_vars(", ".join(duplicates))
        return Orange.data.Domain(attrs, cvars, mets)

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

        if not version or version < 2 and "attr_pairs" in settings:
            data_exists, extra_exists, attr_pairs = settings.pop("attr_pairs")
            if not (data_exists and extra_exists):
                settings["context_settings"] = []
                return

            mapper = {0: (INDEX, 100), 1: (INSTANCEID, 100)}
            context = ContextHandler().new_context()
            context.values["attr_pairs"] = [tuple(mapper.get(var, (var, 100))
                                                  for var in pair)
                                            for pair in attr_pairs]
            context.variables1 = {}
            context.variables2 = {}
            settings["context_settings"] = [context]


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWMergeData).run(
        set_data=Orange.data.Table("tests/data-gender-region"),
        set_extra_data=Orange.data.Table("tests/data-regions"))
