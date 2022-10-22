from functools import partial
from types import SimpleNamespace
from typing import NamedTuple, Optional, Callable, Union, Dict, Tuple, List

import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import Qt, QSize, QAbstractListModel, QObject
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import \
    QButtonGroup, QRadioButton, QListView, QGridLayout, QStyledItemDelegate

from orangewidget.utils import listview

from Orange.data import DiscreteVariable, ContinuousVariable, Domain, Table
from Orange.preprocess import Continuize as Continuizer
from Orange.preprocess.transformation import Identity, Indicator, Normalizer
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output


# When used as class, NamedTuple does not support inheritance, hence we use
# it as function and "inherit" by adding elements to the list
method_descs_base = [
    ("id_", int),
    ("label", str),  # Label used for radio button
    ("short_desc", str),  # Short description for list views
    ("tooltip", str),  # Tooltip for radio button
    ("function", Optional[Callable[..., Union[DiscreteVariable, str]]])
]

DiscreteMethodDesc = NamedTuple(
    "DiscreteMethodDesc",
    method_descs_base
)

ContinuousMethodDesc = NamedTuple(
    "ContinuousMethodDesc",
    method_descs_base + [
        ("supports_sparse", bool)
    ]
)


Continuize = SimpleNamespace(
    Default=99,
    **{v.name: v.value for v in Continuizer.MultinomialTreatment})

DiscreteOptions: Dict[int, DiscreteMethodDesc] = {
    method.id_: method
    for method in (
        DiscreteMethodDesc(
            Continuize.Default, "Use default setting", "default",
            "Treat the variable as defined in 'default setting'",
            None),
        DiscreteMethodDesc(
            Continuize.Leave, "Leave categorical", "leave",
            "Leave the variable discrete",
            None),
        DiscreteMethodDesc(
            Continuize.FirstAsBase,
            "First value as base", "first as base",
            "",
            None),
        DiscreteMethodDesc(
            Continuize.FrequentAsBase,
            "Most frequent value as base","frequent as base",
            "",
            None),
        DiscreteMethodDesc(
            Continuize.Indicators,
            "One indicator variable per value", "indicators",
            "",
            None),
        DiscreteMethodDesc(
            Continuize.RemoveMultinomial,
            "Remove if more than 3 values", "remove if >3",
            "",
            None),
        DiscreteMethodDesc(
            Continuize.Remove,
            "Remove", "remove",
            "",
            None),
        DiscreteMethodDesc(
            Continuize.AsOrdinal,
            "Treat as ordinal", "as ordinal",
            "",
            None),
        DiscreteMethodDesc(
            Continuize.AsNormalizedOrdinal,
            "Treat as normalized ordinal", "as norm. ordinal",
            "",
            None),
    )}

Normalize = SimpleNamespace(Default=99,
                            Leave=0, Standardize=1, Center=2, Scale=3,
                            Normalize11=4, Normalize01=5)

ContinuousOptions: Dict[int, ContinuousMethodDesc] = {
    method.id_: method
    for method in (
        ContinuousMethodDesc(
            Normalize.Default,
            "Use default setting", "default",
            "Treat the variable as defined in 'default setting'",
            None, True),
        ContinuousMethodDesc(
            Normalize.Leave,
            "Leave as it is", "leave",
            "",
            None, True),
        ContinuousMethodDesc(
            Normalize.Standardize,
            "Standardize to μ=0, σ²=1", "standardize",
            "",
            None, False),
        ContinuousMethodDesc(
            Normalize.Center,
            "Center to μ=0", "center",
            "",
            None, False),
        ContinuousMethodDesc(
            Normalize.Scale,
            "Scale to σ²=1", "scale",
            "",
            None, True),
        ContinuousMethodDesc(
            Normalize.Normalize11,
            "Normalize to interval [-1, 1]", "to [-1, 1]",
            "",
            None, False),
        ContinuousMethodDesc(
            Normalize.Normalize01,
            "Normalize to interval [0, 1]", "to [0, 1]",
            "",
            None, False),
    )}


class ContDomainModel(VariableListModel):
    """
    Domain model that adds description of continuization methods
    """
    def data(self, index, role=Qt.DisplayRole):
        value = super().data(index, role)
        row = index.row()
        if role == Qt.DisplayRole:
            hint = index.data(Qt.UserRole)
            if hint is not None:
                methods = \
                    DiscreteOptions if isinstance(self[row], DiscreteVariable) \
                    else ContinuousOptions
                return value + f": {methods[hint].short_desc}"
        return value


class DefaultContModel(QAbstractListModel):
    """
    A model used for showing "Default settings" above the list view with var
    """
    icon = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if DefaultContModel.icon is None:
            DefaultContModel.icon = gui.createAttributePixmap(
                "★", QColor(0, 0, 0, 0), Qt.black)
        self.hints: Tuple[VarHint, VarHint] = (
            Normalize.Leave, Continuize.FirstAsBase)

    @staticmethod
    def rowCount(parent):
        return 0 if parent.isValid() else 1

    @staticmethod
    def columnCount(parent):
        return 0 if parent.isValid() else 1

    def data(self, _, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return f"Default setting: " + \
                   DiscreteOptions[self.hints[1]].short_desc + " / " + \
                   ContinuousOptions[self.hints[0]].short_desc
        elif role == Qt.DecorationRole:
            return self.icon
        elif role == Qt.ToolTipRole:
            return "Default for variables without specific setings"
        return None

    def setData(self, index, value, role=Qt.DisplayRole):
        if role == Qt.UserRole:
            self.hints = value
            self.dataChanged.emit(index, index)


class ListViewSearch(listview.ListViewSearch):
    """
    A list view with two components shown above it:
    - a listview containing a single item representing default settings
    - a filter for search

    The class is based on listview.ListViewSearch and needs to have the same
    name in order to override its private method __layout.

    Inherited __init__ calls __layout, so `default_view` must be constructed
    there. Construction before calling super().__init__ doesn't work because
    PyQt does not allow it.
    """
    class Delegate(QStyledItemDelegate):
        """
        A delegate that shows items (variables) with specific settings in bold
        """
        def initStyleOption(self, option, index):
            super().initStyleOption(option, index)
            option.font.setBold(index.data(Qt.UserRole) is not None)

    def __init__(self, *args, **kwargs):
        self.default_view = None
        super().__init__(preferred_size=QSize(350, -1), *args, **kwargs)
        self.setItemDelegate(self.Delegate(self))

    def select_default(self, idx):
        """Select the item representing default settings"""
        index = self.default_view.model().index(idx)
        self.default_view.selectionModel().select(
            index, QItemSelectionModel.Select)

    # pylint: disable=unused-private-member
    def __layout(self):
        if self.default_view is None:  # __layout was called from __init__
            view = self.default_view = QListView(self)
            view.setModel(DefaultContModel())
            view.verticalScrollBar().setDisabled(True)
            view.horizontalScrollBar().setDisabled(True)
            view.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            view.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            font = view.font()
            font.setBold(True)
            view.setFont(font)
        else:
            view = self.default_view

        # Put the list view with default on top
        margins = self.viewportMargins()
        def_height = view.sizeHintForRow(0) + 2 * view.spacing() + 2
        view.setGeometry(0, 0, self.geometry().width(), def_height)
        view.setFixedHeight(def_height)

        # Then search
        search = self.__search
        src_height = search.sizeHint().height()
        size = self.size()
        search.setGeometry(0, def_height + 2, size.width(), src_height)

        # Then the real list view
        margins.setTop(def_height + 2 + src_height)
        self.setViewportMargins(margins)

KeyType = Tuple[str, bool]  # the True if categorical
DefaultKey = "", True

def variable_key(var: ContinuousVariable) -> KeyType:
    """Key for that variable in var_hints and discretized_vars"""
    return var.name, isinstance(var, DiscreteVariable)


class OWContinuize(widget.OWWidget):
    name = "Continuize"
    description = ("Transform categorical attributes into numeric and, " +
                   "optionally, normalize numeric values.")
    icon = "icons/Continuize.svg"
    category = "Transform"
    keywords = "continuize, encode, dummy, numeric, one-hot, binary, treatment, contrast"
    priority = 2120

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    want_main_area = False

    settings_version = 3
    var_hints: Dict[KeyType, Union[int, Tuple[int, int]]] = Setting(
        {DefaultKey: (Normalize.Leave, Continuize.FirstAsBase)},
        schema_only=True)
    continuize_class = Setting(False, schema_only=True)
    autosend = Setting(True)

    def __init__(self):
        super().__init__()
        self.data = None
        self._var_cache = {}

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        gui.widgetBox(self.controlArea, True, spacing=8, orientation=grid)

        self.varview = ListViewSearch(
            selectionMode=ListViewSearch.ExtendedSelection,
            uniformItemSizes=True)
        # TODO: Add argument strict_valid_types that does not allow TimeVariable
        # to pass as ContinuousVariable
        self.varview.setModel(ContDomainModel())
        self.varview.selectionModel().selectionChanged.connect(
            self._var_selection_changed)
        self.varview.default_view.selectionModel().selectionChanged.connect(
            self._default_selected)
        self._update_default_model()
        grid.addWidget(self.varview, 0, 0, 6, 1)

        def create_buttons(methods, title):
            bbox = gui.vBox(None, title)
            bgroup = QButtonGroup(self)
            bgroup.idClicked.connect(self.update_hints)
            for desc in methods.values():
                button = QRadioButton(desc.label)
                button.setToolTip(desc.tooltip)
                bgroup.addButton(button, desc.id_)
                bbox.layout().addWidget(button)
            return bbox, bgroup

        self.discrete_buttons, self.discrete_group = create_buttons(
            DiscreteOptions, "Continuization (for categorical)")
        self.continuous_buttons, self.continuous_group = create_buttons(
            ContinuousOptions, "Normalizations (for numeric)")
        grid.addWidget(self.discrete_buttons, 0, 1)
        grid.setRowMinimumHeight(1, 16)
        grid.addWidget(self.continuous_buttons, 2, 1)

        grid.setRowMinimumHeight(3, 16)
        grid.addWidget(
            gui.checkBox(None, self, "continuize_class",
                         "Change class to numeric", box="Target").box,
            4, 1)

        grid.setRowStretch(5, 1)
        gui.rubber(self.buttonsArea)

        gui.auto_apply(self.buttonsArea, self, "autosend")

    def _update_default_model(self):
        """Update data in the model showing default settings"""
        model = self.varview.default_view.model()
        model.setData(model.index(0), self.var_hints[DefaultKey], Qt.UserRole)

    def _var_selection_changed(self):
        selected = self.varview.selectionModel().selectedIndexes()
        self.varview.default_view.selectionModel().clearSelection()
        self._update_interface()

    def _default_selected(self):
        self.varview.selectionModel().clearSelection()
        self._update_interface()

    def varkeys_for_selection(self) -> List[KeyType]:
        """
        Return list of KeyType's for selected variables (for indexing var_hints)

        If 'Default settings' are selected, this returns DefaultKey
        """
        model = self.varview.model()
        varkeys = [variable_key(model[index.row()])
                   for index in self.varview.selectionModel().selectedRows()]
        return varkeys or [DefaultKey]  # default settings are selected

    def _update_interface(self):
        keys = self.varkeys_for_selection()

        parts = (
            (self.discrete_group, self.discrete_buttons, True),
            (self.continuous_group, self.continuous_buttons, False))

        if keys == [DefaultKey]:
            hints = self.var_hints[DefaultKey]
            for group, box, type_ in parts:
                self._check_button(group, hints[type_], True)
                self._set_radio_enabled(group, 99, False)
                box.setDisabled(False)
            return

        hints = {(self.var_hints.get(key, 99), key[1]) for key in keys}
        for group, box, type_ in parts:
            options = {hint for hint, a_type in hints if a_type is type_}
            box.setDisabled(not options)
            self._set_radio_enabled(group, 99, True)
            if len(options) == 1:
                self._check_button(group, options.pop(), True)
            else:
                self._uncheck_all_buttons(group)

        # Disable or re-enable options that don't support sparse data
        if self.data.is_sparse:
            conts = {var for var, type_ in keys if not type_}
            any_sparse = bool(
                sp.issparse(self.data.X) and conts & set(self.data.attributes)
                or sp.issparse(self.data.metas) and conts & set(self.data.metas)
            )
        else:
            any_sparse = False
        for desc in ContinuousOptions.values():
            if not desc.supports_sparse:
                self._set_radio_enabled(
                    self.continuous_group, desc.id_, not any_sparse)

    def _uncheck_all_buttons(self, group: QButtonGroup):
        button = group.checkedButton()
        if button is not None:
            group.setExclusive(False)
            button.setChecked(False)
            group.setExclusive(True)

    def _set_radio_enabled(
            self, group: QButtonGroup, method_id: int, value: bool):
        if group.button(method_id).isChecked() and not value:
            self._uncheck_all_buttons(group)
        group.button(method_id).setEnabled(value)

    def _check_button(
            self, group: QButtonGroup, method_id: int, checked: bool):
        group.button(method_id).setChecked(checked)

    def update_hints(self, method_id: int):
        is_discrete = QObject().sender() is self.discrete_group
        keys = self.varkeys_for_selection()
        if keys == [DefaultKey]:
            self.var_hints[DefaultKey] = \
                (self.var_hints[1], method_id) if is_discrete \
                else (method_id, self.var_hints[0])
            self._update_default_model()
        else:
            # Keep only discrete (continuous) indices and keys when the user
            # changed the button for discrete (continuous) variables
            selected_indexes = self.varview.selectionModel().selectedIndexes()
            indexes = [index
                       for index, (_, type_) in zip(selected_indexes, keys)
                       if type_ is is_discrete]
            keys = [key for key in keys if key[1] is is_discrete]
            model = self.varview.model()
            if method_id == 99:
                for key in keys:
                    if key in self.var_hints:
                        del self.var_hints[key]
                data = None
            else:
                self.var_hints.update(dict.fromkeys(keys, method_id))
                data = method_id
            for index in indexes:
                model.setData(index, data, Qt.UserRole)
        self.commit.deferred()

    @Inputs.data
    @check_sql_input
    def setData(self, data):
        model = self.varview.model()

        self.data = data
        self._var_cache.clear()
        if data is None:
            model.clear()
            self.Outputs.data.send(None)
            return

        valid = (DiscreteVariable, ContinuousVariable)
        attrs, metas = ([var for var in part if type(var) in valid]
                        for part in (data.domain.attributes, data.domain.metas))
        model[:] = attrs + [model.Separator] * bool(attrs and metas) + metas
        self.commit.now()

    @gui.deferred
    def commit(self):
        if not self.data:
            self.Outputs.data.send(None)
            return

        domain = self.data.domain
        attrs = self._create_vars(domain.attributes)
        class_vars = domain.class_vars
        if self.continuize_class:
            class_vars = [
                self._continuized_vars(var, Continuize.AsOrdinal)[0]
                if var.is_discrete else var
                for var in class_vars]
        metas = self._create_vars(domain.metas)
        data = self.data.transform(Domain(attrs, class_vars, metas))
        self.Outputs.data.send(data)

    def _create_vars(self, part):
        created = []
        defaults = self.var_hints[DefaultKey]
        for var in part:
            key = variable_key(var)
            hint = self.var_hints.get(key, defaults[key[1]])
            if hint is None:
                created.append(var)
            elif var.is_continuous:
                created.append(self._scaled_var(var, hint))
            else:
                created += self._continuized_vars(var, hint)
        return created

    def _get(self, var, stat):
        def most_frequent(col):
            col = col[np.isfinite(col)].astype(int)
            counts = np.bincount(col, minlength=len(var.values))
            return np.argmax(counts)

        funcs = {"min": np.nanmin, "max": np.nanmax,
                 "mean": np.nanmean, "std": np.nanstd,
                 "major": most_frequent}
        name = var.name
        cache = self._var_cache.setdefault(name, {})
        if stat not in cache:
            # TODO: Sparse
            # TODO: get_column
            cache[stat] = funcs[stat](self.data.get_column_view(var)[0].astype(float))
        return cache[stat]

    def _scaled_var(self, var, hint):
        if hint == Normalize.Leave:
            return var

        get = partial(self._get, var)
        if hint == Normalize.Standardize:
            off, scale = get("mean"), get("std")
        elif hint == Normalize.Center:
            off, scale = get("mean"), 1
        elif hint == Normalize.Scale:
            off, scale = 0, get("std")
        else:
            min_, max_ = get("min"), get("max")
            span = (max_ - min_) or 1
            if hint == Normalize.Normalize11:
                off, scale = (min_ + max_) / 2, 2 / span
            else:
                off, scale = min_, 1 / span
        return ContinuousVariable(
            var.name,
            compute_value=Normalizer(var, off, scale))

    def _continuized_vars(self, var, hint):
        if hint == Continuize.Leave:
            return [var]
        if hint == Continuize.Remove:
            return []
        if hint == Continuize.RemoveMultinomial:
            return [var] if len(var.values) <= 2 else []
        if hint == Continuize.AsOrdinal or len(var.values) < 2:
            return [ContinuousVariable(
                var.name,
                compute_value=Identity(var))]
        if hint == Continuize.AsNormalizedOrdinal:
            scale = 1 / (len(var.values) - 1 or 1)
            return [
                ContinuousVariable(var.name,
                                   compute_value=Normalizer(var, 0, scale))]

        if hint == Continuize.FirstAsBase:
            base = 0
        elif hint == Continuize.FrequentAsBase:
            base = self._get(var, "major")
        elif hint == Continuize.Indicators:
            base = -1
        else:
            assert False
        return [
            ContinuousVariable(f"{var.name}={value}",
                               compute_value=Indicator(var, value=i))
            for i, value in enumerate(var.values)
            if i != base
        ]

    def send_report(self):
        # TOTO: Implement
        pass

    @classmethod
    def migrate_settings(cls, settings, version):
        if version < 2:
            Normalize = cls.Normalize
            cont_treat = settings.pop("continuous_treatment", 0)
            zero_based = settings.pop("zero_based", True)
            if cont_treat == 1:
                if zero_based:
                    settings["continuous_treatment"] = Normalize.Normalize01
                else:
                    settings["continuous_treatment"] = Normalize.Normalize11
            elif cont_treat == 2:
                settings["continuous_treatment"] = Normalize.Standardize
        # TODO: Migrate to hints


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWContinuize).run(Table("heart_disease"))
