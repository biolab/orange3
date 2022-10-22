from functools import partial, reduce
from operator import add
from types import SimpleNamespace
from typing import NamedTuple, Optional, Callable, Union, Dict, List

import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import Qt, QSize, QAbstractListModel, QObject, \
    QItemSelectionModel
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import \
    QButtonGroup, QRadioButton, QListView, QStyledItemDelegate

from orangewidget.utils import listview

from Orange.data import DiscreteVariable, ContinuousVariable, Domain, Table
from Orange.preprocess import Continuize as Continuizer
from Orange.preprocess.transformation import Identity, Indicator, Normalizer
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemmodels import DomainModel
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


class ContDomainModel(DomainModel):
    """
    Domain model that adds description of continuization methods
    """
    def __init__(self, valid_type):
        super().__init__(
            (DomainModel.ATTRIBUTES, DomainModel.Separator, DomainModel.METAS),
            valid_types=(valid_type, ), strict_type=True)

    def data(self, index, role=Qt.DisplayRole):
        value = super().data(index, role)
        if role == Qt.DisplayRole:
            hint = index.data(Qt.UserRole)
            if hint is not None:
                value += f": {hint}"
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
        self.hint = ""

    @staticmethod
    def rowCount(parent):
        return 0 if parent.isValid() else 1

    @staticmethod
    def columnCount(parent):
        return 0 if parent.isValid() else 1

    def data(self, _, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return f"Default setting: {self.hint}"
        elif role == Qt.DecorationRole:
            return self.icon
        elif role == Qt.ToolTipRole:
            return "Default for variables without specific settings"
        return None

    def setData(self, index, value, role=Qt.DisplayRole):
        if role == Qt.UserRole:
            self.hint = value
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

    def select_default(self):
        """Select the item representing default settings"""
        self.default_view.selectionModel().select(
            self.default_view.model().index(0),
            QItemSelectionModel.Select)

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


DefaultKey = ""


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

    want_control_area = False
    want_main_area = True

    settings_version = 3
    disc_var_hints: Dict[str, int] = Setting(
        {DefaultKey: Continuize.FirstAsBase}, schema_only=True)
    cont_var_hints: Dict[str, int] = Setting(
        {DefaultKey: Normalize.Leave}, schema_only=True)
    continuize_class = Setting(False, schema_only=True)
    autosend = Setting(True)

    def __init__(self):
        super().__init__()
        self.data = None
        self._var_cache = {}

        def create(title, vartype, methods):
            hbox = gui.hBox(box, title)
            view = ListViewSearch(
                selectionMode=ListViewSearch.ExtendedSelection,
                uniformItemSizes=True)
            view.setModel(ContDomainModel(vartype))
            view.selectionModel().selectionChanged.connect(
                lambda: self._var_selection_changed(view))
            view.default_view.selectionModel().selectionChanged.connect(
                lambda selected: self._default_selected(view, selected))
            hbox.layout().addWidget(view)

            bbox = gui.vBox(hbox)
            bgroup = QButtonGroup(self)
            bgroup.idClicked.connect(self.update_hints)
            for desc in methods.values():
                button = QRadioButton(desc.label)
                button.setToolTip(desc.tooltip)
                bgroup.addButton(button, desc.id_)
                bbox.layout().addWidget(button)
            bbox.layout().addStretch(1)
            return hbox, view, bbox, bgroup

        box = gui.vBox(self.mainArea, True, spacing=8)
        self.disc_box, self.disc_view, self.disc_radios, self.disc_group = \
            create("Categorical Variables", DiscreteVariable, DiscreteOptions)
        self._update_default_model(
            self.disc_view,
            DiscreteOptions[self.disc_var_hints[DefaultKey]].short_desc)
        self.disc_view.select_default()

        self.cont_box, self.cont_view, self.cont_radios, self.cont_group = \
            create("Numeric Variables", ContinuousVariable, ContinuousOptions)
        self._update_default_model(
            self.cont_view,
            ContinuousOptions[self.cont_var_hints[DefaultKey]].short_desc)
        self.cont_view.select_default()
        
        boxes = (self.disc_radios, self.cont_radios)
        width = max(box.sizeHint().width() for box in boxes)
        for box in boxes:
            box.setFixedWidth(width)

        box = self.class_box = gui.hBox(self.mainArea)
        gui.checkBox(
            box, self, "continuize_class",
            "Change class to numeric", box="Target Variable")

        box = gui.hBox(self.mainArea)
        gui.rubber(box)
        gui.auto_apply(box, self, "autosend")

    def _update_default_model(self, view, desc):
        """Update data in the model showing default settings"""
        model = view.default_view.model()
        model.setData(model.index(0), desc, Qt.UserRole)

    def _var_selection_changed(self, view):
        if not view.selectionModel().selectedIndexes():
            # Prevent infinite recursion (with _default_selected)
            return
        view.default_view.selectionModel().clearSelection()
        self._update_interface(view)

    def _default_selected(self, view, selected):
        if not selected:
            # Prevent infinite recursion (with _var_selection_selected)
            return
        view.selectionModel().clearSelection()
        self._update_interface(view)

    def varkeys_for_selection(self, view) -> List[str]:
        """
        Return names of selected variables for indexing var hints

        If 'Default settings' are selected, this returns DefaultKey
        """
        model = view.model()
        varkeys = [model[index.row()].name
                   for index in view.selectionModel().selectedRows()]
        return varkeys or [DefaultKey]  # default settings are selected

    def _update_interface(self, view):
        keys = self.varkeys_for_selection(view)
        if view is self.disc_view:
            group, buttons = self.disc_group, self.disc_radios
            hints = self.disc_var_hints
        else:
            group, buttons = self.cont_group, self.cont_radios
            hints = self.cont_var_hints

        if keys == [DefaultKey]:
            self._check_button(group, hints[DefaultKey], True)
            self._set_radio_enabled(group, 99, False)
            return

        self._set_radio_enabled(group, 99, True)
        options = {hints.get(key, 99) for key in keys}
        if len(options) == 1:
            self._check_button(group, options.pop(), True)
        else:
            self._uncheck_all_buttons(group)

        if view is self.cont_view:
            # Disable or re-enable options that don't support sparse data
            domain = self.data.domain
            if self.data.is_sparse:
                skeys = set(keys)
                any_sparse = bool(
                    sp.issparse(self.data.X)
                    and skeys & {var.name for var in domain.attributes}
                    or
                    sp.issparse(self.data.metas)
                    and skeys & {var.name for var in domain.metas}
                )
            else:
                any_sparse = False
            for desc in ContinuousOptions.values():
                if not desc.supports_sparse:
                    self._set_radio_enabled(group, desc.id_, not any_sparse)

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
        if QObject().sender() is self.disc_group:
            view, hints, methods = \
                self.disc_view, self.disc_var_hints, DiscreteOptions
        else:
            view, hints, methods = \
                self.cont_view, self.cont_var_hints, ContinuousOptions
        keys = self.varkeys_for_selection(view)
        if keys == [DefaultKey]:
            hints[DefaultKey] = method_id
            self._update_default_model(view, methods[method_id].short_desc)
        else:
            indexes = view.selectionModel().selectedIndexes()
            model = view.model()
            if method_id == 99:
                for key in keys:
                    if key in hints:
                        del hints[key]
                desc = None
            else:
                hints.update(dict.fromkeys(keys, method_id))
                desc = methods[method_id].short_desc
            for index in indexes:
                model.setData(index, desc, Qt.UserRole)
        self.commit.deferred()

    @Inputs.data
    @check_sql_input
    def setData(self, data):
        self.data = data
        self._var_cache.clear()
        domain = data.domain if data else None
        self.disc_view.model().set_domain(domain)
        self.cont_view.model().set_domain(domain)
        self.disc_box.setVisible(
            domain is None or domain.has_discrete_attributes(include_metas=True))
        self.cont_box.setVisible(
            domain is None or domain.has_continuous_attributes(include_metas=True))
        self.class_box.setVisible(
            domain is None or domain.has_discrete_class)
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
        return reduce(
            add,
            (self._continuized_vars(var) if var.is_discrete
             else self._scaled_vars(var) if type(var) is ContinuousVariable
             else [var]
             for var in part),
             [])

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

    def _scaled_vars(self, var):
        hint = self.cont_var_hints.get(var.name, self.cont_var_hints[DefaultKey])
        if hint == Normalize.Leave:
            return [var]

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
        return [ContinuousVariable(
            var.name,
            compute_value=Normalizer(var, off, scale))]

    def _continuized_vars(self, var):
        hint = self.disc_var_hints.get(var.name, self.cont_var_hints[DefaultKey])

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
            base = -1 #assert False
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
