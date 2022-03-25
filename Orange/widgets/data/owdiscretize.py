import re
from collections import namedtuple
from enum import IntEnum
from typing import Optional, Tuple, Union, Callable, NamedTuple, Dict
import html

from AnyQt.QtCore import (
    Qt, QTimer, QPoint, QItemSelectionModel, QSize, QAbstractListModel
)
from AnyQt.QtGui import (QValidator, QPalette, QDoubleValidator, QIntValidator,
    QColor)
from AnyQt.QtWidgets import (
    QListView, QHBoxLayout, QStyledItemDelegate, QButtonGroup, QWidget,
    QLineEdit, QToolTip, QLabel, QApplication,
    QSpinBox, QSizePolicy, QRadioButton, QComboBox
)

from Orange.widgets.gui import createAttributePixmap
from Orange.widgets.utils.itemmodels import DomainModel
from orangewidget.settings import ContextHandler, Setting, ContextSetting
from orangewidget.utils.listview import ListViewSearch

import Orange.data
import Orange.preprocess.discretize as disc
from Orange.data import ContinuousVariable, DiscreteVariable, TimeVariable, \
    Domain, Table
from Orange.widgets import widget, gui
from Orange.widgets.utils import unique_everseen
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output
from Orange.widgets.data.oweditdomain import FixedSizeButton


re_custom_sep = re.compile(r"\s*,\s*")
time_units = ["year", "month", "day", "week", "hour", "minute", "second"]


def fixed_width_discretization(data, var, width):
    digits = len(width) - width.index(".") - 1 if "." in width else 0
    try:
        width = float(width)
    except ValueError:
        return "invalid width"
    if width <= 0:
        return "invalid width"
    try:
        return disc.FixedWidth(width, digits)(data, var)
    except disc.TooManyIntervals:
        return "too many intervals"


def fixed_time_width_discretization(data, var, width, unit):
    try:
        width = int(width)
    except ValueError:
        return "invalid width"
    if width <= 0:
        return "invalid width"
    if unit == 3:  # week
        width *= 7
    unit -= unit >= 3
    try:
        return disc.FixedTimeWidth(width, unit)(data, var)
    except disc.TooManyIntervals:
        return "too many intervals"


def custom_discretization(data, var, points):
    try:
        cuts = [float(x) for x in re_custom_sep.split(points.strip())]
    except ValueError:
        cuts = []
    if any(x >= y for x, y in zip(cuts, cuts[1:])):
        cuts = []
    if not cuts:
        return "invalid cuts"
    return disc.Discretizer.create_discretized_var(var, cuts)


class Methods(IntEnum):
    Default, Keep, MDL, EqualFreq, EqualWidth, Remove, Custom, Binning, \
        FixedWidth, FixedWidthTime = range(10)


class MethodDesc(NamedTuple):
    id_: Methods
    label: str
    short_desc: str
    tooltip: str
    function: Optional[Callable]
    controls: Tuple[str, ...] = ()


Options: Dict[Methods, MethodDesc] = {
    method.id_: method
    for method in (
        MethodDesc(Methods.Default,
                   "Use default setting", "default",
                   "Treat the variable as defined in 'default setting'",
                   None,
                   ()),
        MethodDesc(Methods.Keep,
                   "Keep numeric", "keep",
                   "Keep the variable as is",
                   lambda data, var: var,
                   ()),
        MethodDesc(Methods.MDL,
                   "Entropy vs. MDL", "entropy",
                   "Split values until MDL exceeds the entropy (Fayyad-Irani)\n"
                   "(requires discrete class variable)",
                   disc.EntropyMDL(),
                   ()),
        MethodDesc(Methods.EqualFreq,
                   "Equal frequency, intervals: ", "equal freq, k={}",
                   "Create bins with same number of instances",
                   lambda data, var, k: disc.EqualFreq(k)(data, var),
                   ("freq_spin", )),
        MethodDesc(Methods.EqualWidth,
                   "Equal width, intervals: ", "equal width, k={}",
                   "Create bins of the same width",
                   lambda data, var, k: disc.EqualWidth(k)(data, var),
                   ("width_spin", )),
        MethodDesc(Methods.Remove,
                   "Remove", "remove",
                   "Remove variable",
                   lambda *_: None,
                   ()),
        MethodDesc(Methods.Binning,
                   "Natural binning, desired bins: ", "binning, desired={}",
                   "Create bins with nice thresholds; "
                   "try matching desired number of bins",
                   lambda data, var, nbins: disc.Binning(nbins)(data, var),
                   ("binning_spin", )),
        MethodDesc(Methods.FixedWidth,
                   "Fixed width: ", "fixed width {}",
                   "Create bins with the given width (not for time variables)",
                   fixed_width_discretization,
                   ("width_line", )),
        MethodDesc(Methods.FixedWidthTime,
                   "Time interval: ", "time interval, {} {}",
                   "Create bins with the give width (for time variables)",
                   fixed_time_width_discretization,
                   ("width_time_line", "width_time_unit")),
        MethodDesc(Methods.Custom,
                   "Custom: ", "custom: {}",
                   "Use manually specified thresholds",
                   custom_discretization,
                   ("threshold_line", ))
    )
}


class VarHint(NamedTuple):
    """Description for settings"""
    method_id: Methods
    args: Tuple[Union[str, float, int]]


class DiscDesc(NamedTuple):
    """Data for list view model"""
    hint: VarHint
    points: str
    values: Tuple[str]


KeyType = Optional[Tuple[str, bool]]
DefaultHint = VarHint(Methods.Keep, ())
DefaultKey = None


def variable_key(var: ContinuousVariable) -> KeyType:
    return var.name, isinstance(var, TimeVariable)


class ListViewSearch(ListViewSearch):
    class DiscDelegate(QStyledItemDelegate):
        def initStyleOption(self, option, index):
            super().initStyleOption(option, index)
            option.font.setBold(index.data(Qt.UserRole).hint is not None)

    # Same name so that we can override a private method __layout
    def __init__(self, *args, **kwargs):
        self.default_view = None
        super().__init__(preferred_size=QSize(350, -1), *args, **kwargs)
        self.setItemDelegate(self.DiscDelegate(self))

    def select_default(self):
        index = self.default_view.model().index(0)
        self.default_view.selectionModel().select(index,
                                                  QItemSelectionModel.Select)

    def __layout(self):
        if self.default_view is None:  # __layout was called from __init__
            view = self.default_view = QListView(self)
            view.setModel(DefaultDiscModel())
            view.verticalScrollBar().setDisabled(True)
            view.horizontalScrollBar().setDisabled(True)
            view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
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


def format_desc(hint):
    if hint is None:
        return Options[Methods.Default].short_desc
    desc = Options[hint.method_id].short_desc
    if hint.method_id == Methods.FixedWidthTime:
        width, unit = hint.args
        try:
            width = int(width)
        except ValueError:
            pass
        else:
            return desc.format(width, time_units[unit] + "s" * (int(width) != 1))
    return desc.format(*hint.args)


class DiscDomainModel(DomainModel):
    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.ToolTipRole:
            var = self[index.row()]
            data = index.data(Qt.UserRole)
            tip = f"<b>{var.name}: </b>"
            values = map(html.escape, data.values)
            if not data.values:
                return None
            if len(data.values) <= 3:
                return f'<p style="white-space:pre">{tip}' \
                       f'{",&nbsp;&nbsp;".join(values)}</p>'
            else:
                return tip + "<br/>" \
                    + "".join(f"- {value}<br/>" for value in values)
        value = super().data(index, role)
        if role == Qt.DisplayRole:
            hint, points, values = index.data(Qt.UserRole)
            value += f" ({format_desc(hint)}){points}"
        return value


class DefaultDiscModel(QAbstractListModel):
    icon = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if DefaultDiscModel.icon is None:
            DefaultDiscModel.icon = createAttributePixmap("★", QColor(0, 0, 0, 0), Qt.black)
        self.hint: VarHint = DefaultHint

    def rowCount(self, parent):
        return 0 if parent.isValid() else 1

    def columnCount(self, parent):
        return 0 if parent.isValid() else 1

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return "Default setting: " + format_desc(self.hint)
        elif role == Qt.DecorationRole:
            return DefaultDiscModel.icon
        elif role == Qt.ToolTipRole:
            return "Default setting for variables without specific setings"

    def setData(self, index, value, role):
        if role == Qt.UserRole:
            self.hint = value
            self.dataChanged.emit(index, index)
        else:
            super().set_data(index, value, role)


class IncreasingNumbersListValidator(QValidator):
    def validate(self, string: str, pos: int) -> Tuple[QValidator.State, str, int]:
        for i, c in enumerate(string, start=1):
            if c not in "+-., 0123456789":
                return QValidator.Invalid, string, i
        prev = None
        if pos == len(string) >= 2 \
                and string[-1] == " " and string[-2].isdigit():
            string = string[:-1] + ", "
            pos += 1
        for valuestr in re_custom_sep.split(string.strip()):
            try:
                value = float(valuestr)
            except ValueError:
                return QValidator.Intermediate, string, pos
            if prev is not None and value <= prev:
                return QValidator.Intermediate, string, pos
            prev = value
        return QValidator.Acceptable, string, pos


def show_tip(
        widget: QWidget, pos: QPoint, text: str, timeout=-1,
        textFormat=Qt.AutoText, wordWrap=None):
    propname = __name__ + "::show_tip_qlabel"
    if timeout < 0:
        timeout = widget.toolTipDuration()
    if timeout < 0:
        timeout = 5000 + 40 * max(0, len(text) - 100)
    tip = widget.property(propname)
    if not text and tip is None:
        return

    def hide():
        w = tip.parent()
        w.setProperty(propname, None)
        tip.timer.stop()
        tip.close()
        tip.deleteLater()

    if not isinstance(tip, QLabel):
        tip = QLabel(objectName="tip-label", focusPolicy=Qt.NoFocus)
        tip.setBackgroundRole(QPalette.ToolTipBase)
        tip.setForegroundRole(QPalette.ToolTipText)
        tip.setPalette(QToolTip.palette())
        tip.setFont(QApplication.font("QTipLabel"))
        tip.setContentsMargins(2, 2, 2, 2)
        tip.timer = QTimer(tip, singleShot=True, objectName="hide-timer")
        tip.timer.timeout.connect(hide)
        widget.setProperty(propname, tip)
        tip.setParent(widget, Qt.ToolTip)

    tip.setText(text)
    tip.setTextFormat(textFormat)
    if wordWrap is None:
        wordWrap = textFormat != Qt.PlainText
    tip.setWordWrap(wordWrap)

    if not text:
        hide()
    else:
        tip.timer.start(timeout)
        tip.show()
        tip.move(pos)


class DiscretizeContextHandler(ContextHandler):
    def match(self, context, data: Table):
        if data is None:
            return self.NO_MATCH
        domain: Domain = data.domain
        types = (ContinuousVariable, TimeVariable)
        var_hints = context.values.get("var_hints")
        if var_hints is None:  # sanity check
            return self.NO_MATCH
        for key, hint in var_hints.items():
            if hint.method_id == Methods.MDL and not domain.has_discrete_class:
                return self.NO_MATCH
            if key is DefaultKey:
                continue
            name, tpe = key
            if name not in domain or not isinstance(domain[name], types[tpe]):
                return self.NO_MATCH
        return self.PERFECT_MATCH


# These are no longer used, but needed for loading and migrating old pickles.
# We insert them into namespace instead of normally defining them, in order
# to hide it from IDE's and avoid mistakenly using them.
globals().update(dict(
    DState=namedtuple(
        "DState",
        ["method",    # discretization method
         "points",    # induced cut points
         "disc_var"]  # induced discretized variable
    ),
    Default=namedtuple("Default", ["method"]),
    Leave=namedtuple("Leave", []),
    MDL=namedtuple("MDL", []),
    EqualFreq=namedtuple("EqualFreq", ["k"]),
    EqualWidth=namedtuple("EqualWidth", ["k"]),
    Remove=namedtuple("Remove", []),
    Custom=namedtuple("Custom", ["points"])
))


class OWDiscretize(widget.OWWidget):
    # pylint: disable=too-many-instance-attributes
    name = "Discretize"
    description = "Discretize numeric variables"
    category = "Transform"
    icon = "icons/Discretize.svg"
    keywords = ["bin", "categorical", "nominal", "ordinal"]
    priority = 2130

    class Inputs:
        data = Input("Data", Table, doc="Input data table")

    class Outputs:
        data = Output("Data", Table, doc="Table with categorical features")

    settingsHandler = DiscretizeContextHandler()
    settings_version = 3

    #: Default setting (key DefaultKey) and specific settings for variables;
    # if variable is not in the dict, it uses default
    var_hints: Dict[KeyType, VarHint] = ContextSetting({DefaultKey: DefaultHint})
    autosend = Setting(True)

    want_main_area = False

    def __init__(self):
        super().__init__()

        #: input data
        self.data = None
        #: Cached discretized variables
        self.discretized_vars: Dict[KeyType, DiscreteVariable] = {}

        box = gui.hBox(self.controlArea, True, spacing=8)
        self._create_var_list(box)
        self._create_buttons(box)
        gui.auto_apply(self.buttonsArea, self, "autosend")
        gui.rubber(self.buttonsArea)

        self.varview.select_default()

    def _create_var_list(self, box):
        # If we don't elide, remove the `uniformItemSize` argument
        self.varview = ListViewSearch(
            selectionMode=QListView.ExtendedSelection, uniformItemSizes=True)
        self.varview.setModel(
            DiscDomainModel(
                valid_types=(ContinuousVariable, TimeVariable),
                order=DiscDomainModel.MIXED
            ))
        self.varview.selectionModel().selectionChanged.connect(
            self._var_selection_changed)
        self.varview.default_view.selectionModel().selectionChanged.connect(
            self._default_selected)
        self._set_default_data()
        box.layout().addWidget(self.varview)

    def _create_buttons(self, box):
        def intspin():
            s = QSpinBox(self)
            s.setMinimum(2)
            s.setMaximum(10)
            s.setFixedWidth(60)
            s.setAlignment(Qt.AlignRight)
            s.setContentsMargins(0, 0, 0, 0)
            return s, s.valueChanged

        def widthline(validator):
            s = QLineEdit(self)
            s.setFixedWidth(60)
            s.setAlignment(Qt.AlignRight)
            s.setValidator(validator)
            s.setContentsMargins(0, 0, 0, 0)
            return s, s.textChanged

        def manual_cut_editline(text="", enabled=True) -> QLineEdit:
            edit = QLineEdit(
                text=text,
                placeholderText="e.g. 0.0, 0.5, 1.0",
                toolTip='<p style="white-space:pre">' +
                        'Enter cut points as a comma-separate list of \n'
                        'strictly increasing numbers e.g. 0.0, 0.5, 1.0).</p>',
                enabled=enabled,
            )
            edit.setValidator(IncreasingNumbersListValidator())
            edit.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)

            @edit.textChanged.connect
            def update():
                validator = edit.validator()
                if validator is not None and edit.text().strip():
                    state, _, _ = validator.validate(edit.text(), 0)
                else:
                    state = QValidator.Acceptable
                palette = edit.palette()
                colors = {
                    QValidator.Intermediate: (Qt.yellow, Qt.black),
                    QValidator.Invalid: (Qt.red, Qt.black),
                }.get(state, None)
                if colors is None:
                    palette = QPalette()
                else:
                    palette.setColor(QPalette.Base, colors[0])
                    palette.setColor(QPalette.Text, colors[1])

                cr = edit.cursorRect()
                p = edit.mapToGlobal(cr.bottomRight())
                edit.setPalette(palette)
                if state != QValidator.Acceptable and edit.isVisible():
                    show_tip(edit, p, edit.toolTip(), textFormat=Qt.RichText)
                else:
                    show_tip(edit, p, "")
            return edit, edit.textChanged

        children = []

        def button(id_, *controls, stretch=True):
            layout = QHBoxLayout()
            desc = Options[id_]
            button = QRadioButton(desc.label)
            button.setToolTip(desc.tooltip)
            self.button_group.addButton(button, id_)
            layout.addWidget(button)
            if controls:
                if stretch:
                    layout.addStretch(1)
                for c, signal in controls:
                    layout.addWidget(c)
                    if signal is not None:
                        @signal.connect
                        def arg_changed():
                            self.button_group.button(id_).setChecked(True)
                            self.update_hints(id_)

            children.append(layout)
            button_box.layout().addLayout(layout)
            return (*controls, (None, ))[0][0]

        button_box = gui.vBox(box)
        button_box.layout().setSpacing(0)
        self.button_group = QButtonGroup(self)
        self.button_group.idClicked.connect(self.update_hints)

        button(Methods.Keep)
        button(Methods.Remove)

        self.binning_spin = button(Methods.Binning, intspin())
        validator = QDoubleValidator()
        validator.setBottom(0)
        self.width_line = button(Methods.FixedWidth, widthline(validator))

        self.width_time_unit = u = QComboBox(self)
        u.setContentsMargins(0, 0, 0, 0)
        u.addItems([unit + "(s)" for unit in time_units])
        validator = QIntValidator()
        validator.setBottom(1)
        self.width_time_line = button(Methods.FixedWidthTime,
                                      widthline(validator),
                                      (u, u.currentTextChanged))

        self.freq_spin = button(Methods.EqualFreq, intspin())
        self.width_spin = button(Methods.EqualWidth, intspin())
        button(Methods.MDL)

        self.copy_to_custom = FixedSizeButton(
            text="CC", toolTip="Copy the current cut points to manual mode")
        self.copy_to_custom.clicked.connect(self._copy_to_manual)
        self.threshold_line = button(Methods.Custom,
                                     manual_cut_editline(),
                                     (self.copy_to_custom, None),
                                     stretch=False)
        button(Methods.Default)
        maxheight = max(w.sizeHint().height() for w in children)
        for w in children:
            w.itemAt(0).widget().setFixedHeight(maxheight)
        button_box.layout().addStretch(1)

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.data = data

        self.var_hints = {DefaultKey: DefaultHint}
        self.discretized_vars = {}

        if self.data is not None:
            self.varview.model().set_domain(data.domain)
            self.openContext(data)
            self._update_points()
        else:
            self.varview.model().set_domain(None)

        self._set_default_data()
        self._set_mdl_button()
        self.commit.now()

    def _get_values(self, controls: Tuple[str]):
        values = []
        for control_name in controls:
            control = getattr(self, control_name)
            if isinstance(control, QSpinBox):
                values.append(control.value())
            elif isinstance(control, QComboBox):
                values.append(control.currentIndex())
            else:
                values.append(control.text())
        return tuple(values)

    def _set_values(self, controls: Tuple[str],
                   values: Tuple[Union[str, int, float]]):
        for control_name, value in zip(controls, values):
            control = getattr(self, control_name)
            if isinstance(control, QSpinBox):
                control.setValue(value)
            elif isinstance(control, QComboBox):
                control.setCurrentIndex(value)
            else:
                control.setText(value)

    def _set_mdl_button(self):
        mdl_button = self.button_group.button(Methods.MDL)
        if self.data is None or self.data.domain.has_discrete_class:
            mdl_button.setEnabled(True)
        else:
            if mdl_button.isChecked():
                self._set_button(Methods.Keep, True)
            mdl_button.setEnabled(False)

    def varkeys_for_selection(self):
        model = self.varview.model()
        varkeys = [variable_key(model[i]) for i in self.selected_indices()]
        return varkeys or [DefaultKey]  # default settings are selected

    def update_hints(self, method_id):
        args = self._get_values(Options[method_id].controls)
        keys = self.varkeys_for_selection()
        if method_id == Methods.Default:
            for key in keys:
                if key in self.var_hints:
                    del self.var_hints[key]
        else:
            self.var_hints.update(dict.fromkeys(keys, VarHint(method_id, args)))
        if keys == [DefaultKey]:
            invalidate = set(self.discretized_vars) - set(self.var_hints)
        else:
            invalidate = keys
        for key in invalidate:
            del self.discretized_vars[key]

        if keys == [DefaultKey]:
            self._set_default_data()
        self._update_points()

    def _set_default_data(self):
        model = self.varview.default_view.model()
        model.setData(model.index(0), self.var_hints[DefaultKey], Qt.UserRole)

    def _update_points(self):
        if self.data is None:
            return

        def induce_cuts(data, var, hint: VarHint):
            if isinstance(var, TimeVariable):
                if hint.method_id in (Methods.FixedWidth, Methods.Custom):
                    return ": <keep, time var>", var
            else:
                if hint.method_id == Methods.FixedWidthTime:
                    return ": <keep, not time>", var

            function = Options[hint.method_id].function
            dvar = function(data, var, *hint.args)
            if isinstance(dvar, str):
                return f" <{dvar}>", None  # error
            if dvar is None:
                return "", None  # removed
            elif dvar is var:
                return "", var  # no transformation
            elif isinstance(dvar.compute_value, disc.Discretizer):
                points = dvar.compute_value.points
                if len(points) == 0:
                    return " <removed>", None
                return ": " + ", ".join(map(var.repr_val, points)), dvar
            raise ValueError

        default_hint = self.var_hints[DefaultKey]
        model = self.varview.model()
        for index, var in enumerate(model):
            key = variable_key(var)
            if key not in self.discretized_vars:
                var_hint = self.var_hints.get(key)
                points, dvar = induce_cuts(self.data, var, var_hint or default_hint)
                self.discretized_vars[key] = dvar
                values = getattr(dvar, "values", ())
                model.setData(model.index(index),
                              DiscDesc(var_hint, points, values),
                              Qt.UserRole)

        self.commit.deferred()

    def _copy_to_manual(self):
        varkeys = self.varkeys_for_selection()
        texts = set()
        for key in varkeys:
            dvar = self.discretized_vars.get(key)
            fmt = self.data.domain[key[0]].repr_val
            if isinstance(dvar, DiscreteVariable):
                text = ", ".join(map(fmt, dvar.compute_value.points))
                texts.add(text)
                self.var_hints[key] = VarHint(Methods.Custom, (text, ))
                del self.discretized_vars[key]
        if len(texts) == 1:
            self.threshold_line.setText(texts.pop())
        else:
            self._uncheck_all_radios()
        self._update_points()

    def _set_button(self, method_id, checked):
        self.button_group.button(method_id).setChecked(checked)

    def _default_selected(self, selected):
        if not selected:
            return
        self.varview.selectionModel().clearSelection()
        self.set_method_state([DefaultKey])
        self.enable_radios([DefaultKey])

    def _var_selection_changed(self, selected):
        if not selected:
            return
        self.varview.default_view.selectionModel().clearSelection()
        indices = self.selected_indices()
        # set of all methods for the current selection
        model = self.varview.model()
        keys = [variable_key(model[i]) for i in indices]
        self.set_method_state(keys)
        self.enable_radios(keys)

    def enable_radios(self, keys):
        def set_enabled(method_id, value):
            self.button_group.button(method_id).setEnabled(value)
            for control_name in Options[method_id].controls:
                getattr(self, control_name).setEnabled(value)

        if keys == [DefaultKey]:
            set_enabled(Methods.Default, False)
            set_enabled(Methods.FixedWidth, True)
            set_enabled(Methods.FixedWidthTime, True)
            set_enabled(Methods.Custom, True)
        else:
            set_enabled(Methods.Default, True)
            vars_ = [self.data.domain[name] for name, _ in keys]
            no_time = not any(isinstance(var, TimeVariable) for var in vars_)
            set_enabled(Methods.FixedWidth, no_time)
            set_enabled(Methods.Custom, no_time)
            set_enabled(Methods.FixedWidthTime,
                        all(isinstance(var, TimeVariable) for var in vars_))

    def set_method_state(self, keys):
        mset = list(unique_everseen(map(self.var_hints.get, keys)))
        if len(mset) == 1:
            if mset == [None]:
                method_id, args = Methods.Default, ()
            else:
                method_id, args = mset.pop()
            method = Options[method_id]
            self._set_button(method_id, True)
            self._set_values(method.controls, args)
            if method_id != Methods.Custom:
                self.threshold_line.setText("")
        else:
            self._uncheck_all_radios()

    def _uncheck_all_radios(self):
        group = self.button_group
        button = group.checkedButton()
        if button is not None:
            group.setExclusive(False)
            button.setChecked(False)
            group.setExclusive(True)

    def selected_indices(self):
        rows = self.varview.selectionModel().selectedRows()
        return [index.row() for index in rows]

    def discretized_domain(self):
        def disc_var(var: ContinuousVariable) -> Optional[DiscreteVariable]:
            return self.discretized_vars.get(variable_key(var), var)

        if self.data is None:
            return None

        attributes = (disc_var(v) for v in self.data.domain.attributes)
        attributes = [v for v in attributes if v is not None]

        class_vars = (disc_var(v) for v in self.data.domain.class_vars)
        class_vars = [v for v in class_vars if v is not None]

        return Domain(attributes, class_vars, metas=self.data.domain.metas)

    @gui.deferred
    def commit(self):
        output = None
        if self.data is not None:
            domain = self.discretized_domain()
            output = self.data.transform(domain)
        self.Outputs.data.send(output)

    def send_report(self):
        dmodel = self.varview.default_view.model()
        desc = dmodel.data(dmodel.index(0))
        self.report_items((tuple(desc.split(": ", maxsplit=1)), ))
        model = self.varview.model()
        reported = []
        for row in range(model.rowCount()):
            name = model[row].name
            desc = model.data(model.index(row), Qt.UserRole)
            if desc.hint is not None:
                name = f"{name} ({format_desc(desc.hint)})"
            reported.append((name, ', '.join(desc.values)))
        self.report_items("Variables", reported)

    @classmethod
    def migrate_settings(cls, settings, version):
        if version is None or version < 2:
            # was stored as int indexing Methods (but offset by 1)
            default = settings.pop("default_method", 0)
            default = Methods(default + 1)
            settings["default_method_name"] = default.name

        if version is None or version < 3:
            method_name = settings.pop("default_method_name",
                                       DefaultHint.method_id.name)
            k = settings.pop("default_k", 3)
            cut_points = settings.pop("default_cutpoints", ())

            method_id = getattr(Methods, method_name)
            if method_id in (Methods.EqualFreq, Methods.EqualWidth):
                args = (k, )
            elif method_id == Methods.Custom:
                args = (cut_points, )
            else:
                args = ()
            default_hint = VarHint(method_id, args)
            for context in settings.get("context_settings", []):
                values = context.values
                if "saved_var_states" not in values:
                    continue
                var_states, _ = values.pop("saved_var_states")
                var_hints = {DefaultKey: default_hint}
                for (tpe, name), dstate in var_states.items():
                    key = (name, tpe == 4)  # time variable == 4
                    method = dstate.method
                    method_name = type(method).__name__.replace("Leave", "Keep")
                    if method_name == "Default":
                        continue
                    if method_name == "Custom":
                        args = (", ".join(f"{x:g}" for x in method.points), )
                    else:
                        args = tuple(method)
                    var_hints[key] = VarHint(getattr(Methods, method_name), args)
                values["var_hints"] = var_hints


if __name__ == "__main__":  # pragma: no cover
    #WidgetPreview(OWDiscretize).run(Table("/Users/janez/Downloads/banking-crises.tab"))
    WidgetPreview(OWDiscretize).run(Table("heart_disease"))
