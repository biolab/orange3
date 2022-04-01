import re
from enum import IntEnum
from collections import namedtuple
from typing import Optional, Tuple, Iterable, Union, Callable, Any

from AnyQt.QtWidgets import (
    QListView, QHBoxLayout, QStyledItemDelegate, QButtonGroup, QWidget,
    QLineEdit, QToolTip, QLabel, QApplication
)
from AnyQt.QtGui import QValidator, QPalette
from AnyQt.QtCore import Qt, QTimer, QPoint
from orangewidget.utils.listview import ListViewSearch

import Orange.data
import Orange.preprocess.discretize as disc
from Orange.data import Variable
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels, vartype, unique_everseen
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output
from Orange.widgets.data.oweditdomain import FixedSizeButton

__all__ = ["OWDiscretize"]

# 'Default' method delegates to 'method'
Default = namedtuple("Default", ["method"])
Leave = namedtuple("Leave", [])
MDL = namedtuple("MDL", [])
EqualFreq = namedtuple("EqualFreq", ["k"])
EqualWidth = namedtuple("EqualWidth", ["k"])
Remove = namedtuple("Remove", [])
Custom = namedtuple("Custom", ["points"])


MethodType = Union[
    Default,
    Leave,
    MDL,
    EqualFreq,
    EqualWidth,
    Remove,
    Custom,
]

_dispatch = {
    Default:
        lambda m, data, var: _dispatch[type(m.method)](m.method, data, var),
    Leave: lambda m, data, var: var,
    MDL: lambda m, data, var: disc.EntropyMDL()(data, var),
    EqualFreq: lambda m, data, var: disc.EqualFreq(m.k)(data, var),
    EqualWidth: lambda m, data, var: disc.EqualWidth(m.k)(data, var),
    Remove: lambda m, data, var: None,
    Custom:
        lambda m, data, var:
        disc.Discretizer.create_discretized_var(var, m.points)
}

# Variable discretization state (back compat for deserialization)
DState = namedtuple(
    "DState",
    ["method",    # discretization method
     "points",    # induced cut points
     "disc_var"]  # induced discretized variable
)


def is_discretized(var):
    return isinstance(var.compute_value, disc.Discretizer)


def variable_key(var):
    return vartype(var), var.name


def button_group_reset(group):
    button = group.checkedButton()
    if button is not None:
        group.setExclusive(False)
        button.setChecked(False)
        group.setExclusive(True)


class DiscDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        state = index.data(Qt.UserRole)
        var = index.data(Qt.EditRole)

        if state is not None:
            if isinstance(var, Variable):
                fmt = var.repr_val
            else:
                fmt = str
            extra = self.cutsText(state, fmt)
            option.text = option.text + ": " + extra

    @staticmethod
    def cutsText(state: DState, fmt: Callable[[Any], str] = str):
        # This function has many branches, but they don't hurt readabability
        # pylint: disable=too-many-branches
        method = state.method
        # Need a better way to distinguish discretization states
        # i.e. between 'induced no points v.s. 'removed by choice'
        if state.points is None and state.disc_var is not None:
            points = ""
        elif state.points is None:
            points = "..."
        elif state.points == []:
            points = "<removed>"
        else:
            points = ", ".join(map(fmt, state.points))

        if isinstance(method, Default):
            name = None
        elif isinstance(method, Leave):
            name = "(leave)"
        elif isinstance(method, MDL):
            name = "(entropy)"
        elif isinstance(method, EqualFreq):
            name = "(equal frequency k={})".format(method.k)
        elif isinstance(method, EqualWidth):
            name = "(equal width k={})".format(method.k)
        elif isinstance(method, Remove):
            name = "(removed)"
        elif isinstance(method, Custom):
            name = "(custom)"
        else:
            assert False

        if name is not None:
            return points + " " + name
        else:
            return points


#: Discretization methods
class Methods(IntEnum):
    Default, Leave, MDL, EqualFreq, EqualWidth, Remove, Custom = range(7)

    @staticmethod
    def from_method(method):
        return Methods[type(method).__name__]


def parse_float(string: str) -> Optional[float]:
    try:
        return float(string)
    except ValueError:
        return None


class IncreasingNumbersListValidator(QValidator):
    """
    Match a comma separated list of non-empty and increasing number strings.

    Example
    -------
    >>> v = IncreasingNumbersListValidator()
    >>> v.validate("", 0)   # Acceptable
    (2, '', 0)
    >>> v.validate("1", 1)  # Acceptable
    (2, '1', 1)
    >>> v.validate("1,,", 1)  # Intermediate
    (1, '1,,', 1)
    """
    @staticmethod
    def itersplit(string: str) -> Iterable[Tuple[int, int]]:
        sepiter = re.finditer(r"(?<!\\),", string)
        start = 0
        for match in sepiter:
            yield start, match.start()
            start = match.end()
        # yield the rest if any
        if start < len(string):
            yield start, len(string)

    def validate(self, string: str, pos: int) -> Tuple[QValidator.State, str, int]:
        state = QValidator.Acceptable
        # Matches non-complete intermediate numbers (while editing)
        intermediate = re.compile(r"([+-]?\s?\d*\s?\d*\.?\d*\s?\d*)")
        values = []
        for start, end in self.itersplit(string):
            valuestr = string[start:end].strip()
            if not valuestr:
                # Middle element is empty (will be fixed by fixup)
                continue
            value = parse_float(valuestr)
            if value is None:
                if intermediate.fullmatch(valuestr):
                    state = min(state, QValidator.Intermediate)
                    continue
                return QValidator.Invalid, string, pos
            if values and value <= values[-1]:
                state = min(state, QValidator.Intermediate)
            else:
                values.append(value)
        return state, string, pos

    def fixup(self, string):
        # type: (str) -> str
        """
        Fixup the input. Remove empty parts from the string.
        """
        parts = [string[start: end] for start, end in self.itersplit(string)]
        parts = [part for part in parts if part.strip()]
        return ", ".join(parts)


def show_tip(
        widget: QWidget, pos: QPoint, text: str, timeout=-1,
        textFormat=Qt.AutoText, wordWrap=None
):
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


class OWDiscretize(widget.OWWidget):
    # pylint: disable=too-many-instance-attributes
    name = "Discretize"
    description = "Discretize the numeric data features."
    category = "Transform"
    icon = "icons/Discretize.svg"
    keywords = ["bin", "categorical", "nominal", "ordinal"]
    priority = 2130

    class Inputs:
        data = Input("Data", Orange.data.Table, doc="Input data table")

    class Outputs:
        data = Output("Data", Orange.data.Table, doc="Table with discretized features")

    settingsHandler = settings.DomainContextHandler()
    settings_version = 2
    saved_var_states = settings.ContextSetting({})

    #: The default method name
    default_method_name = settings.Setting(Methods.EqualFreq.name)
    #: The k for Equal{Freq,Width}
    default_k = settings.Setting(3)
    #: The default cut points for custom entry
    default_cutpoints: Tuple[float, ...] = settings.Setting(())
    autosend = settings.Setting(True)

    #: Discretization methods
    Default, Leave, MDL, EqualFreq, EqualWidth, Remove, Custom = list(Methods)

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()

        #: input data
        self.data = None
        self.class_var = None
        #: Current variable discretization state
        self.var_state = {}
        #: Saved variable discretization settings (context setting)
        self.saved_var_states = {}

        self.method = Methods.Default
        self.k = 5
        self.cutpoints = ()

        box = gui.vBox(self.controlArea, self.tr("Default Discretization"))
        self._default_method_ = 0
        self.default_bbox = rbox = gui.radioButtons(
            box, self, "_default_method_", callback=self._default_disc_changed)
        self.default_button_group = bg = rbox.findChild(QButtonGroup)
        bg.buttonClicked[int].connect(self.set_default_method)

        rb = gui.hBox(rbox)
        self.left = gui.vBox(rb)
        right = gui.vBox(rb)
        rb.layout().setStretch(0, 1)
        rb.layout().setStretch(1, 1)
        self.options = [
            (Methods.Default, self.tr("Default")),
            (Methods.Leave, self.tr("Leave numeric")),
            (Methods.MDL, self.tr("Entropy-MDL discretization")),
            (Methods.EqualFreq, self.tr("Equal-frequency discretization")),
            (Methods.EqualWidth, self.tr("Equal-width discretization")),
            (Methods.Remove, self.tr("Remove numeric variables")),
            (Methods.Custom, self.tr("Manual")),
        ]

        for id_, opt in self.options[1:]:
            t = gui.appendRadioButton(rbox, opt)
            bg.setId(t, id_)
            t.setChecked(id_ == self.default_method)
            [right, self.left][opt.startswith("Equal")].layout().addWidget(t)

        def _intbox(parent, attr, callback):
            box = gui.indentedBox(parent)
            s = gui.spin(
                box, self, attr, minv=2, maxv=10, label="Num. of intervals:",
                callback=callback)
            s.setMaximumWidth(60)
            s.setAlignment(Qt.AlignRight)
            gui.rubber(s.box)
            return box.box

        self.k_general = _intbox(self.left, "default_k",
                                 self._default_disc_changed)
        self.k_general.layout().setContentsMargins(0, 0, 0, 0)

        def manual_cut_editline(text="", enabled=True) -> QLineEdit:
            edit = QLineEdit(
                text=text,
                placeholderText="e.g. 0.0, 0.5, 1.0",
                toolTip="Enter fixed discretization cut points (a comma "
                        "separated list of strictly increasing numbers e.g. "
                        "0.0, 0.5, 1.0).",
                enabled=enabled,
            )
            @edit.textChanged.connect
            def update():
                validator = edit.validator()
                if validator is not None:
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
            return edit

        self.manual_cuts_edit = manual_cut_editline(
            text=", ".join(map(str, self.default_cutpoints)),
            enabled=self.default_method == Methods.Custom,
        )

        def set_manual_default_cuts():
            text = self.manual_cuts_edit.text()
            self.default_cutpoints = tuple(
                float(s.strip()) for s in text.split(",") if s.strip())
            self._default_disc_changed()
        self.manual_cuts_edit.editingFinished.connect(set_manual_default_cuts)

        validator = IncreasingNumbersListValidator()
        self.manual_cuts_edit.setValidator(validator)
        ibox = gui.indentedBox(right, orientation=Qt.Horizontal)
        ibox.layout().addWidget(self.manual_cuts_edit)

        right.layout().addStretch(10)
        self.left.layout().addStretch(10)

        self.connect_control(
            "default_cutpoints",
            lambda values: self.manual_cuts_edit.setText(", ".join(map(str, values)))
        )
        vlayout = QHBoxLayout()
        box = gui.widgetBox(
            self.controlArea, "Individual Attribute Settings",
            orientation=vlayout, spacing=8
        )

        # List view with all attributes
        self.varview = ListViewSearch(
            selectionMode=QListView.ExtendedSelection,
            uniformItemSizes=True,
        )
        self.varview.setItemDelegate(DiscDelegate())
        self.varmodel = itemmodels.VariableListModel()
        self.varview.setModel(self.varmodel)
        self.varview.selectionModel().selectionChanged.connect(
            self._var_selection_changed
        )

        vlayout.addWidget(self.varview)
        # Controls for individual attr settings
        self.bbox = controlbox = gui.radioButtons(
            box, self, "method", callback=self._disc_method_changed
        )
        vlayout.addWidget(controlbox)
        self.variable_button_group = bg = controlbox.findChild(QButtonGroup)
        for id_, opt in self.options[:5]:
            b = gui.appendRadioButton(controlbox, opt)
            bg.setId(b, id_)

        self.k_specific = _intbox(controlbox, "k", self._disc_method_changed)

        gui.appendRadioButton(controlbox, "Remove attribute", id=Methods.Remove)
        b = gui.appendRadioButton(controlbox, "Manual", id=Methods.Custom)

        self.manual_cuts_specific = manual_cut_editline(
            text=", ".join(map(str, self.cutpoints)),
            enabled=self.method == Methods.Custom
        )
        self.manual_cuts_specific.setValidator(validator)
        b.toggled[bool].connect(self.manual_cuts_specific.setEnabled)

        def set_manual_cuts():
            text = self.manual_cuts_specific.text()
            points = [t for t in text.split(",") if t.split()]
            self.cutpoints = tuple(float(t) for t in points)
            self._disc_method_changed()
        self.manual_cuts_specific.editingFinished.connect(set_manual_cuts)

        self.connect_control(
            "cutpoints",
            lambda values: self.manual_cuts_specific.setText(", ".join(map(str, values)))
        )
        ibox = gui.indentedBox(controlbox, orientation=Qt.Horizontal)
        self.copy_current_to_manual_button = b = FixedSizeButton(
            text="CC", toolTip="Copy the current cut points to manual mode",
            enabled=False
        )
        b.clicked.connect(self._copy_to_manual)
        ibox.layout().addWidget(self.manual_cuts_specific)
        ibox.layout().addWidget(b)

        gui.rubber(controlbox)
        controlbox.setEnabled(False)
        bg.button(self.method)
        self.controlbox = controlbox

        gui.auto_apply(self.buttonsArea, self, "autosend")

        self._update_spin_positions()

    @property
    def default_method(self) -> Methods:
        return Methods[self.default_method_name]

    @default_method.setter
    def default_method(self, method):
        self.set_default_method(method)

    def set_default_method(self, method: Methods):
        if isinstance(method, int):
            method = Methods(method)
        else:
            method = Methods.from_method(method)

        if method != self.default_method:
            self.default_method_name = method.name
            self.default_button_group.button(method).setChecked(True)
            self._default_disc_changed()
        self.manual_cuts_edit.setEnabled(method == Methods.Custom)

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.data = data
        if self.data is not None:
            self._initialize(data)
            self.openContext(data)
            # Restore the per variable discretization settings
            self._restore(self.saved_var_states)
            # Complete the induction of cut points
            self._update_points()
        else:
            self._clear()
        self.commit.now()

    def _initialize(self, data):
        # Initialize the default variable states for new data.
        self.class_var = data.domain.class_var
        cvars = [var for var in data.domain.variables
                 if var.is_continuous]
        self.varmodel[:] = cvars

        has_disc_class = data.domain.has_discrete_class

        def set_enabled(box: QWidget, id_: Methods, state: bool):
            bg = box.findChild(QButtonGroup)
            b = bg.button(id_)
            b.setEnabled(state)

        set_enabled(self.default_bbox, self.MDL, has_disc_class)
        bg = self.bbox.findChild(QButtonGroup)
        b = bg.button(Methods.MDL)
        b.setEnabled(has_disc_class)
        set_enabled(self.bbox, self.MDL, has_disc_class)

        # If the newly disabled MDL button is checked then change it
        if not has_disc_class and self.default_method == self.MDL:
            self.default_method = Methods.Leave
        if not has_disc_class and self.method == self.MDL:
            self.method = Methods.Default

        # Reset (initialize) the variable discretization states.
        self._reset()

    def _restore(self, saved_state):
        # Restore variable states from a saved_state dictionary.
        def_method = self._current_default_method()
        for i, var in enumerate(self.varmodel):
            key = variable_key(var)
            if key in saved_state:
                state = saved_state[key]
                if isinstance(state.method, Default):
                    state = DState(Default(def_method), None, None)
                self._set_var_state(i, state)

    def _reset(self):
        # restore the individual variable settings back to defaults.
        def_method = self._current_default_method()
        self.var_state = {}
        for i in range(len(self.varmodel)):
            state = DState(Default(def_method), None, None)
            self._set_var_state(i, state)

    def _set_var_state(self, index, state):
        # set the state of variable at `index` to `state`.
        self.var_state[index] = state
        self.varmodel.setData(self.varmodel.index(index), state, Qt.UserRole)

    def _clear(self):
        self.data = None
        self.varmodel[:] = []
        self.var_state = {}
        self.saved_var_states = {}
        self.default_button_group.button(self.MDL).setEnabled(True)
        self.variable_button_group.button(self.MDL).setEnabled(True)

    def _update_points(self):
        """
        Update the induced cut points.
        """
        if self.data is None:
            return

        def induce_cuts(method, data, var):
            dvar = _dispatch[type(method)](method, data, var)
            if dvar is None:
                # removed
                return [], None
            elif dvar is var:
                # no transformation took place
                return None, var
            elif is_discretized(dvar):
                return dvar.compute_value.points, dvar
            raise ValueError

        for i, var in enumerate(self.varmodel):
            state = self.var_state[i]
            if state.points is None and state.disc_var is None:
                points, dvar = induce_cuts(state.method, self.data, var)
                new_state = state._replace(points=points, disc_var=dvar)
                self._set_var_state(i, new_state)

    def _current_default_method(self):
        method = self.default_method
        k = self.default_k
        if method == Methods.Leave:
            def_method = Leave()
        elif method == Methods.MDL:
            def_method = MDL()
        elif method == Methods.EqualFreq:
            def_method = EqualFreq(k)
        elif method == Methods.EqualWidth:
            def_method = EqualWidth(k)
        elif method == Methods.Remove:
            def_method = Remove()
        elif method == Methods.Custom:
            def_method = Custom(self.default_cutpoints)
        else:
            assert False
        return def_method

    def _current_method(self):
        if self.method == Methods.Default:
            method = Default(self._current_default_method())
        elif self.method == Methods.Leave:
            method = Leave()
        elif self.method == Methods.MDL:
            method = MDL()
        elif self.method == Methods.EqualFreq:
            method = EqualFreq(self.k)
        elif self.method == Methods.EqualWidth:
            method = EqualWidth(self.k)
        elif self.method == Methods.Remove:
            method = Remove()
        elif self.method == Methods.Custom:
            method = Custom(self.cutpoints)
        else:
            assert False
        return method

    def _update_spin_positions(self):
        kmethods = [Methods.EqualFreq, Methods.EqualWidth]
        self.k_general.setDisabled(self.default_method not in kmethods)
        if self.default_method == Methods.EqualFreq:
            self.left.layout().insertWidget(1, self.k_general)
        elif self.default_method == Methods.EqualWidth:
            self.left.layout().insertWidget(2, self.k_general)

        self.k_specific.setDisabled(self.method not in kmethods)
        if self.method == Methods.EqualFreq:
            self.bbox.layout().insertWidget(4, self.k_specific)
        elif self.method == Methods.EqualWidth:
            self.bbox.layout().insertWidget(5, self.k_specific)

    def _default_disc_changed(self):
        self._update_spin_positions()
        method = self._current_default_method()
        state = DState(Default(method), None, None)
        for i, _ in enumerate(self.varmodel):
            if isinstance(self.var_state[i].method, Default):
                self._set_var_state(i, state)
        self._update_points()
        self.commit.deferred()

    def _disc_method_changed(self):
        self._update_spin_positions()
        indices = self.selected_indices()
        method = self._current_method()
        state = DState(method, None, None)
        for idx in indices:
            self._set_var_state(idx, state)
        self._update_points()
        self._copy_to_manual_update_enabled()
        self.commit.deferred()

    def _copy_to_manual(self):
        indices = self.selected_indices()
        # set of all methods for the current selection
        if len(indices) != 1:
            return
        index = indices[0]
        state = self.var_state[index]
        var = self.varmodel[index]
        fmt = var.repr_val
        points = state.points
        if points is None:
            points = ()
        else:
            points = tuple(state.points)
        state = state._replace(method=Custom(points), points=None, disc_var=None)
        self._set_var_state(index, state)
        self.method = Methods.Custom
        self.cutpoints = points
        self.manual_cuts_specific.setText(", ".join(map(fmt, points)))
        self._update_points()
        self.commit.deferred()

    def _copy_to_manual_update_enabled(self):
        indices = self.selected_indices()
        methods = [self.var_state[i].method for i in indices]
        self.copy_current_to_manual_button.setEnabled(
            len(indices) == 1 and not isinstance(methods[0], Custom))

    def _var_selection_changed(self, *_):
        self._copy_to_manual_update_enabled()
        indices = self.selected_indices()
        # set of all methods for the current selection
        methods = [self.var_state[i].method for i in indices]

        def key(method):
            if isinstance(method, Default):
                return Default, (None, )
            return type(method), tuple(method)

        mset = list(unique_everseen(methods, key=key))

        self.controlbox.setEnabled(len(mset) > 0)
        if len(mset) == 1:
            method = mset.pop()
            self.method = Methods.from_method(method)
            if isinstance(method, (EqualFreq, EqualWidth)):
                self.k = method.k
            elif isinstance(method, Custom):
                self.cutpoints = method.points
        else:
            # deselect the current button
            self.method = -1
            bg = self.controlbox.group
            button_group_reset(bg)
        self._update_spin_positions()

    def selected_indices(self):
        rows = self.varview.selectionModel().selectedRows()
        return [index.row() for index in rows]

    def method_for_index(self, index):
        state = self.var_state[index]
        return state.method

    def discretized_var(self, index):
        # type: (int) -> Optional[Orange.data.DiscreteVariable]
        state = self.var_state[index]
        if state.disc_var is not None and state.points == []:
            # Removed by MDL Entropy
            return None
        else:
            return state.disc_var

    def discretized_domain(self):
        """
        Return the current effective discretized domain.
        """
        if self.data is None:
            return None

        # a mapping of all applied changes for variables in `varmodel`
        mapping = {var: self.discretized_var(i)
                   for i, var in enumerate(self.varmodel)}

        def disc_var(source):
            return mapping.get(source, source)

        # map the full input domain to the new variables (where applicable)
        attributes = [disc_var(v) for v in self.data.domain.attributes]
        attributes = [v for v in attributes if v is not None]

        class_vars = [disc_var(v) for v in self.data.domain.class_vars]
        class_vars = [v for v in class_vars if v is not None]

        domain = Orange.data.Domain(
            attributes, class_vars, metas=self.data.domain.metas
        )
        return domain

    @gui.deferred
    def commit(self):
        output = None
        if self.data is not None:
            domain = self.discretized_domain()
            output = self.data.transform(domain)
        self.Outputs.data.send(output)

    def storeSpecificSettings(self):
        super().storeSpecificSettings()
        self.saved_var_states = {
            variable_key(var):
                self.var_state[i]._replace(points=None, disc_var=None)
            for i, var in enumerate(self.varmodel)
        }

    def send_report(self):
        self.report_items((
            ("Default method", self.options[self.default_method][1]),))
        if self.varmodel:
            self.report_items("Thresholds", [
                (var.name,
                 DiscDelegate.cutsText(self.var_state[i], var.repr_val) or "leave numeric")
                for i, var in enumerate(self.varmodel)])

    @classmethod
    def migrate_settings(cls, settings, version):  # pylint: disable=redefined-outer-name
        if version is None or version < 2:
            # was stored as int indexing Methods (but offset by 1)
            default = settings.pop("default_method", 0)
            default = Methods(default + 1)
            settings["default_method_name"] = default.name


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWDiscretize).run(Orange.data.Table("brown-selected"))
