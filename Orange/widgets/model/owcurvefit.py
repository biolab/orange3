import ast
import math
from itertools import chain
from typing import Optional, List, Tuple, Any, Mapping, Callable, Set, Union

import numpy as np

from AnyQt.QtCore import Signal
from AnyQt.QtWidgets import QSizePolicy, QWidget, QGridLayout, QLabel, \
    QLineEdit, QVBoxLayout, QPushButton, QDoubleSpinBox, QCheckBox, QGroupBox

from orangewidget.utils.combobox import ComboBoxSearch
from Orange.data import Table, ContinuousVariable
from Orange.data.util import get_unique_names
from Orange.regression import CurveFitLearner
from Orange.widgets import gui
from Orange.widgets.data.owfeatureconstructor import sanitized_name, \
    validate_exp
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemmodels import DomainModel, PyListModel, \
    PyListModelTooltip
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Output, Msg

FUNCTIONS = {k: v for k, v in np.__dict__.items()
             if k in dir(math) and not k.startswith("_")
             and k not in ["prod", "frexp", "modf"]
             or k in ["abs", "arccos", "arccosh", "arcsin", "arcsinh",
                      "arctan", "arctan2", "arctanh", "power", "round"]}
GLOBALS = {name: getattr(np, name) for name in FUNCTIONS}


class Parameter:
    def __init__(self, name: str, initial: float = 1, use_lower: bool = False,
                 lower=0, use_upper: bool = False, upper: float = 100):
        self.name = name
        self.initial = initial
        self.use_lower = use_lower
        self.lower = lower
        self.use_upper = use_upper
        self.upper = upper

    def to_tuple(self) -> Tuple[str, float, bool, float, bool, float]:
        return (self.name, self.initial, self.use_lower,
                self.lower, self.use_upper, self.upper)

    def __repr__(self) -> str:
        return f"Parameter(name={self.name}, initial={self.initial}, " \
               f"use_lower={self.use_lower}, lower={self.lower}, " \
               f"use_upper={self.use_upper}, upper={self.upper})"


class ParametersWidget(QWidget):
    REMOVE, NAME, INITIAL, LOWER, LOWER_SPIN, UPPER, UPPER_SPIN = range(7)
    sigDataChanged = Signal(list)

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.__data: List[Parameter] = []
        self.__labels: List[QLabel] = None
        self.__layout: QGridLayout = None
        self.__button: QPushButton = None
        self.__controls: List[
            Tuple[QPushButton, QLineEdit, QDoubleSpinBox, QCheckBox,
                  QDoubleSpinBox, QCheckBox, QDoubleSpinBox]] = []
        self._setup_gui()

    @property
    def _remove_buttons(self) -> List[QPushButton]:
        return [controls[self.REMOVE] for controls in self.__controls]

    @property
    def _name_edits(self) -> List[QLineEdit]:
        return [controls[self.NAME] for controls in self.__controls]

    @property
    def _init_spins(self) -> List[QDoubleSpinBox]:
        return [controls[self.INITIAL] for controls in self.__controls]

    @property
    def _lower_checks(self) -> List[QCheckBox]:
        return [controls[self.LOWER] for controls in self.__controls]

    @property
    def _lower_spins(self) -> List[QDoubleSpinBox]:
        return [controls[self.LOWER_SPIN] for controls in self.__controls]

    @property
    def _upper_checks(self) -> List[QCheckBox]:
        return [controls[self.UPPER] for controls in self.__controls]

    @property
    def _upper_spins(self) -> List[QDoubleSpinBox]:
        return [controls[self.UPPER_SPIN] for controls in self.__controls]

    def _setup_gui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        box = gui.vBox(self, box=False)
        self.__layout = QGridLayout()
        box.layout().addLayout(self.__layout)

        self.__labels = [QLabel("Name"), QLabel("Initial value"),
                         QLabel("Lower bound"), QLabel("Upper bound")]
        self.__layout.addWidget(self.__labels[0], 0, self.NAME)
        self.__layout.addWidget(self.__labels[1], 0, self.INITIAL)
        self.__layout.addWidget(self.__labels[2], 0, self.LOWER, 1, 2)
        self.__layout.addWidget(self.__labels[3], 0, self.UPPER, 1, 2)
        self._set_labels_visible(False)

        button_box = gui.hBox(box)
        gui.rubber(button_box)
        self.__button = gui.button(
            button_box, self, "+", callback=self.__on_add_button_clicked,
            width=34, autoDefault=False, enabled=True,
            sizePolicy=(QSizePolicy.Maximum, QSizePolicy.Maximum)
        )

    def __on_add_button_clicked(self):
        self._add_row()
        self.sigDataChanged.emit(self.__data)

    def _add_row(self, parameter: Optional[Parameter] = None):
        row_id = len(self.__controls)
        if parameter is None:
            parameter = Parameter(f"p{row_id + 1}")

        edit = QLineEdit(text=parameter.name)
        edit.setFixedWidth(60)
        edit.textChanged.connect(self.__on_text_changed)

        button = gui.button(
            None, self, "×", callback=self.__on_remove_button_clicked,
            autoDefault=False, width=34,
            sizePolicy=(QSizePolicy.Maximum,
                        QSizePolicy.Maximum)
        )
        kwargs = {"minimum": -2147483647, "maximum": 2147483647}
        init_spin = QDoubleSpinBox(decimals=4, **kwargs)
        lower_spin = QDoubleSpinBox(**kwargs)
        upper_spin = QDoubleSpinBox(**kwargs)
        init_spin.setValue(parameter.initial)
        lower_spin.setValue(parameter.lower)
        upper_spin.setValue(parameter.upper)
        lower_check = QCheckBox(checked=bool(parameter.use_lower))
        upper_check = QCheckBox(checked=bool(parameter.use_upper))

        lower_spin.setEnabled(lower_check.isChecked())
        upper_spin.setEnabled(upper_check.isChecked())

        init_spin.valueChanged.connect(self.__on_init_spin_changed)
        lower_spin.valueChanged.connect(self.__on_lower_spin_changed)
        upper_spin.valueChanged.connect(self.__on_upper_spin_changed)
        lower_check.stateChanged.connect(self.__on_lower_check_changed)
        upper_check.stateChanged.connect(self.__on_upper_check_changed)

        controls = (button, edit, init_spin, lower_check,
                    lower_spin, upper_check, upper_spin)
        n_rows = self.__layout.rowCount()
        for i, control in enumerate(controls):
            self.__layout.addWidget(control, n_rows, i)

        self.__data.append(parameter)
        self.__controls.append(controls)
        self._set_labels_visible(True)

    def __on_text_changed(self):
        line_edit: QLineEdit = self.sender()
        row_id = self._name_edits.index(line_edit)
        self.__data[row_id].name = line_edit.text()
        self.sigDataChanged.emit(self.__data)

    def __on_init_spin_changed(self):
        spin: QDoubleSpinBox = self.sender()
        row_id = self._init_spins.index(spin)
        self.__data[row_id].initial = spin.value()
        self.sigDataChanged.emit(self.__data)

    def __on_lower_check_changed(self):
        check: QCheckBox = self.sender()
        row_id = self._lower_checks.index(check)
        self.__data[row_id].use_lower = check.isChecked()
        self.sigDataChanged.emit(self.__data)
        self._lower_spins[row_id].setEnabled(check.isChecked())

    def __on_lower_spin_changed(self):
        spin: QDoubleSpinBox = self.sender()
        row_id = self._lower_spins.index(spin)
        self.__data[row_id].lower = spin.value()
        self.sigDataChanged.emit(self.__data)

    def __on_upper_check_changed(self):
        check: QCheckBox = self.sender()
        row_id = self._upper_checks.index(check)
        self.__data[row_id].use_upper = check.isChecked()
        self.sigDataChanged.emit(self.__data)
        self._upper_spins[row_id].setEnabled(check.isChecked())

    def __on_upper_spin_changed(self):
        spin: QDoubleSpinBox = self.sender()
        row_id = self._upper_spins.index(spin)
        self.__data[row_id].upper = spin.value()
        self.sigDataChanged.emit(self.__data)

    def __on_remove_button_clicked(self):
        index = self._remove_buttons.index(self.sender())
        self._remove_row(index)
        self.sigDataChanged.emit(self.__data)

    def _remove_row(self, row_index: int):
        assert len(self.__controls) > row_index
        for col_index in range(len(self.__controls[row_index])):
            widget = self.__controls[row_index][col_index]
            if widget is not None:
                self.__layout.removeWidget(widget)
                widget.deleteLater()
        del self.__controls[row_index]
        del self.__data[row_index]
        if len(self.__controls) == 0:
            self._set_labels_visible(False)

    def _set_labels_visible(self, visible: True):
        for label in self.__labels:
            label.setVisible(visible)

    def _remove_rows(self):
        for row in range(len(self.__controls) - 1, -1, -1):
            self._remove_row(row)
        self.__data.clear()
        self.__controls.clear()

    def clear_all(self):
        self._remove_rows()

    def set_data(self, parameters: List[Parameter]):
        self._remove_rows()
        for param in parameters:
            self._add_row(param)

    def set_add_enabled(self, enable: bool):
        self.__button.setEnabled(enable)


class OWCurveFit(OWBaseLearner):
    name = "Curve Fit"
    description = "Fit a function to data."
    icon = "icons/CurveFit.svg"
    priority = 90
    keywords = ["function"]

    class Outputs(OWBaseLearner.Outputs):
        coefficients = Output("Coefficients", Table, explicit=True)

    class Warning(OWBaseLearner.Warning):
        duplicate_parameter = Msg("Duplicated parameter name.")
        unused_parameter = Msg("Unused parameter '{}' in "
                               "'Parameters' declaration.")

    class Error(OWBaseLearner.Error):
        invalid_exp = Msg("Invalid expression.")
        no_parameter = Msg("Missing a fitting parameter.\n"
                           "Use 'Feature Constructor' widget instead.")
        unknown_parameter = Msg("Unknown parameter '{}'.\n"
                                "Declare the parameter in 'Parameters' box")

    LEARNER = CurveFitLearner
    supports_sparse = False

    parameters: Mapping[str, Tuple[Any, ...]] = Setting({}, schema_only=True)
    expression: str = Setting("", schema_only=True)

    FEATURE_PLACEHOLDER = "Select Feature"
    PARAM_PLACEHOLDER = "Select Parameter"
    FUNCTION_PLACEHOLDER = "Select Function"

    _feature: Optional[ContinuousVariable] = None
    _parameter: str = PARAM_PLACEHOLDER
    _function: str = FUNCTION_PLACEHOLDER

    def __init__(self, *args, **kwargs):
        self.__param_widget: ParametersWidget = None
        self.__function_box: QGroupBox = None
        self.__expression_edit: QLineEdit = None
        self.__feature_combo: ComboBoxSearch = None
        self.__parameter_combo: ComboBoxSearch = None
        self.__function_combo: ComboBoxSearch = None

        self.__feature_model = DomainModel(
            order=DomainModel.ATTRIBUTES,
            placeholder=self.FEATURE_PLACEHOLDER,
            separators=False, valid_types=ContinuousVariable
        )
        self.__param_model = PyListModel([self.PARAM_PLACEHOLDER])

        self.__pending_parameters = self.parameters

        super().__init__(*args, **kwargs)

    def add_main_layout(self):
        box = gui.vBox(self.controlArea, "Parameters")
        self.__param_widget = ParametersWidget(self)
        self.__param_widget.set_add_enabled(False)
        self.__param_widget.sigDataChanged.connect(
            self.__on_parameters_changed)
        box.layout().addWidget(self.__param_widget)

        self.__function_box = gui.vBox(self.controlArea, box="Expression")
        self.__function_box.setEnabled(False)
        self.__expression_edit = gui.lineEdit(
            self.__function_box, self, "expression",
            placeholderText="Expression...", callback=self.settings_changed
        )
        hbox = gui.hBox(self.__function_box)
        combo_options = dict(sendSelectedValue=True, searchable=True,
                             contentsLength=13)
        self.__feature_combo = gui.comboBox(
            hbox, self, "_feature", model=self.__feature_model,
            callback=self.__on_feature_added, **combo_options
        )
        self.__parameter_combo = gui.comboBox(
            hbox, self, "_parameter", model=self.__param_model,
            callback=self.__on_parameter_added, **combo_options
        )
        sorted_funcs = sorted(FUNCTIONS)
        function_model = PyListModelTooltip(
            chain([self.FUNCTION_PLACEHOLDER], sorted_funcs),
            [""] + [FUNCTIONS[f].__doc__ for f in sorted_funcs],
            parent=self
        )
        self.__function_combo = gui.comboBox(
            hbox, self, "_function", model=function_model,
            callback=self.__on_function_added, **combo_options
        )

    def __on_parameters_changed(self, parameters: List[Parameter]):
        self.parameters = params = {p.name: p.to_tuple() for p in parameters}
        self.__param_model[:] = chain([self.PARAM_PLACEHOLDER], params)
        self.settings_changed()
        self.Warning.duplicate_parameter.clear()
        if len(self.parameters) != len(parameters):
            self.Warning.duplicate_parameter()

    def __on_feature_added(self):
        index = self.__feature_combo.currentIndex()
        if index > 0:
            self.__insert_into_expression(sanitized_name(self._feature.name))
            self.__feature_combo.setCurrentIndex(0)
            self.settings_changed()

    def __on_parameter_added(self):
        index = self.__parameter_combo.currentIndex()
        if index > 0:
            self.__insert_into_expression(sanitized_name(self._parameter))
            self.__parameter_combo.setCurrentIndex(0)
            self.settings_changed()

    def __on_function_added(self):
        index = self.__function_combo.currentIndex()
        if index > 0:
            if not callable(FUNCTIONS[self._function]):  # e, pi, inf, nan
                self.__insert_into_expression(self._function)
            elif self._function in [
                "arctan2", "copysign", "fmod", "gcd", "hypot",
                "isclose", "ldexp", "power", "remainder"
            ]:
                self.__insert_into_expression(self._function + "(,)", 2)
            else:
                self.__insert_into_expression(self._function + "()", 1)
            self.__function_combo.setCurrentIndex(0)
            self.settings_changed()

    def __insert_into_expression(self, what: str, offset=0):
        pos = self.__expression_edit.cursorPosition()
        text = self.__expression_edit.text()
        self.__expression_edit.setText(text[:pos] + what + text[pos:])
        self.__expression_edit.setCursorPosition(pos + len(what) - offset)
        self.__expression_edit.setFocus()

    def set_data(self, data: Optional[Table]):
        super().set_data(data)
        self.__clear()
        self.__init_models()
        self.__enable_controls()
        self.__set_parameters_box()
        self.unconditional_apply()

    def __clear(self):
        self.expression = ""
        self.__param_widget.clear_all()
        self.__on_parameters_changed([])

    def __init_models(self):
        domain = self.data.domain if self.data else None
        self.__feature_model.set_domain(domain)
        self._feature = self.__feature_model[0]

    def __enable_controls(self):
        self.__param_widget.set_add_enabled(bool(self.data))
        self.__function_box.setEnabled(bool(self.data))

    def __set_parameters_box(self):
        if self.__pending_parameters:
            parameters = [Parameter(*p) for p in
                          self.__pending_parameters.values()]
            self.__param_widget.set_data(parameters)
            self.__on_parameters_changed(parameters)
            self.__pending_parameters = []

    def create_learner(self) -> Optional[CurveFitLearner]:
        self.Error.invalid_exp.clear()
        self.Error.no_parameter.clear()
        self.Error.unknown_parameter.clear()
        self.Warning.unused_parameter.clear()
        if not self.data or not self.expression:
            return None

        try:
            tree = ast.parse(self.expression, mode="eval")
        except SyntaxError:
            self.Error.invalid_exp()
            return None

        if not self.__validate_expression(tree):
            self.Error.invalid_exp()
            return None

        domain = self.data.domain
        vars_names = {sanitized_name(var.name): i for i, var
                      in enumerate(domain.attributes) if var.is_continuous}
        function, params_names = \
            _create_lambda(tree, vars_names, list(FUNCTIONS))

        if not params_names:
            self.Error.no_parameter()
            return None
        unknown = [p for p in params_names if p not in self.parameters]
        if unknown:
            self.Error.unknown_parameter(unknown[0])
            return None
        unused = [p for p in self.parameters if p not in params_names]
        if unused:
            self.Warning.unused_parameter(unused[0])

        p0, lower_bounds, upper_bounds = [], [], []
        for name in params_names:
            param = Parameter(*self.parameters[name])
            p0.append(param.initial)
            lower_bounds.append(param.lower if param.use_lower else -np.inf)
            upper_bounds.append(param.upper if param.use_upper else np.inf)
        bounds = (lower_bounds, upper_bounds)

        return self.LEARNER(function, params_names,
                            p0=p0, bounds=bounds,
                            preprocessors=self.preprocessors)

    def get_learner_parameters(self) -> Tuple[Tuple[str, Any]]:
        return tuple(("Function", self.expression),)

    def update_model(self):
        super().update_model()
        coefficients = None
        if self.model is not None:
            coefficients = self.model.coefficients
        self.Outputs.coefficients.send(coefficients)

    @staticmethod
    def __validate_expression(tree: ast.Expression):
        try:
            valid = validate_exp(tree)
        # pylint: disable=broad-except
        except Exception:
            return False
        return valid


def _create_lambda(exp: ast.Expression, vars_mapper: Mapping[str, int],
                   functions: List[str]) -> Tuple[Callable, List[str]]:
    search = ParametersSearch(vars_mapper, functions)
    search.visit(exp)
    args = search.parameters

    name = sanitized_name(get_unique_names(args, "x"))
    exp = ReplaceVars(name, vars_mapper, functions).visit(exp)

    lambda_ = ast.Lambda(
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=arg) for arg in [name] + args],
            varargs=None,
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=exp.body
    )
    exp = ast.Expression(body=lambda_)
    ast.fix_missing_locations(exp)
    return eval(compile(exp, "<lambda>", mode="eval"), GLOBALS), args


class ParametersSearch(ast.NodeVisitor):
    def __init__(self, vars_mapper: Mapping, functions: List):
        super().__init__()
        self.__vars_mapper = vars_mapper
        self.__functions = functions
        self.__parameters = []

    @property
    def parameters(self) -> Set[str]:
        return self.__parameters

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id not in list(self.__vars_mapper) + self.__functions:
            # don't use Set in order to preserve parameters order
            if node.id not in self.__parameters:
                self.__parameters.append(node.id)
        return node


class ReplaceVars(ast.NodeTransformer):
    """
    Replace feature names with X[:, i], where i is index of feature.
    """
    def __init__(self, name: str, vars_mapper: Mapping, functions: List):
        super().__init__()
        self.__name = name
        self.__vars_mapper = vars_mapper
        self.__functions = functions

    def visit_Name(self, node: ast.Name) -> Union[ast.Name, ast.Subscript]:
        if node.id not in self.__vars_mapper or node.id in self.__functions:
            return node
        else:
            n = self.__vars_mapper[node.id]
            return ast.Subscript(
                value=ast.Name(id=self.__name, ctx=ast.Load()),
                slice=ast.ExtSlice(
                    dims=[ast.Slice(lower=None, upper=None, step=None),
                          ast.Index(value=ast.Num(n=n))]),
                ctx=node.ctx
            )


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWCurveFit).run(Table("housing"))
