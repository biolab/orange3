import ast
from itertools import chain
from typing import Optional, List, Tuple, Any, Mapping

import numpy as np

from AnyQt.QtCore import Signal
from AnyQt.QtWidgets import QSizePolicy, QWidget, QGridLayout, QLabel, \
    QLineEdit, QVBoxLayout, QPushButton, QDoubleSpinBox, QCheckBox

from orangewidget.utils.combobox import ComboBoxSearch
from Orange.data import Table, ContinuousVariable
from Orange.data.util import sanitized_name
from Orange.preprocess import Preprocess
from Orange.regression import CurveFitLearner
from Orange.widgets import gui
from Orange.widgets.data.owfeatureconstructor import validate_exp
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemmodels import DomainModel, PyListModel, \
    PyListModelTooltip
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Output, Msg

FUNCTIONS = {k: v for k, v in np.__dict__.items() if k in
             ("isclose", "inf", "nan", "arccos", "arccosh", "arcsin",
              "arcsinh", "arctan", "arctan2", "arctanh", "ceil", "copysign",
              "cos", "cosh", "degrees", "e", "exp", "expm1", "fabs", "floor",
              "fmod", "gcd", "hypot", "isfinite", "isinf", "isnan", "ldexp",
              "log", "log10", "log1p", "log2", "pi", "power", "radians",
              "remainder", "sin", "sinh", "sqrt", "tan", "tanh", "trunc",
              "round", "abs", "any", "all")}


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
        gui.button(
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

    def _set_labels_visible(self, visible: bool):
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
        data_missing = Msg("Provide data on the input.")

    class Error(OWBaseLearner.Error):
        invalid_exp = Msg("Invalid expression.")
        no_parameter = Msg("Missing a fitting parameter.\n"
                           "Use 'Feature Constructor' widget instead.")
        unknown_parameter = Msg("Unknown parameter '{}'.\n"
                                "Declare the parameter in 'Parameters' box")
        parameter_in_attrs = Msg("Some parameters and features have the same "
                                 "name '{}'.")

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
        self.__pp_data: Optional[Table] = None
        self.__param_widget: ParametersWidget = None
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
        self.__pending_expression = self.expression

        super().__init__(*args, **kwargs)

        self.Warning.data_missing()

    def add_main_layout(self):
        box = gui.vBox(self.controlArea, "Parameters")
        self.__param_widget = ParametersWidget(self)
        self.__param_widget.sigDataChanged.connect(
            self.__on_parameters_changed)
        box.layout().addWidget(self.__param_widget)

        function_box = gui.vBox(self.controlArea, box="Expression")
        self.__expression_edit = gui.lineEdit(
            function_box, self, "expression",
            placeholderText="Expression...", callback=self.settings_changed
        )
        hbox = gui.hBox(function_box)
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
        self.Error.parameter_in_attrs.clear()
        self.Warning.duplicate_parameter.clear()
        if len(self.parameters) != len(parameters):
            self.Warning.duplicate_parameter()
        names = [f.name for f in self.__feature_model[1:]]
        forbidden = [p.name for p in parameters if p.name in names]
        if forbidden:
            self.Error.parameter_in_attrs(forbidden[0])

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

    @OWBaseLearner.Inputs.data
    def set_data(self, data: Optional[Table]):
        self.Warning.data_missing(shown=not bool(data))
        self.learner = None
        super().set_data(data)

    def set_preprocessor(self, preprocessor: Preprocess):
        self.preprocessors = preprocessor
        feature_names_changed = False
        if self.data and self.__pp_data:
            pp_data = preprocess(self.data, preprocessor)
            feature_names_changed = \
                set(a.name for a in pp_data.domain.attributes) != \
                set(a.name for a in self.__pp_data.domain.attributes)
        if feature_names_changed:
            self.expression = ""

    def handleNewSignals(self):
        self.__preprocess_data()
        self.__init_models()
        self.__set_pending()
        super().handleNewSignals()

    def __preprocess_data(self):
        self.__pp_data = preprocess(self.data, self.preprocessors)

    def __init_models(self):
        domain = self.__pp_data.domain if self.__pp_data else None
        self.__feature_model.set_domain(domain)
        self._feature = self.__feature_model[0]

    def __set_pending(self):
        if self.__pending_parameters:
            parameters = [Parameter(*p) for p in
                          self.__pending_parameters.values()]
            self.__param_widget.set_data(parameters)
            self.__on_parameters_changed(parameters)
            self.__pending_parameters = []

        if self.__pending_expression:
            self.expression = self.__pending_expression
            self.__pending_expression = ""

    def create_learner(self) -> Optional[CurveFitLearner]:
        self.Error.invalid_exp.clear()
        self.Error.no_parameter.clear()
        self.Error.unknown_parameter.clear()
        self.Warning.unused_parameter.clear()
        expression = self.expression.strip()
        if not self.__pp_data or not expression:
            return None

        if not self.__validate_expression(expression):
            self.Error.invalid_exp()
            return None

        p0, bounds = {}, {}
        for name in self.parameters:
            param = Parameter(*self.parameters[name])
            p0[name] = param.initial
            bounds[name] = (param.lower if param.use_lower else -np.inf,
                            param.upper if param.use_upper else np.inf)

        learner = self.LEARNER(
            expression,
            available_feature_names=[a.name for a in self.__feature_model[1:]],
            functions=FUNCTIONS,
            sanitizer=sanitized_name,
            p0=p0,
            bounds=bounds,
            preprocessors=self.preprocessors
        )

        params_names = learner.parameters_names
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

        return learner

    def get_learner_parameters(self) -> Tuple[Tuple[str, Any]]:
        return (("Expression", self.expression),)

    def update_model(self):
        super().update_model()
        coefficients = None
        if self.model is not None:
            coefficients = self.model.coefficients
        self.Outputs.coefficients.send(coefficients)

    def check_data(self):
        learner_existed = self.learner is not None
        if self.data:
            data = preprocess(self.data, self.preprocessors)
            dom = data.domain
            cont_attrs = [a for a in dom.attributes if a.is_continuous]
            if len(cont_attrs) == 0:
                self.Error.data_error("Data has no continuous features.")
            elif not self.learner:
                # create dummy learner in order to check data
                self.learner = self.LEARNER(lambda: 1, [], [])
        # parent's check_data() needs learner instantiated
        self.valid_data = super().check_data()
        if not learner_existed:
            self.valid_data = False
            self.learner = None
        return self.valid_data

    @staticmethod
    def __validate_expression(expression: str):
        try:
            tree = ast.parse(expression, mode="eval")
            valid = validate_exp(tree)
        # pylint: disable=broad-except
        except Exception:
            return False
        return valid


def preprocess(data: Optional[Table], preprocessor: Optional[Preprocess]) \
        -> Optional[Table]:
    if not data or not preprocessor:
        return data
    return preprocessor(data)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWCurveFit).run(Table("housing"))
