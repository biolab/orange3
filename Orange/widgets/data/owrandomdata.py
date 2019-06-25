from collections import namedtuple
from itertools import chain
from math import ceil, log10

import numpy as np
from scipy import stats

from AnyQt.QtCore import Qt, pyqtSignal as Signal, QTimer
from AnyQt.QtWidgets import QComboBox, QFormLayout, \
    QLineEdit, QGroupBox, QStyle, QPushButton, QLabel, QVBoxLayout, QScrollArea
from AnyQt.QtGui import QIntValidator, QDoubleValidator

from Orange.data import Table, ContinuousVariable, Domain, DiscreteVariable
from Orange.widgets.settings import Setting
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Output, Msg
from Orange.widgets import gui


ParameterDef = namedtuple(
    "ParameterDef",
    ("label", "arg_name", "default", "arg_type"))


class pos_int(int):  # pylint: disable=invalid-name
    validator = QIntValidator()


class any_float(float):  # pylint: disable=invalid-name
    validator = QDoubleValidator()


class pos_float(float):  # pylint: disable=invalid-name
    validator = QDoubleValidator()
    validator.setBottom(0.0001)


class prob_float(float):  # pylint: disable=invalid-name
    validator = QDoubleValidator(0, 1, 5)


class ParametersEditor(QGroupBox):
    remove_clicked = Signal()

    default_prefix = "Var"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.name)
        self.setLayout(QVBoxLayout())

        hbox = gui.hBox(self)
        self.add_standard_parameters(hbox)
        gui.separator(hbox, 20)
        gui.rubber(hbox)
        self.add_specific_parameters(hbox)

        self.error = QLabel()
        self.error.setHidden(True)
        self.layout().addWidget(self.error)

        self.trash_button = trash = QPushButton(
            self,
            icon=self.style().standardIcon(QStyle.SP_DockWidgetCloseButton))
        trash.setGeometry(0, 20, 15, 15)
        trash.setFlat(True)
        trash.setHidden(True)
        trash.clicked.connect(self.on_trash_clicked)

    def on_trash_clicked(self):
        self.remove_clicked.emit()

    def enterEvent(self, e):
        super().enterEvent(e)
        self.trash_button.setHidden(False)

    def leaveEvent(self, e):
        super().enterEvent(e)
        self.trash_button.setHidden(True)

    def add_standard_parameters(self, parent):
        form = QFormLayout()
        parent.layout().addLayout(form)

        self.number_of_vars = edit = QLineEdit()
        edit.setValidator(pos_int.validator)
        edit.setText("10")
        edit.setAlignment(Qt.AlignRight)
        edit.setFixedWidth(50)
        form.addRow("Variables", edit)

        self.name_prefix = edit = QLineEdit()
        edit.setPlaceholderText(self.default_prefix)
        edit.setFixedWidth(50)
        form.addRow("Name prefix", edit)

    def fix_standard_parameters(self, number_of_vars, name_prefix):
        if number_of_vars is not None:
            self.number_of_vars.setText("2")
        self.number_of_vars.setDisabled(number_of_vars is not None)
        if name_prefix is not None:
            self.name_prefix.setText("x, y")
        self.name_prefix.setDisabled(name_prefix is not None)

    def add_specific_parameters(self, parent):
        form = QFormLayout()
        self.parameter_edits = {}
        for parameter in self.parameters:
            edit = self.parameter_edits[parameter.arg_name] = QLineEdit()
            edit.convert = getattr(
                parameter.arg_type, "convert", parameter.arg_type)
            validator = getattr(parameter.arg_type, "validator", None)
            if validator is not None:
                edit.setValidator(validator)
            edit.setText(str(parameter.default))
            edit.setAlignment(Qt.AlignRight)
            edit.setFixedWidth(50)
            form.addRow(parameter.label, edit)
        parent.layout().addLayout(form)

    @property
    def nvars(self):
        return int(self.number_of_vars.text())

    def get(self, name):
        edit = self.parameter_edits[name]
        return edit.convert(edit.text())

    def get_parameters(self):
        return {name: self.get(name) for name in self.parameter_edits}

    @staticmethod
    def check(**_):
        return None

    def set_error(self, error):
        self.error.setText(f"<font color='red'>{error}</font>" if error else "")
        self.error.setHidden(error is None)

    def prepare_variables(self, used_names, ndigits):
        raise NotImplementedError

    def get_name_prefix(self, used_names):
        name = self.name_prefix.text() or self.default_prefix
        start = used_names.get(name, 0) + 1
        return name, start

    def generate_data(self, ninstances):
        parameters = self.get_parameters()
        error = self.check(**parameters)  # pylint: disable=assignment-from-none
        data = None
        if not error:
            try:
                data = self.rvs(size=(ninstances, self.nvars), **parameters)
            except:  # can throw anything, pylint: disable=bare-except
                error = f"Error while sampling. Check distribution parameters."
        self.set_error(error)
        return data

    def pack_settings(self):
        return dict(number_of_vars=self.number_of_vars.text(),
                    name_prefix=self.name_prefix.text(),
                    **{name: edit.text()
                       for name, edit in self.parameter_edits.items()})

    def unpack_settings(self, settings):
        edits = dict(number_of_vars=self.number_of_vars,
                     name_prefix=self.name_prefix,
                     **self.parameter_edits)
        for name, value in settings.items():
            edits[name].setText(value)


class ParametersEditorContinuous(ParametersEditor):
    def prepare_variables(self, used_names, ndigits):
        name, start = self.get_name_prefix(used_names)
        used_names[name] = start + self.nvars - 1
        return (ContinuousVariable(f"{name}{i:0{ndigits}}")
                for i in range(start, start + self.nvars))


class ParametersEditorDiscrete(ParametersEditor):
    def prepare_variables(self, used_names, ndigits):
        name, start = self.get_name_prefix(used_names)
        used_names[name] = start + self.nvars - 1
        values = self.get_values(**self.get_parameters())
        return (DiscreteVariable(f"{name}{i:0{ndigits}}", values=values)
                for i in range(start, start + self.nvars))


class Bernoulli(ParametersEditorDiscrete):
    name = "Bernoulli distribution"
    parameters = (ParameterDef("Probability", "p", 0.5, prob_float), )
    rvs = stats.bernoulli.rvs

    @staticmethod
    def get_values(**_):
        return "0", "1"


class ContinuousUniform(ParametersEditorContinuous):
    name = "Uniform distribution"
    parameters = (
        ParameterDef("Low bound", "loc", 0, any_float),
        ParameterDef("High bound", "scale", 1, any_float)
    )
    rvs = stats.uniform.rvs

    @staticmethod
    def check(*, loc, scale):  # pylint: disable=arguments-differ
        if loc >= scale:
            return "Lower bound is must be below the upper."
        return None


class DiscreteUniform(ParametersEditorDiscrete):
    name = "Discrete uniform distribution"
    parameters = (ParameterDef("Number of values", "k", 6, pos_int), )

    @staticmethod
    def rvs(k, size):
        return stats.randint.rvs(0, k, size=size)

    @staticmethod
    def get_values(k):
        return [str(i) for i in range(1, k + 1)]


class Multinomial(ParametersEditorContinuous):
    name = "Multinomial distribution"
    parameters = (
        ParameterDef("Probabilities", "ps", "0.5, 0.3, 0.2", str),
        ParameterDef("Number of trials", "n", 100, pos_int)
    )

    def __init__(self):
        super().__init__()
        self.parameter_edits["ps"].setFixedWidth(120)
        self.parameter_edits["ps"].setAlignment(Qt.AlignLeft)
        self.get("ps")

    @property
    def nvars(self):
        return len(self.get("ps"))

    def get(self, name):
        if name != "ps":
            return super().get(name)
        s = self.parameter_edits["ps"].text()
        try:
            ps = [float(x)
                  for x in s.replace(",", " ").replace(";", " ").split()]
        except ValueError:
            self.set_error("Probabilities must be given as list of numbers.")
            return None
        tot = sum(ps)
        if abs(tot - 1) > 1e-6:
            self.set_error(f"Probabilities must sum to 1, not {tot:.4f}.")
            return None
        self.fix_standard_parameters(len(ps), None)
        return ps

    @staticmethod
    def rvs(ps, n, size):
        if ps is None:
            return None
        return stats.multinomial.rvs(p=ps, n=n, size=size[0])


class HyperGeometric(ParametersEditorContinuous):
    name = "Hypergeometric distribution"
    parameters = (
        ParameterDef("Number of objects", "M", 100, pos_int),
        ParameterDef("Number of positives", "n", 20, pos_int),
        ParameterDef("Number of trials", "N", 20, pos_int)
    )
    rvs = stats.hypergeom.rvs

    @staticmethod
    def check(*, M, n, N):  # pylint: disable=arguments-differ
        if n > M:
            return "Number of positives exceeds number of objects."
        if N > M:
            return "Number of trials exceeds number of objects."
        return None


class BivariateNormal(ParametersEditor):
    name = "Bivariate normal distribution"
    parameters = (
        ParameterDef("Mean x", "mu1", 0, any_float),
        ParameterDef("Variance x", "var1", 1, pos_float),
        ParameterDef("Mean y", "mu2", 0, any_float),
        ParameterDef("Variance y", "var2", 1, pos_float),
        ParameterDef("Covariance", "covar", 0.5, pos_float)
    )
    nvars = 2

    def add_standard_parameters(self, parent):
        super().add_standard_parameters(parent)
        self.fix_standard_parameters("2", "x, y")

    def prepare_variables(self, used_names, ndigits):
        start = 1 + max(used_names.get("x", 0), used_names.get("y", 0))
        used_names["x"] = used_names["y"] = start
        return [ContinuousVariable(f"x{start:0{ndigits}}"),
                ContinuousVariable(f"y{start:0{ndigits}}")]

    @staticmethod
    def rvs(*, mu1, mu2, var1, var2, covar, size):
        return stats.multivariate_normal.rvs(
            mean=np.array([mu1, mu2]),
            cov=np.array([[var1, covar], [covar, var2]]),
            size=size[0])


def cd(name, rvs, *parameters):  # short-lived, pylint: disable=invalid-name
    return type(
        name.title().replace(" ", ""),
        (ParametersEditorContinuous,),
        dict(name=name, rvs=rvs,
             parameters=[ParameterDef(*p) for p in parameters]))


dist_defs = [
    cd("Normal distribution", stats.norm.rvs,
       ("Mean", "loc", 0, any_float),
       ("Variance", "scale", 1, pos_float)),
    Bernoulli,
    cd("Binomial distribution", stats.binom.rvs,
       ("Number of trials", "n", 100, pos_int),
       ("Probability of success", "p", 0.5, prob_float)),
    ContinuousUniform,
    DiscreteUniform,
    Multinomial,
    HyperGeometric,
    cd("Negative binomial distribution", stats.nbinom.rvs,
       ("Number of successes", "n", 10, pos_int),
       ("Probability of success", "p", 0.5, prob_float)),
    cd("Poisson distribution", stats.poisson.rvs,
       ("Event rate (λ)", "mu", 5, pos_float)),
    cd("Exponential distribution", stats.expon.rvs),
    cd("Gamma distribution", stats.gamma.rvs,
       ("Shape (α)", "a", 2, pos_float),
       ("Scale", "scale", 2, pos_float)),
    cd("Student's t distribution", stats.t.rvs,
       ("Degrees of freedom", "df", 1, pos_float)),
    BivariateNormal
]

distributions = {dist.name: dist for dist in dist_defs}
del dist_defs
del cd


class OWRandomData(OWWidget):
    name = "Random Data"
    description = "Generate random data sample"
    icon = "icons/RandomData.svg"
    priority = 2100
    keywords = []

    class Error(OWWidget.Error):
        sampling_error = Msg("Error while sampling.")

    class Outputs:
        data = Output("Data", Table)

    want_main_area = False
    left_side_scrolling = True
    # resizing_enabled = False   # This would disable scrolling

    n_instances = Setting(1000)
    distributions = Setting([
        ('Normal distribution',
         {'number_of_vars': '10', 'name_prefix': '', 'loc': '0', 'scale': '1'}),
        ('Binomial distribution',
         {'number_of_vars': '1', 'name_prefix': '', 'n': '100', 'p': '0.5'})])

    def __init__(self):
        super().__init__()
        self.editors = []

        combo = QComboBox()
        combo.addItem("Add more variables ...")
        combo.addItems(list(distributions))
        combo.currentTextChanged.connect(self.on_add_distribution)
        self.controlArea.layout().addWidget(combo)
        gui.separator(self.controlArea, 16)

        box = gui.vBox(self.controlArea, box=True)
        box2 = gui.hBox(box)
        gui.lineEdit(
            box2, self, "n_instances", "Sample size",
            orientation=Qt.Horizontal, controlWidth=70, alignment=Qt.AlignRight,
            valueType=int, validator=QIntValidator(1, 1_000_000))
        gui.rubber(box2)
        gui.button(
            box2, self, label="Generate", callback=self.generate, width=160)

        self.settingsAboutToBePacked.connect(self.pack_editor_settings)
        self.unpack_editor_settings()
        self.generate()

    def on_add_distribution(self, dist_name):
        combo = self.sender()
        if not combo.currentIndex():
            return
        editor_class = distributions[dist_name]
        self.add_editor(editor_class())
        combo.setCurrentIndex(0)

    def _scroll_area(self):
        # OWWidget does not expose it, but I need it and I know it's there
        scroll_area = self.left_side
        while scroll_area and not isinstance(scroll_area, QScrollArea):
            scroll_area = scroll_area.parent()
        return scroll_area

    def _resize(self):
        scroll_area = self._scroll_area()
        scroll_area.updateGeometry()
        scroll_area.adjustSize()
        QTimer.singleShot(0, self.adjustSize)

    def add_editor(self, editor):
        editor.remove_clicked.connect(self.remove_editor)
        self.controlArea.layout().insertWidget(len(self.editors), editor)
        self.editors.append(editor)
        self._resize()
        scroll_bar = self._scroll_area().verticalScrollBar()
        QTimer.singleShot(0, lambda: scroll_bar.setValue(scroll_bar.maximum()))
        self.generate()

    def remove_editor(self):
        editor = self.sender()
        self.controlArea.layout().removeWidget(editor)
        self.editors.remove(editor)
        editor.deleteLater()
        self._resize()
        self.generate()

    def generate(self):
        used_names = {}
        editors = self.editors
        ndigits = int(ceil(log10(1 + sum(e.nvars for e in editors))))
        attrs = tuple(e.prepare_variables(used_names, ndigits) for e in editors)
        parts = tuple(e.generate_data(self.n_instances) if attr else None
                      for e, attr in zip(editors, attrs))
        if not editors:
            data = None
        elif None in attrs or any(part is None for part in parts):
            data = None
            self.Error.sampling_error()
        else:
            domain = Domain(list(chain(*attrs)))
            data = Table(domain, np.hstack(parts))
            self.Error.sampling_error.clear()
        self.Outputs.data.send(data)

    def pack_editor_settings(self):
        self.distributions = [(editor.name, editor.pack_settings())
                              for editor in self.editors]

    def unpack_editor_settings(self):
        self.editors = []
        for name, editor_args in self.distributions:
            editor = distributions[name]()
            editor.unpack_settings(editor_args)
            self.add_editor(editor)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWRandomData).run()
