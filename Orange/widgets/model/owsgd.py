from collections import OrderedDict

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QHBoxLayout, QGridLayout, QLabel, QWidget

from Orange.widgets.report import bool_str
from Orange.data import ContinuousVariable, StringVariable, Domain, Table
from Orange.modelling.linear import SGDLearner
from Orange.widgets import gui
from Orange.widgets.model.owlogisticregression import create_coef_table
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.signals import Output
from Orange.widgets.utils.widgetpreview import WidgetPreview

MAXINT = 2 ** 31 - 1


class OWSGD(OWBaseLearner):
    name = 'Stochastic Gradient Descent'
    description = 'Minimize an objective function using a stochastic ' \
                  'approximation of gradient descent.'
    icon = "icons/SGD.svg"
    replaces = [
        "Orange.widgets.regression.owsgdregression.OWSGDRegression",
    ]
    priority = 90
    keywords = ["sgd"]

    settings_version = 2

    LEARNER = SGDLearner

    class Outputs(OWBaseLearner.Outputs):
        coefficients = Output("Coefficients", Table, explicit=True)

    reg_losses = (
        ('Squared Loss', 'squared_error'),
        ('Huber', 'huber'),
        ('ε insensitive', 'epsilon_insensitive'),
        ('Squared ε insensitive', 'squared_epsilon_insensitive'))

    cls_losses = (
        ('Hinge', 'hinge'),
        ('Logistic regression', 'log'),
        ('Modified Huber', 'modified_huber'),
        ('Squared Hinge', 'squared_hinge'),
        ('Perceptron', 'perceptron')) + reg_losses

    #: Regularization methods
    penalties = (
        ('None', 'none'),
        ('Lasso (L1)', 'l1'),
        ('Ridge (L2)', 'l2'),
        ('Elastic Net', 'elasticnet'))

    learning_rates = (
        ('Constant', 'constant'),
        ('Optimal', 'optimal'),
        ('Inverse scaling', 'invscaling'))

    learner_name = Setting('SGD')
    #: Loss function index for classification problems
    cls_loss_function_index = Setting(0)
    #: Epsilon loss function parameter for classification problems
    cls_epsilon = Setting(.1)
    #: Loss function index for regression problems
    reg_loss_function_index = Setting(0)
    #: Epsilon loss function parameter for regression problems
    reg_epsilon = Setting(.1)

    penalty_index = Setting(2)
    #: Regularization strength
    alpha = Setting(1e-5)
    #: Elastic Net mixing parameter
    l1_ratio = Setting(.15)

    shuffle = Setting(True)
    use_random_state = Setting(False)
    random_state = Setting(0)
    learning_rate_index = Setting(0)
    eta0 = Setting(.01)
    power_t = Setting(.25)
    max_iter = Setting(1000)
    tol = Setting(1e-3)
    tol_enabled = Setting(True)

    def add_main_layout(self):
        main_widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        main_widget.setLayout(layout)
        self.controlArea.layout().addWidget(main_widget)

        left = gui.vBox(main_widget).layout()
        right = gui.vBox(main_widget).layout()
        self._add_algorithm_to_layout(left)
        self._add_regularization_to_layout(left)
        self._add_learning_params_to_layout(right)

    def _foc_frame_width(self):
        style = self.style()
        return style.pixelMetric(style.PM_FocusFrameHMargin) + \
               style.pixelMetric(style.PM_ComboBoxFrameWidth)

    def _add_algorithm_to_layout(self, layout):
        # this is part of init, pylint: disable=attribute-defined-outside-init
        grid = QGridLayout()
        box = gui.widgetBox(None, 'Loss functions', orientation=grid)
        layout.addWidget(box)
        # Classfication loss function
        self.cls_loss_function_combo = gui.comboBox(
            None, self, 'cls_loss_function_index', orientation=Qt.Horizontal,
            items=list(zip(*self.cls_losses))[0],
            callback=self._on_cls_loss_change)
        hbox = gui.hBox(None)
        hbox.layout().addSpacing(self._foc_frame_width())
        self.cls_epsilon_spin = gui.spin(
            hbox, self, 'cls_epsilon', 0, 1., 1e-2, spinType=float,
            label='ε: ', controlWidth=80, alignment=Qt.AlignRight,
            callback=self.settings_changed)
        hbox.layout().addStretch()
        grid.addWidget(QLabel("Classification: "), 0, 0)
        grid.addWidget(self.cls_loss_function_combo, 0, 1)
        grid.addWidget(hbox, 1, 1)

        # Regression loss function
        self.reg_loss_function_combo = gui.comboBox(
            None, self, 'reg_loss_function_index', orientation=Qt.Horizontal,
            items=list(zip(*self.reg_losses))[0],
            callback=self._on_reg_loss_change)
        hbox = gui.hBox(None)
        hbox.layout().addSpacing(self._foc_frame_width())
        self.reg_epsilon_spin = gui.spin(
            hbox, self, 'reg_epsilon', 0, 1., 1e-2, spinType=float,
            label='ε: ', controlWidth=80, alignment=Qt.AlignRight,
            callback=self.settings_changed)
        hbox.layout().addStretch()
        grid.addWidget(QLabel("Regression: "), 2, 0)
        grid.addWidget(self.reg_loss_function_combo, 2, 1)
        grid.addWidget(hbox, 3, 1)

        # Enable/disable appropriate controls
        self._on_cls_loss_change()
        self._on_reg_loss_change()

    def _add_regularization_to_layout(self, layout):
        # this is part of init, pylint: disable=attribute-defined-outside-init
        box = gui.widgetBox(None, 'Regularization')
        layout.addWidget(box)
        hlayout = gui.hBox(box)
        self.penalty_combo = gui.comboBox(
            hlayout, self, 'penalty_index',
            items=list(zip(*self.penalties))[0], orientation=Qt.Horizontal,
            callback=self._on_regularization_change)
        self.l1_ratio_box = gui.spin(
            hlayout, self, 'l1_ratio', 0, 1., 1e-2, spinType=float,
            label='Mixing: ', controlWidth=80,
            alignment=Qt.AlignRight, callback=self.settings_changed).box

        hbox = gui.indentedBox(
            box, sep=self._foc_frame_width(), orientation=Qt.Horizontal)
        self.alpha_spin = gui.spin(
            hbox, self, 'alpha', 0, 10., .1e-4, spinType=float, controlWidth=80,
            label='Strength (α): ', alignment=Qt.AlignRight,
            callback=self.settings_changed)
        hbox.layout().addStretch()

        # Enable/disable appropriate controls
        self._on_regularization_change()

    def _add_learning_params_to_layout(self, layout):
        # this is part of init, pylint: disable=attribute-defined-outside-init
        box = gui.widgetBox(None, 'Optimization')
        layout.addWidget(box)
        self.learning_rate_combo = gui.comboBox(
            box, self, 'learning_rate_index', label='Learning rate: ',
            items=list(zip(*self.learning_rates))[0],
            orientation=Qt.Horizontal, callback=self._on_learning_rate_change)
        self.eta0_spin = gui.spin(
            box, self, 'eta0', 1e-4, 1., 1e-4, spinType=float,
            label='Initial learning rate (η<sub>0</sub>): ',
            alignment=Qt.AlignRight, controlWidth=80,
            callback=self.settings_changed)
        self.power_t_spin = gui.spin(
            box, self, 'power_t', 0, 1., 1e-4, spinType=float,
            label='Inverse scaling exponent (t): ',
            alignment=Qt.AlignRight, controlWidth=80,
            callback=self.settings_changed)
        gui.separator(box, height=12)

        self.max_iter_spin = gui.spin(
            box, self, 'max_iter', 1, MAXINT - 1, label='Number of iterations: ',
            controlWidth=80, alignment=Qt.AlignRight,
            callback=self.settings_changed)

        self.tol_spin = gui.spin(
            box, self, 'tol', 0, 10., .1e-3, spinType=float, controlWidth=80,
            label='Tolerance (stopping criterion): ', checked='tol_enabled',
            alignment=Qt.AlignRight, callback=self.settings_changed)
        gui.separator(box, height=12)

        # Wrap shuffle_cbx inside another hbox to align it with the random_seed
        # spin box on OSX
        self.shuffle_cbx = gui.checkBox(
            gui.hBox(box), self, 'shuffle',
            'Shuffle data after each iteration',
            callback=self._on_shuffle_change)
        self.random_seed_spin = gui.spin(
            box, self, 'random_state', 0, MAXINT,
            label='Fixed seed for random shuffling: ', controlWidth=80,
            alignment=Qt.AlignRight, callback=self.settings_changed,
            checked='use_random_state', checkCallback=self.settings_changed)

        # Enable/disable appropriate controls
        self._on_learning_rate_change()
        self._on_shuffle_change()

    def _on_cls_loss_change(self):
        # Epsilon parameter for classification loss
        if self.cls_losses[self.cls_loss_function_index][1] in (
                'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'):
            self.cls_epsilon_spin.setEnabled(True)
        else:
            self.cls_epsilon_spin.setEnabled(False)

        self.settings_changed()

    def _on_reg_loss_change(self):
        # Epsilon parameter for regression loss
        if self.reg_losses[self.reg_loss_function_index][1] in (
                'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'):
            self.reg_epsilon_spin.setEnabled(True)
        else:
            self.reg_epsilon_spin.setEnabled(False)

        self.settings_changed()

    def _on_regularization_change(self):
        # Alpha parameter
        if self.penalties[self.penalty_index][1] in ('l1', 'l2', 'elasticnet'):
            self.alpha_spin.setEnabled(True)
        else:
            self.alpha_spin.setEnabled(False)
        # Elastic Net mixing parameter
        self.l1_ratio_box.setHidden(
            self.penalties[self.penalty_index][1] != 'elasticnet')

        self.settings_changed()

    def _on_learning_rate_change(self):
        # Eta_0 parameter
        if self.learning_rates[self.learning_rate_index][1] in \
                ('constant', 'invscaling'):
            self.eta0_spin.setEnabled(True)
        else:
            self.eta0_spin.setEnabled(False)
        # Power t parameter
        if self.learning_rates[self.learning_rate_index][1] in \
                ('invscaling',):
            self.power_t_spin.setEnabled(True)
        else:
            self.power_t_spin.setEnabled(False)

        self.settings_changed()

    def _on_shuffle_change(self):
        if self.shuffle:
            self.random_seed_spin[0].setEnabled(True)
        else:
            self.use_random_state = False
            self.random_seed_spin[0].setEnabled(False)

        self.settings_changed()

    def create_learner(self):
        params = {}
        if self.use_random_state:
            params['random_state'] = self.random_state

        return self.LEARNER(
            classification_loss=self.cls_losses[self.cls_loss_function_index][1],
            classification_epsilon=self.cls_epsilon,
            regression_loss=self.reg_losses[self.reg_loss_function_index][1],
            regression_epsilon=self.reg_epsilon,
            penalty=self.penalties[self.penalty_index][1],
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            shuffle=self.shuffle,
            learning_rate=self.learning_rates[self.learning_rate_index][1],
            eta0=self.eta0,
            power_t=self.power_t,
            max_iter=self.max_iter,
            tol=self.tol if self.tol_enabled else None,
            preprocessors=self.preprocessors,
            **params)

    def get_learner_parameters(self):
        params = OrderedDict({})
        # Classification loss function
        params['Classification loss function'] = self.cls_losses[
            self.cls_loss_function_index][0]
        if self.cls_losses[self.cls_loss_function_index][1] in (
                'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'):
            params['Epsilon (ε) for classification'] = self.cls_epsilon
        # Regression loss function
        params['Regression loss function'] = self.reg_losses[
            self.reg_loss_function_index][0]
        if self.reg_losses[self.reg_loss_function_index][1] in (
                'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'):
            params['Epsilon (ε) for regression'] = self.reg_epsilon

        params['Regularization'] = self.penalties[self.penalty_index][0]
        # Regularization strength
        if self.penalties[self.penalty_index][1] in ('l1', 'l2', 'elasticnet'):
            params['Regularization strength (α)'] = self.alpha
        # Elastic Net mixing
        if self.penalties[self.penalty_index][1] in ('elasticnet',):
            params['Elastic Net mixing parameter (L1 ratio)'] = self.l1_ratio

        params['Learning rate'] = self.learning_rates[
            self.learning_rate_index][0]
        # Eta
        if self.learning_rates[self.learning_rate_index][1] in \
                ('constant', 'invscaling'):
            params['Initial learning rate (η<sub>0</sub>)'] = self.eta0
        # t
        if self.learning_rates[self.learning_rate_index][1] in \
                ('invscaling',):
            params['Inverse scaling exponent (t)'] = self.power_t

        params['Shuffle data after each iteration'] = bool_str(self.shuffle)
        if self.use_random_state:
            params['Random seed for shuffling'] = self.random_state

        return list(params.items())

    def update_model(self):
        super().update_model()
        coeffs = None
        if self.model is not None:
            if self.model.domain.class_var.is_discrete:
                coeffs = create_coef_table(self.model)
            else:
                attrs = [ContinuousVariable("coef")]
                domain = Domain(attrs, metas=[StringVariable("name")])
                cfs = list(self.model.intercept) + list(self.model.coefficients)
                names = ["intercept"] + \
                        [attr.name for attr in self.model.domain.attributes]
                coeffs = Table.from_list(domain, list(zip(cfs, names)))
                coeffs.name = "coefficients"
        self.Outputs.coefficients.send(coeffs)

    @classmethod
    def migrate_settings(cls, settings, version):
        if version < 2:
            settings["max_iter"] = settings.pop("n_iter", 5)
            settings["tol_enabled"] = False


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWSGD).run(Table("iris"))
