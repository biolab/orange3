import numpy as np

from AnyQt.QtCore import Signal
from AnyQt.QtWidgets import QWidget, QVBoxLayout

from orangewidget.settings import SettingProvider

from Orange.classification import OneClassSVMLearner, EllipticEnvelopeLearner
from Orange.data import Table, Domain, ContinuousVariable
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg, Input, Output, OWWidget


class ParametersEditor(QWidget, gui.OWComponent):
    param_changed = Signal()

    def __init__(self, parent):
        QWidget.__init__(self, parent)
        gui.OWComponent.__init__(self, parent)

        self.setMinimumWidth(300)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.param_box = gui.vBox(self, spacing=0)

    def parameter_changed(self):
        self.param_changed.emit()

    def get_parameters(self):
        raise NotImplementedError


class SVMEditor(ParametersEditor):
    nu = Setting(50)
    gamma = Setting(0.01)

    def __init__(self, parent):
        super().__init__(parent)

        tooltip = "An upper bound on the fraction of training errors and a " \
                  "lower bound of the fraction of support vectors"
        gui.widgetLabel(self.param_box, "Contamination:", tooltip=tooltip)
        gui.hSlider(self.param_box, self, "nu", minValue=1, maxValue=100,
                    ticks=10, labelFormat="%d %%", tooltip=tooltip,
                    callback=self.parameter_changed)
        gui.doubleSpin(self.param_box, self, "gamma",
                       label="Kernel coefficient:", step=1e-2, minv=0.01,
                       maxv=10, callback=self.parameter_changed)

    def get_parameters(self):
        return {"nu": self.nu / 100,
                "gamma": self.gamma}


class CovarianceEditor(ParametersEditor):
    cont = Setting(10)
    empirical_covariance = Setting(False)
    support_fraction = Setting(1)

    def __init__(self, parent):
        super().__init__(parent)

        gui.widgetLabel(self.param_box, "Contamination:")
        gui.hSlider(self.param_box, self, "cont", minValue=0,
                    maxValue=100, ticks=10, labelFormat="%d %%",
                    callback=self.parameter_changed)

        ebox = gui.hBox(self.param_box)
        gui.checkBox(ebox, self, "empirical_covariance",
                     "Support fraction:", callback=self.parameter_changed)
        gui.doubleSpin(ebox, self, "support_fraction", step=1e-1,
                       minv=0.1, maxv=10, callback=self.parameter_changed)

    def get_parameters(self):
        fraction = self.support_fraction if self.empirical_covariance else None
        return {"contamination": self.cont / 100,
                "support_fraction": fraction}


class OWOutliers(OWWidget):
    name = "Outliers"
    description = "Detect outliers."
    icon = "icons/Outliers.svg"
    priority = 3000
    category = "Data"
    keywords = ["inlier"]

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        inliers = Output("Inliers", Table)
        outliers = Output("Outliers", Table)

    want_main_area = False
    resizing_enabled = False

    OneClassSVM, Covariance = range(2)
    METHODS = (OneClassSVMLearner, EllipticEnvelopeLearner)
    svm_editor = SettingProvider(SVMEditor)
    cov_editor = SettingProvider(CovarianceEditor)

    settings_version = 2
    outlier_method = Setting(OneClassSVM)
    auto_commit = Setting(True)

    MAX_FEATURES = 1500

    class Warning(OWWidget.Warning):
        disabled_cov = Msg("Too many features for covariance estimation.")

    class Error(OWWidget.Error):
        singular_cov = Msg("Singular covariance matrix.")
        memory_error = Msg("Not enough memory")

    def __init__(self):
        super().__init__()
        self.data = None  # type: Table
        self.n_inliers = None  # type: int
        self.n_outliers = None  # type: int
        self.editors = None  # type: Tuple[ParametersEditor]
        self.current_editor = None  # type: ParametersEditor
        self.method_combo = None  # type: QComboBox
        self.init_gui()

    def init_gui(self):
        box = gui.vBox(self.controlArea, "Method")
        self.method_combo = gui.comboBox(box, self, "outlier_method",
                                         items=[m.name for m in self.METHODS],
                                         callback=self.__method_changed)

        self._init_editors()

        gui.auto_send(self.controlArea, self, "auto_commit")

        self.info.set_input_summary(self.info.NoInput)
        self.info.set_output_summary(self.info.NoOutput)

    def _init_editors(self):
        self.svm_editor = SVMEditor(self)
        self.cov_editor = CovarianceEditor(self)

        box = gui.vBox(self.controlArea, "Parameters")
        self.editors = (self.svm_editor,
                        self.cov_editor)
        for editor in self.editors:
            editor.param_changed.connect(lambda: self.commit())
            box.layout().addWidget(editor)
            editor.hide()

        self.set_current_editor()

    def __method_changed(self):
        self.set_current_editor()
        self.commit()

    def set_current_editor(self):
        if self.current_editor:
            self.current_editor.hide()
        self.current_editor = self.editors[self.outlier_method]
        self.current_editor.show()

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        self.clear_messages()
        self.data = data
        self.info.set_input_summary(len(data) if data else self.info.NoOutput)
        self.enable_controls()
        self.unconditional_commit()

    def enable_controls(self):
        self.method_combo.model().item(self.Covariance).setEnabled(True)
        if self.data and len(self.data.domain.attributes) > self.MAX_FEATURES:
            self.outlier_method = self.OneClassSVM
            self.set_current_editor()
            self.method_combo.model().item(self.Covariance).setEnabled(False)
            self.Warning.disabled_cov()

    def _get_outliers(self):
        self.Error.singular_cov.clear()
        self.Error.memory_error.clear()
        try:
            y_pred, amended_data = self.detect_outliers()
        except ValueError:
            self.Error.singular_cov()
            return None, None
        except MemoryError:
            self.Error.memory_error()
            return None, None
        else:
            inliers_ind = np.where(y_pred == 1)[0]
            outliers_ind = np.where(y_pred == -1)[0]
            inliers = amended_data[inliers_ind]
            outliers = amended_data[outliers_ind]
            self.n_inliers = len(inliers)
            self.n_outliers = len(outliers)
            return inliers, outliers

    def commit(self):
        inliers = outliers = None
        self.n_inliers = self.n_outliers = None
        if self.data:
            inliers, outliers = self._get_outliers()

        summary = len(inliers) if inliers else self.info.NoOutput
        self.info.set_output_summary(summary)
        self.Outputs.inliers.send(inliers)
        self.Outputs.outliers.send(outliers)

    def detect_outliers(self):
        learner_class = self.METHODS[self.outlier_method]
        kwargs = self.current_editor.get_parameters()
        learner = learner_class(**kwargs)
        model = learner(self.data)
        y_pred = model(self.data)
        amended_data = self.amended_data(model)
        return np.array(y_pred), amended_data

    def amended_data(self, model):
        if self.outlier_method != self.Covariance:
            return self.data
        mahal = model.mahalanobis(self.data.X)
        mahal = mahal.reshape(len(self.data), 1)
        attrs = self.data.domain.attributes
        classes = self.data.domain.class_vars
        new_metas = list(self.data.domain.metas) + \
                    [ContinuousVariable(name="Mahalanobis")]
        new_domain = Domain(attrs, classes, new_metas)
        amended_data = self.data.transform(new_domain)
        amended_data.metas = np.hstack((self.data.metas, mahal))
        return amended_data

    def send_report(self):
        if self.n_outliers is None or self.n_inliers is None:
            return
        self.report_items("Data",
                          (("Input instances", len(self.data)),
                           ("Inliers", self.n_inliers),
                           ("Outliers", self.n_outliers)))
        if self.outlier_method == 0:
            self.report_items(
                "Detection",
                (("Detection method",
                  "One class SVM with non-linear kernel (RBF)"),
                 ("Regularization (nu)", self.svm_editor.nu),
                 ("Kernel coefficient", self.svm_editor.gamma)))
        else:
            self.report_items(
                "Detection",
                (("Detection method", "Covariance estimator"),
                 ("Contamination", self.cov_editor.cont),
                 ("Support fraction", self.cov_editor.support_fraction)))

    @classmethod
    def migrate_settings(cls, settings, version):
        if version is None or version < 2:
            settings["svm_editor"] = {"nu": settings.get("nu", 50),
                                      "gamma": settings.get("gamma", 0.01)}
            ec, sf = "empirical_covariance", "support_fraction"
            settings["cov_editor"] = {"cont": settings.get("cont", 10),
                                      ec: settings.get(ec, False),
                                      sf: settings.get(sf, 1)}


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWOutliers).run(Table("iris"))
