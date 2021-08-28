from typing import Dict, Tuple
from types import SimpleNamespace

import numpy as np

from AnyQt.QtCore import Signal, Qt
from AnyQt.QtWidgets import QWidget, QVBoxLayout

from orangewidget.settings import SettingProvider

from Orange.base import Learner
from Orange.classification import OneClassSVMLearner, EllipticEnvelopeLearner,\
    LocalOutlierFactorLearner, IsolationForestLearner
from Orange.data import Table
from Orange.util import wrap_callback
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg, Input, Output, OWWidget


class Results(SimpleNamespace):
    inliers = None  # type: Optional[Table]
    outliers = None  # type: Optional[Table]
    annotated_data = None  # type: Optional[Table]


def run(data: Table, learner: Learner, state: TaskState) -> Results:
    results = Results()
    if not data:
        return results

    def callback(i: float, status=""):
        state.set_progress_value(i * 100)
        if status:
            state.set_status(status)
        if state.is_interruption_requested():
            raise Exception

    callback(0, "Initializing...")
    model = learner(data, wrap_callback(callback, end=0.6))
    pred = model(data, wrap_callback(callback, start=0.6, end=0.99))

    col = pred.get_column_view(model.outlier_var)[0]
    inliers_ind = np.where(col == 1)[0]
    outliers_ind = np.where(col == 0)[0]

    results.inliers = data[inliers_ind]
    results.outliers = data[outliers_ind]
    results.annotated_data = pred
    callback(1)
    return results


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

    def get_parameters(self) -> Dict:
        raise NotImplementedError


class SVMEditor(ParametersEditor):
    nu = Setting(50)
    gamma = Setting(0.01)

    def __init__(self, parent):
        super().__init__(parent)

        tooltip = "An upper bound on the fraction of training errors and a " \
                  "lower bound of the fraction of support vectors"
        gui.widgetLabel(self.param_box, "Nu:", tooltip=tooltip)
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


class LocalOutlierFactorEditor(ParametersEditor):
    METRICS = ("euclidean", "manhattan", "cosine", "jaccard",
               "hamming", "minkowski")

    n_neighbors = Setting(20)
    cont = Setting(10)
    metric_index = Setting(0)

    def __init__(self, parent):
        super().__init__(parent)

        gui.widgetLabel(self.param_box, "Contamination:")
        gui.hSlider(self.param_box, self, "cont", minValue=1,
                    maxValue=50, ticks=5, labelFormat="%d %%",
                    callback=self.parameter_changed)
        gui.spin(self.param_box, self, "n_neighbors", label="Neighbors:",
                 minv=1, maxv=100000, callback=self.parameter_changed)
        gui.comboBox(self.param_box, self, "metric_index", label="Metric:",
                     orientation=Qt.Horizontal,
                     items=[m.capitalize() for m in self.METRICS],
                     callback=self.parameter_changed)

    def get_parameters(self):
        return {"n_neighbors": self.n_neighbors,
                "contamination": self.cont / 100,
                "algorithm": "brute",  # works faster for big datasets
                "metric": self.METRICS[self.metric_index]}


class IsolationForestEditor(ParametersEditor):
    cont = Setting(10)
    replicable = Setting(False)

    def __init__(self, parent):
        super().__init__(parent)

        gui.widgetLabel(self.param_box, "Contamination:")
        gui.hSlider(self.param_box, self, "cont", minValue=0,
                    maxValue=100, ticks=10, labelFormat="%d %%",
                    callback=self.parameter_changed)
        gui.checkBox(self.param_box, self, "replicable",
                     "Replicable training", callback=self.parameter_changed)

    def get_parameters(self):
        return {"contamination": self.cont / 100,
                "random_state": 42 if self.replicable else None}


class OWOutliers(OWWidget, ConcurrentWidgetMixin):
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
        data = Output("Data", Table)

    want_main_area = False
    resizing_enabled = False

    OneClassSVM, Covariance, LOF, IsolationForest = range(4)
    METHODS = (OneClassSVMLearner, EllipticEnvelopeLearner,
               LocalOutlierFactorLearner, IsolationForestLearner)
    svm_editor = SettingProvider(SVMEditor)
    cov_editor = SettingProvider(CovarianceEditor)
    lof_editor = SettingProvider(LocalOutlierFactorEditor)
    isf_editor = SettingProvider(IsolationForestEditor)

    settings_version = 2
    outlier_method = Setting(LOF)
    auto_commit = Setting(True)

    MAX_FEATURES = 1500

    class Warning(OWWidget.Warning):
        disabled_cov = Msg("Too many features for covariance estimation.")

    class Error(OWWidget.Error):
        singular_cov = Msg("Singular covariance matrix.")
        memory_error = Msg("Not enough memory")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
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

        gui.auto_apply(self.buttonsArea, self, "auto_commit")

    def _init_editors(self):
        self.svm_editor = SVMEditor(self)
        self.cov_editor = CovarianceEditor(self)
        self.lof_editor = LocalOutlierFactorEditor(self)
        self.isf_editor = IsolationForestEditor(self)

        box = gui.vBox(self.controlArea, "Parameters")
        self.editors = (self.svm_editor, self.cov_editor,
                        self.lof_editor, self.isf_editor)
        for editor in self.editors:
            editor.param_changed.connect(self.commit.deferred)
            box.layout().addWidget(editor)
            editor.hide()

        self.set_current_editor()

    def __method_changed(self):
        self.set_current_editor()
        self.commit.deferred()

    def set_current_editor(self):
        if self.current_editor:
            self.current_editor.hide()
        self.current_editor = self.editors[self.outlier_method]
        self.current_editor.show()

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        self.cancel()
        self.clear_messages()
        self.data = data
        self.enable_controls()
        self.commit.now()

    def enable_controls(self):
        self.method_combo.model().item(self.Covariance).setEnabled(True)
        if self.data and len(self.data.domain.attributes) > self.MAX_FEATURES:
            self.outlier_method = self.LOF
            self.set_current_editor()
            self.method_combo.model().item(self.Covariance).setEnabled(False)
            self.Warning.disabled_cov()

    @gui.deferred
    def commit(self):
        self.Error.singular_cov.clear()
        self.Error.memory_error.clear()
        self.n_inliers = self.n_outliers = None

        learner_class = self.METHODS[self.outlier_method]
        kwargs = self.current_editor.get_parameters()
        learner = learner_class(**kwargs)

        self.start(run, self.data, learner)

    def on_partial_result(self, _):
        pass

    def on_done(self, result: Results):
        inliers, outliers = result.inliers, result.outliers
        self.n_inliers = len(inliers) if inliers else None
        self.n_outliers = len(outliers) if outliers else None

        self.Outputs.inliers.send(inliers)
        self.Outputs.outliers.send(outliers)
        self.Outputs.data.send(result.annotated_data)

    def on_exception(self, ex):
        if isinstance(ex, ValueError):
            self.Error.singular_cov(ex)
        elif isinstance(ex, MemoryError):
            self.Error.memory_error()
        else:
            raise ex

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    def send_report(self):
        if self.n_outliers is None or self.n_inliers is None:
            return
        self.report_items("Data",
                          (("Input instances", len(self.data)),
                           ("Inliers", self.n_inliers),
                           ("Outliers", self.n_outliers)))

        params = self.current_editor.get_parameters()
        if self.outlier_method == self.OneClassSVM:
            self.report_items(
                "Detection",
                (("Detection method",
                  "One class SVM with non-linear kernel (RBF)"),
                 ("Regularization (nu)", params["nu"]),
                 ("Kernel coefficient", params["gamma"])))
        elif self.outlier_method == self.Covariance:
            self.report_items(
                "Detection",
                (("Detection method", "Covariance estimator"),
                 ("Contamination", params["contamination"]),
                 ("Support fraction", params["support_fraction"])))
        elif self.outlier_method == self.LOF:
            self.report_items(
                "Detection",
                (("Detection method", "Local Outlier Factor"),
                 ("Contamination", params["contamination"]),
                 ("Number of neighbors", params["n_neighbors"]),
                 ("Metric", params["metric"])))
        elif self.outlier_method == self.IsolationForest:
            self.report_items(
                "Detection",
                (("Detection method", "Isolation Forest"),
                 ("Contamination", params["contamination"])))
        else:
            raise NotImplementedError

    @classmethod
    def migrate_settings(cls, settings: Dict, version: int):
        if version is None or version < 2:
            settings["svm_editor"] = {"nu": settings.get("nu", 50),
                                      "gamma": settings.get("gamma", 0.01)}
            ec, sf = "empirical_covariance", "support_fraction"
            settings["cov_editor"] = {"cont": settings.get("cont", 10),
                                      ec: settings.get(ec, False),
                                      sf: settings.get(sf, 1)}


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWOutliers).run(Table("iris"))
