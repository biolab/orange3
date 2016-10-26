import sys

import numpy as np
from PyQt4 import QtGui

from Orange.base import SklLearner
from Orange.classification import OneClassSVMLearner, EllipticEnvelopeLearner
from Orange.data import Table, Domain, ContinuousVariable
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.sql import check_sql_input


class OWOutliers(widget.OWWidget):
    name = "Outliers"
    description = "Detect outliers."
    icon = "icons/Outliers.svg"
    priority = 3000
    category = "Data"
    keywords = ["data", "outlier", "inlier"]

    inputs = [("Data", Table, "set_data")]
    outputs = [("Inliers", Table), ("Outliers", Table)]

    want_main_area = False

    OneClassSVM, Covariance = range(2)

    outlier_method = Setting(OneClassSVM)
    nu = Setting(50)
    gamma = Setting(0.01)
    cont = Setting(10)
    empirical_covariance = Setting(False)
    support_fraction = Setting(1)

    data_info_default = 'No data on input.'
    in_out_info_default = ' '

    def __init__(self):
        super().__init__()
        self.data = None
        self.n_inliers = self.n_outliers = None

        box = gui.vBox(self.controlArea, "Information")
        self.data_info_label = gui.widgetLabel(box, self.data_info_default)
        self.in_out_info_label = gui.widgetLabel(box,
                                                 self.in_out_info_default)

        box = gui.vBox(self.controlArea, "Outlier Detection Method")
        detection = gui.radioButtons(box, self, "outlier_method")

        gui.appendRadioButton(detection,
                              "One class SVM with non-linear kernel (RBF)")
        ibox = gui.indentedBox(detection)
        tooltip = "An upper bound on the fraction of training errors and a " \
                  "lower bound of the fraction of support vectors"
        gui.widgetLabel(ibox, 'Nu:', tooltip=tooltip)
        self.nu_slider = gui.hSlider(
            ibox, self, "nu", minValue=1, maxValue=100, ticks=10,
            labelFormat="%d %%", callback=self.nu_changed, tooltip=tooltip)
        self.gamma_spin = gui.spin(
            ibox, self, "gamma", label="Kernel coefficient:", step=1e-2,
            spinType=float, minv=0.01, maxv=10, callback=self.gamma_changed)
        gui.separator(detection, 12)

        self.rb_cov = gui.appendRadioButton(detection, "Covariance estimator")
        ibox = gui.indentedBox(detection)
        self.l_cov = gui.widgetLabel(ibox, 'Contamination:')
        self.cont_slider = gui.hSlider(
            ibox, self, "cont", minValue=0, maxValue=100, ticks=10,
            labelFormat="%d %%", callback=self.cont_changed)

        ebox = gui.hBox(ibox)
        self.cb_emp_cov = gui.checkBox(
            ebox, self, "empirical_covariance",
            "Support fraction:", callback=self.empirical_changed)
        self.support_fraction_spin = gui.spin(
            ebox, self, "support_fraction", step=1e-1, spinType=float,
            minv=0.1, maxv=10, callback=self.support_fraction_changed)

        gui.separator(detection, 12)

        gui.button(self.buttonsArea, self, "Detect Outliers",
                   callback=self.commit)
        self.layout().setSizeConstraint(QtGui.QLayout.SetFixedSize)

    def nu_changed(self):
        self.outlier_method = self.OneClassSVM

    def gamma_changed(self):
        self.outlier_method = self.OneClassSVM

    def cont_changed(self):
        self.outlier_method = self.Covariance

    def support_fraction_changed(self):
        self.outlier_method = self.Covariance

    def empirical_changed(self):
        self.outlier_method = self.Covariance

    def disable_covariance(self):
        self.outlier_method = self.OneClassSVM
        self.rb_cov.setDisabled(True)
        self.l_cov.setDisabled(True)
        self.cont_slider.setDisabled(True)
        self.cb_emp_cov.setDisabled(True)
        self.support_fraction_spin.setDisabled(True)
        self.warning('Too many features for covariance estimation.')

    def enable_covariance(self):
        self.rb_cov.setDisabled(False)
        self.l_cov.setDisabled(False)
        self.cont_slider.setDisabled(False)
        self.cb_emp_cov.setDisabled(False)
        self.support_fraction_spin.setDisabled(False)
        self.warning()

    @check_sql_input
    def set_data(self, dataset):
        self.data = dataset
        if self.data is None:
            self.data_info_label.setText(self.data_info_default)
            self.in_out_info_label.setText(self.in_out_info_default)
        else:
            self.data_info_label.setText('%d instances' % len(self.data))
            self.in_out_info_label.setText(' ')

        self.enable_covariance()
        if self.data and len(self.data.domain.attributes) > 1500:
            self.disable_covariance()

        self.commit()

    def commit(self):
        self.error()
        inliers = outliers = None
        self.n_inliers = self.n_outliers = None
        if self.data is not None and len(self.data) > 0:
            try:
                y_pred = self.detect_outliers()
            except ValueError:
                self.error("Singular covariance matrix.")
                self.in_out_info_label.setText(self.in_out_info_default)
            else:
                inliers_ind = np.where(y_pred == 1)[0]
                outliers_ind = np.where(y_pred == -1)[0]
                inliers = Table(self.new_domain, self.new_data, inliers_ind)
                outliers = Table(self.new_domain,
                                 self.new_data, outliers_ind)
                self.in_out_info_label.setText('%d inliers, %d outliers' %
                                               (len(inliers), len(outliers)))
                self.n_inliers = len(inliers)
                self.n_outliers = len(outliers)

        self.send("Inliers", inliers)
        self.send("Outliers", outliers)

    def detect_outliers(self):
        if self.outlier_method == self.OneClassSVM:
            learner = OneClassSVMLearner(
                gamma=self.gamma, nu=self.nu / 100,
                preprocessors=SklLearner.preprocessors)
        else:
            learner = EllipticEnvelopeLearner(
                support_fraction=self.support_fraction
                if self.empirical_covariance else None,
                contamination=self.cont / 100.)
        model = learner(self.data)
        y_pred = model(self.data)
        self.add_metas(model)
        return np.array(y_pred)

    def add_metas(self, model):
        if self.outlier_method == self.Covariance:
            mahal = model.mahalanobis(self.data.X)
            mahal = mahal.reshape(len(self.data), 1)
            attrs = self.data.domain.attributes
            classes = self.data.domain.class_vars
            new_metas = list(self.data.domain.metas) + \
                        [ContinuousVariable(name="Mahalanobis")]
            self.new_domain = Domain(attrs, classes, new_metas)
            self.new_data = Table(self.new_domain, self.data)
            self.new_data.metas = np.hstack((self.data.metas, mahal))
        else:
            self.new_domain = self.data.domain
            self.new_data = self.data

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
                 ("Regularization (nu)", self.nu),
                 ("Kernel coefficient", self.gamma)))
        else:
            self.report_items(
                "Detection",
                (("Detection method", "Covariance estimator"),
                 ("Contamination", self.cont),
                 ("Support fraction", self.support_fraction)))

def test_main():
    app = QtGui.QApplication([])
    data = Table("iris")
    w = OWOutliers()
    w.set_data(data)
    w.commit()
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(test_main())
