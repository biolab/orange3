import sys

from PyQt4 import QtGui

import numpy as np

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data import Table, Domain, ContinuousVariable
from Orange.classification import OneClassSVMLearner, EllipticEnvelopeLearner
from Orange.base import SklLearner


class OWOutliers(widget.OWWidget):
    name = "Outliers"
    description = "Detects outliers."
    icon = "icons/Outliers.svg"
    priority = 3000
    category = "Data"
    keywords = ["data", "outlier", "inlier"]

    inputs = [("Data", Table, "set_data")]
    outputs = [("Inliers", Table), ("Outliers", Table)]

    want_main_area = False

    OneClassSVM, RobustCovariance, EmpiricalCovariance = range(3)

    outlier_method = Setting(OneClassSVM)
    nu = Setting(10)
    gamma = Setting(0)
    cont_robust = Setting(10)
    cont_emp = Setting(10)
    support_fraction_emp = Setting(1)

    data_info_default = 'No data on input.'
    in_out_info_default = ' '

    def __init__(self):
        super().__init__()
        self.data = None

        box = gui.widgetBox(self.controlArea, "Information")
        self.data_info_label = gui.widgetLabel(box, self.data_info_default)
        self.in_out_info_label = gui.widgetLabel(box,
                                                 self.in_out_info_default)

        box = gui.widgetBox(self.controlArea, "Outlier detection method")
        detection = gui.radioButtons(box, self, "outlier_method")

        gui.appendRadioButton(detection,
                              "One class SVM with non-linear kernel (RBF):")
        ibox = gui.indentedBox(detection)
        gui.widgetLabel(ibox, 'Fraction:')
        self.nu_slider = gui.hSlider(
            ibox, self, "nu", minValue=1, maxValue=100, ticks=10,
            labelFormat="%d %%", callback=self.nu_changed)
        self.gamma_spin = gui.spin(
            ibox, self, "gamma", label="Kernel coefficient:", step=1e-2,
            spinType=float, minv=0, maxv=10, callback=self.gamma_changed)
        gui.separator(detection, 12)

        gui.appendRadioButton(detection, "Robust covariance estimator:")
        ibox = gui.indentedBox(detection)
        gui.widgetLabel(ibox, 'Contamination:')
        self.cont_robust_slider = gui.hSlider(
            ibox, self, "cont_robust", minValue=0, maxValue=100, ticks=10,
            labelFormat="%d %%", callback=self.cont_robust_changed)
        gui.separator(detection, 12)

        gui.appendRadioButton(detection, "Empirical covariance estimator:")
        ibox = gui.indentedBox(detection)
        gui.widgetLabel(ibox, 'Contamination:')
        self.cont_emp_slider = gui.hSlider(
            ibox, self, "cont_emp", minValue=0, maxValue=100, ticks=10,
            labelFormat="%d %%", callback=self.cont_emp_changed)
        self.support_fraction_emp_spin = gui.spin(
            ibox, self, "support_fraction_emp", label="Support fraction: ",
            step=1e-1, spinType=float, minv=0.1, maxv=10,
            callback=self.support_fraction_emp_changed)

        gui.separator(detection, 12)

        gui.button(self.controlArea, self, "Detect Outliers",
                   callback=self.commit)
        self.layout().setSizeConstraint(QtGui.QLayout.SetFixedSize)

    def nu_changed(self):
        self.outlier_method = self.OneClassSVM

    def gamma_changed(self):
        self.outlier_method = self.OneClassSVM

    def cont_robust_changed(self):
        self.outlier_method = self.RobustCovariance

    def cont_emp_changed(self):
        self.outlier_method = self.EmpiricalCovariance

    def support_fraction_emp_changed(self):
        self.outlier_method = self.EmpiricalCovariance

    def set_data(self, dataset):
        self.data = dataset
        if self.data is None:
            self.data_info_label.setText(self.data_info_default)
            self.in_out_info_label.setText(self.in_out_info_default)
        else:
            self.data_info_label.setText('%d instances' % len(self.data))
            self.in_out_info_label.setText(' ')

    def commit(self):
        self.error()
        inliers = outliers = None
        if self.data is not None and len(self.data) > 0:
            try:
                y_pred = self.detect_outliers()
            except ValueError:
                self.error(0, "Singular covariance matrix.")
                self.in_out_info_label.setText(self.in_out_info_default)
            else:
                inliers_ind = np.where(y_pred == 1)[0]
                outliers_ind = np.where(y_pred == -1)[0]
                inliers = Table(self.new_domain, self.new_data, inliers_ind)
                outliers = Table(self.new_domain,
                                 self.new_data, outliers_ind)
                self.in_out_info_label.setText('%d inliers, %d outliers' %
                                               (len(inliers), len(outliers)))

        self.send("Inliers", inliers)
        self.send("Outliers", outliers)

    def detect_outliers(self):
        if self.outlier_method == self.OneClassSVM:
            learner = OneClassSVMLearner(
                gamma=self.gamma, nu=self.nu / 100,
                preprocessors=SklLearner.preprocessors)
        elif self.outlier_method == self.RobustCovariance:
            learner = EllipticEnvelopeLearner(
                support_fraction=None,
                contamination=self.cont_robust / 100.)
        else:
            learner = EllipticEnvelopeLearner(
                support_fraction=self.support_fraction_emp,
                contamination=self.cont_emp / 100.)
        model = learner(self.data)
        y_pred = model(self.data)
        self.add_metas(model)
        return np.array(y_pred)

    def add_metas(self, model):
        if self.outlier_method in (self.RobustCovariance,
                                   self.EmpiricalCovariance):
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
