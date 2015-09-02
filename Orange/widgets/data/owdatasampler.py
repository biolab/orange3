import sys
import math

from PyQt4 import QtGui
from PyQt4.QtCore import Qt

import numpy as np
import sklearn.cross_validation as skl_cross_validation

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data.table import Table


class OWDataSampler(widget.OWWidget):
    name = "Data Sampler"
    description = "Randomly draw a subset of data points " \
                  "from the input data set."
    icon = "icons/DataSampler.svg"
    priority = 100
    category = "Data"
    keywords = ["data", "sample"]
    inputs = [("Data", Table, "set_data")]
    outputs = [("Data Sample", Table, widget.Default), ("Remaining Data", Table)]

    want_main_area = False

    RandomSeed = 42
    FixedProportion, FixedSize, CrossValidation = range(3)

    use_seed = Setting(False)
    replacement = Setting(False)
    stratify = Setting(False)
    sampling_type = Setting(FixedProportion)
    sampleSizeNumber = Setting(1)
    sampleSizePercentage = Setting(70)
    number_of_folds = Setting(10)
    selectedFold = Setting(1)

    def __init__(self):
        super().__init__()
        self.data = None
        self.indices = None

        box = gui.widgetBox(self.controlArea, "Information")
        self.dataInfoLabel = gui.widgetLabel(box, 'No data on input.')
        self.outputInfoLabel = gui.widgetLabel(box, ' ')

        box = gui.widgetBox(self.controlArea, "Sampling Type")
        sampling = gui.radioButtons(
            box, self, "sampling_type", callback=self.sampling_type_changed)

        def set_sampling_type(i):
            def f():
                self.sampling_type = i
                self.sampling_type_changed()
            return f

        gui.appendRadioButton(sampling, "Fixed proportion of data:")
        self.sampleSizePercentageSlider = gui.hSlider(
            gui.indentedBox(sampling), self,
            "sampleSizePercentage",
            minValue=0, maxValue=99, ticks=10, labelFormat="%d %%",
            callback=set_sampling_type(self.FixedProportion))

        gui.appendRadioButton(sampling, "Fixed sample size:")
        ibox = gui.indentedBox(sampling)
        self.sampleSizeSpin = gui.spin(
            ibox, self, "sampleSizeNumber", label="Instances: ",
            minv=1, maxv=2 ** 31 - 1,
            callback=set_sampling_type(self.FixedSize))
        gui.checkBox(
            ibox, self, "replacement", "Sample with replacement",
            callback=set_sampling_type(self.FixedSize))
        gui.separator(sampling, 12)

        gui.separator(sampling, 12)
        gui.appendRadioButton(sampling, "Cross Validation:")
        form = QtGui.QFormLayout(
            formAlignment=Qt.AlignLeft | Qt.AlignTop,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QtGui.QFormLayout.AllNonFixedFieldsGrow)
        ibox = gui.indentedBox(sampling, addSpace=True, orientation=form)
        form.addRow("Number of folds",
                    gui.spin(
                        ibox, self, "number_of_folds", 2, 100,
                        addToLayout=False,
                        callback=self.number_of_folds_changed))
        self.selected_fold_spin = gui.spin(
            ibox, self, "selectedFold", 1, self.number_of_folds,
            addToLayout=False, callback=self.fold_changed)

        form.addRow("Selected fold", self.selected_fold_spin)

        box = gui.widgetBox(self.controlArea, "Options")
        gui.checkBox(box, self, "use_seed",
                     "Replicable (deterministic) sampling",
                     callback=self.settings_changed)
        gui.checkBox(box, self, "stratify",
                     "Stratify sample (when possible)",
                     callback=self.settings_changed)

        gui.button(self.controlArea, self, "Sample Data",
                   callback=self.commit)

        self.layout().setSizeConstraint(QtGui.QLayout.SetFixedSize)

    def sampling_type_changed(self):
        self.settings_changed()

    def number_of_folds_changed(self):
        self.selected_fold_spin.setMaximum(self.number_of_folds)
        self.sampling_type = self.CrossValidation
        self.settings_changed()

    def fold_changed(self):
        # a separate callback - if we decide to cache indices
        self.sampling_type = self.CrossValidation

    def settings_changed(self):
        self.indices = None

    def set_data(self, dataset):
        self.data = dataset
        if dataset is not None:
            self.dataInfoLabel.setText(
                '%d instances in input data set.' % len(dataset))
            self.sampleSizeSpin.setMaximum(len(dataset))
            self.updateindices()
        else:
            self.dataInfoLabel.setText('No data on input.')
            self.outputInfoLabel.setText('')
            self.indices = None
        self.commit()

    def commit(self):
        if self.data is None:
            sample = other = None
            self.outputInfoLabel.setText("")
        else:
            if self.indices is None or not self.use_seed:
                self.updateindices()
            if self.sampling_type in [self.FixedProportion, self.FixedSize]:
                remaining, sample = self.indices
                self.outputInfoLabel.setText(
                    'Outputting %d instance%s.' %
                    (len(sample), "s" * (len(sample) != 1)))
            else:
                remaining, sample = self.indices[self.selectedFold - 1]
                self.outputInfoLabel.setText(
                    'Outputting fold %d, %d instance%s.' %
                    (self.selectedFold, len(sample), "s" * (len(sample) != 1))
                )
            sample = self.data[sample]
            other = self.data[remaining]
        self.send("Data Sample", sample)
        self.send("Remaining Data", other)

    def updateindices(self):
        rnd = self.RandomSeed if self.use_seed else None
        stratified = (self.stratify and
                      type(self.data) == Table and
                      self.data.domain.has_discrete_class)
        if self.sampling_type == self.FixedSize:
            self.indices = sample_random_n(
                self.data, self.sampleSizeNumber,
                stratified=stratified, replace=self.replacement,
                random_state=rnd)
        elif self.sampling_type == self.FixedProportion:
            self.indices = sample_random_p(
                self.data, self.sampleSizePercentage / 100,
                stratified=stratified, random_state=rnd)
        else:
            self.indices = sample_fold_indices(
                self.data, self.number_of_folds, stratified=stratified,
                random_state=rnd)


def sample_fold_indices(table, folds=10, stratified=False, random_state=None):
    """
    :param Orange.data.Table table:
    :param int folds: Number of folds
    :param bool stratified: Return stratified indices (if applicable).
    :param Random random_state:
    :rval tuple-of-arrays: A tuple of array indices one for each fold.
    """
    if stratified and table.domain.has_discrete_class:
        # XXX: StratifiedKFold does not support random_state
        ind = skl_cross_validation.StratifiedKFold(
            table.Y.ravel(), folds, random_state=random_state)
    else:
        ind = skl_cross_validation.KFold(
            len(table), folds, shuffle=True, random_state=random_state)
    return tuple(ind)


def sample_random_n(table, n, stratified=False, replace=False,
                    random_state=None):
    if replace:
        if random_state is None:
            rgen = np.random
        else:
            rgen = np.random.mtrand.RandomState(random_state)
        sample = rgen.random_integers(0, len(table) - 1, n)
        o = np.ones(len(table))
        o[sample] = 0
        others = np.nonzero(o)[0]
        return others, sample
    if stratified and table.domain.has_discrete_class:
        test_size = max(len(table.domain.class_var.values), n)
        ind = skl_cross_validation.StratifiedShuffleSplit(
            table.Y.ravel(), n_iter=1,
            test_size=test_size, train_size=len(table) - test_size,
            random_state=random_state)
    else:
        ind = skl_cross_validation.ShuffleSplit(
            len(table), n_iter=1,
            test_size=n, random_state=random_state)
    return next(iter(ind))


def sample_random_p(table, p, stratified=False, random_state=None):
    n = int(math.ceil(len(table) * p))
    return sample_random_n(table, n, stratified, False, random_state)


def test_main():
    app = QtGui.QApplication([])
    data = Table("iris")
    w = OWDataSampler()
    w.set_data(data)
    w.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(test_main())
