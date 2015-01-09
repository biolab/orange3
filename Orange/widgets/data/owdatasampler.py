import sys
import math

from PyQt4 import QtGui
from PyQt4.QtCore import Qt

from sklearn import cross_validation

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data.table import Table
from Orange.data import DiscreteVariable


class OWDataSampler(widget.OWWidget):
    name = "Data Sampler"
    description = "Selects a subset of instances from the data set."
    icon = "icons/DataSampler.svg"
    priority = 100
    category = "Data"
    keywords = ["data", "sample"]
    inputs = [("Data", Table, "setData")]
    outputs = [("Data Sample", Table), ("Remaining Data", Table)]

    want_main_area = False

    #: Ways to specify sample sizes for RandomSampling
    Fixed, Ratio = 0, 1

    stratified = Setting(False)
    useSeed = Setting(False)
    seed = Setting(0)
    replacement = Setting(False)
    samplingType = Setting(0)
    sampleSizeType = Setting(Ratio)
    sampleSizeNumber = Setting(1)
    sampleSizePercentage = Setting(70)
    cvType = Setting(0)
    numberOfFolds = Setting(10)
    selectedFold = Setting(1)

    #: Sampling types
    RandomSampling, CrossValidation = 0, 1

    #: CV Type
    KFold, LeaveOneOut = 0, 1

    def __init__(self):
        super().__init__()

        self.data = None

        self.samplingBox = [None, None]
        self.sampleSizeBox = [None, None]
        self.cvBox = [None, None]

        box = gui.widgetBox(self.controlArea, "Information")
        self.dataInfoLabel = gui.widgetLabel(box, 'No data on input.')
        self.outputInfoLabel = gui.widgetLabel(box, ' ')

        form = QtGui.QGridLayout()

        box = gui.widgetBox(self.controlArea, "Options", orientation=form)
        cb = gui.checkBox(box, self, "stratified", "Stratified (if possible)",
                          callback=self.settingsChanged, addToLayout=False)

        form.addWidget(cb, 0, 0)

        cb = gui.checkBox(box, self, "useSeed", "Random seed:",
                          callback=self.settingsChanged, addToLayout=False)

        spin = gui.spin(box, self, "seed", minv=0, maxv=2 ** 31 - 1,
                        callback=self.settingsChanged, addToLayout=False)
        spin.setEnabled(self.useSeed)
        cb.toggled[bool].connect(spin.setEnabled)

        form.addWidget(cb, 1, 0)
        form.addWidget(spin, 1, 1)

        box = gui.widgetBox(self.controlArea, self.tr("Sampling Type"))

        sampling = gui.radioButtons(box, self, "samplingType",
                                    callback=self.samplingTypeChanged)

        gui.appendRadioButton(sampling, "Random Sampling:")

        self.samplingBox[0] = ibox = gui.indentedBox(sampling)
        ibox.setEnabled(self.samplingType == OWDataSampler.RandomSampling)

        ibox = gui.radioButtons(ibox, self, "sampleSizeType",
                                callback=self.sampleSizeTypeChanged)

        gui.checkBox(ibox, self, "replacement", "With replacement",
                     callback=self.settingsChanged)

        gui.appendRadioButton(ibox, "Sample size:")

        self.sampleSizeSpin = gui.spin(
            gui.indentedBox(ibox), self, "sampleSizeNumber",
            minv=1, maxv=2 ** 31 - 1, callback=self.settingsChanged
        )
        self.sampleSizeSpin.setEnabled(self.sampleSizeType == self.Fixed)

        gui.appendRadioButton(ibox, "Sample proportions:")

        self.sampleSizePercentageSlider = gui.hSlider(
            gui.indentedBox(ibox), self,
            "sampleSizePercentage",
            minValue=1, maxValue=100, ticks=10, labelFormat="%d%%",
            callback=self.settingsChanged
        )

        self.sampleSizePercentageSlider.setEnabled(
            self.sampleSizeType == self.Ratio)

        self.sampleSizeBox = [self.sampleSizeSpin,
                              self.sampleSizePercentageSlider]

        gui.appendRadioButton(sampling, "Cross Validation:")
        self.samplingBox[1] = ibox = gui.indentedBox(sampling, addSpace=True)
        ibox.setEnabled(self.samplingType == OWDataSampler.CrossValidation)

        form = QtGui.QFormLayout(
            formAlignment=Qt.AlignLeft | Qt.AlignTop,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QtGui.QFormLayout.AllNonFixedFieldsGrow
        )
        bbox = gui.radioButtons(ibox, self, "cvType", orientation=form,
                                callback=self.cvTypeChanged)
        bbox.setContentsMargins(1, 1, 1, 1)

        kfold_rb = gui.appendRadioButton(bbox, "K-Fold:", insertInto=None,
                                         addToLayout=False)
        loo_rb = gui.appendRadioButton(bbox, "Leave one out", insertInto=None,
                                       addToLayout=False)
        kfold_spin = gui.spin(ibox, self, "numberOfFolds", 2, 100,
                              addToLayout=False,
                              callback=self.numberOfFoldsChanged)
        kfold_spin.setEnabled(self.cvType == self.KFold)

        form.addRow(kfold_rb, kfold_spin)
        form.addRow(loo_rb)

        self.cvBox = [kfold_spin]

        self.selectedFoldSpin = gui.spin(
             ibox, self, "selectedFold", 1, 100,
             addToLayout=False)

        form.addRow(QtGui.QLabel("Selected fold:"), self.selectedFoldSpin)

        gui.button(self.controlArea, self, "Sample Data",
                   callback=self.commit)

        self.layout().setSizeConstraint(QtGui.QLayout.SetFixedSize)

    def samplingTypeChanged(self):
        for i, sbox in enumerate(self.samplingBox):
            sbox.setEnabled(self.samplingType == i)
        self.settingsChanged()

    def sampleSizeTypeChanged(self):
        for i, box in enumerate(self.sampleSizeBox):
            box.setEnabled(self.sampleSizeType == i)
        self.settingsChanged()

    def cvTypeChanged(self):
        for i, box in enumerate(self.cvBox):
            box.setEnabled(self.cvType == i)

        if self.cvType == self.KFold:
            self.selectedFoldSpin.setMaximum(self.numberOfFolds)
        elif self.data is not None:
            self.selectedFoldSpin.setMaximum(len(self.data))

        self.settingsChanged()

    def numberOfFoldsChanged(self):
        self.updateSelectedFoldSpinMaximum()
        self.settingsChanged()

    def settingsChanged(self):
        """
        The sampling settings have changed, invalidate the indices.
        """
        self.indices = None

    def updateSelectedFoldSpinMaximum(self):
        if self.cvType == self.KFold:
            self.selectedFoldSpin.setMaximum(self.numberOfFolds)
        elif self.data is not None:
            self.selectedFoldSpin.setMaximum(len(self.data))

    def setData(self, dataset):
        self.data = dataset
        self.indices = None
        if dataset is not None:
            self.dataInfoLabel.setText('%d instances in input data set.' %
                                       len(dataset))
            self.sampleSizeSpin.setMaximum(len(dataset))
            self.updateSelectedFoldSpinMaximum()
            self.updateindices()
        else:
            self.dataInfoLabel.setText('No data on input.')
            self.outputInfoLabel.setText('')
            self.indices = None

        self.commit()

    def commit(self):
        sample = None
        other = None
        self.outputInfoLabel.setText("")
        if self.data is not None:
            if self.indices is None:
                self.updateindices()

            assert self.indices is not None
            if self.samplingType == OWDataSampler.RandomSampling:
                train, test = self.indices[0]

                sample = self.data[train]
                other = self.data[test]
                self.outputInfoLabel.setText(
                    'Outputting %d instances.' % len(sample)
                )
            else:
                train, test = self.indices[self.selectedFold - 1]
                sample = self.data[train]
                other = self.data[test]
                self.outputInfoLabel.setText(
                    'Outputting fold %d.' % self.selectedFold
                )

        self.send("Data Sample", sample)
        self.send("Remaining Data", other)

    def updateindices(self):
        if self.useSeed:
            rnd = self.seed
        else:
            rnd = None

        stratified = (self.stratified
                      and not type(self.data) == Table
                      and is_discrete(self.data.domain.class_var))

        if self.samplingType == OWDataSampler.RandomSampling:
            if self.sampleSizeType == OWDataSampler.Fixed:
                indices = sample_random_n(
                    self.data, self.sampleSizeNumber,
                    stratified=stratified,
                    replace=self.replacement,
                    random_state=rnd
                )
            else:
                p = self.sampleSizePercentage / 100.0
                indices = sample_random_p(
                    self.data, p,
                    stratified=stratified,
                    replace=self.replacement,
                    random_state=rnd
                )
            indices = [indices]
        else:
            folds = (self.numberOfFolds if self.cvType == self.KFold
                     else len(self.data))
            indices = sample_fold_indices(
                self.data, folds, stratified=stratified,
                random_state=rnd
            )
        self.indices = indices


def is_discrete(var):
    return isinstance(var, DiscreteVariable)


def sample_fold_indices(table, folds=10, stratified=False, random_state=None):
    """
    :param Orange.data.Table table:
    :param int folds: Number of folds
    :param bool stratified: Return stratified indices (if applicable).
    :param Random random_state:
    :rval tuple-of-arrays: A tuple of array indices one for each fold.
    """
    n = len(table)
    if stratified and is_discrete(table.domain.class_var):
        # XXX: StratifiedKFold does not support random_state
        ind = cross_validation.StratifiedKFold(
            table.Y.ravel(), folds,  # random_state=random_state
        )
    else:
        ind = cross_validation.KFold(
            n, folds, shuffle=True, random_state=random_state
        )

    return tuple(ind)


def sample_loo_indices(table, random_state=None):
    """
    :param Orange.data.Table table:
    """
    return sample_fold_indices(table, len(table), stratified=False,
                               random_state=random_state)


def sample_random_n(table, n, stratified=False, replace=False,
                    random_state=None):
    assert n > 0
    n = int(n)
    if replace:
        if n == 1 and len(table):
            # one example is needed, not the whole (100%) set
            n = n / len(table)
        ind = cross_validation.Bootstrap(
            len(table), train_size=n, random_state=random_state
        )
    elif stratified and is_discrete(table.domain.class_var):
        train_size = max(len(table.domain.class_var.values), n)
        test_size = max(len(table) - train_size, 0)
        ind = cross_validation.StratifiedShuffleSplit(
            table.Y.ravel(), n_iter=1,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state
        )
    else:
        train_size = n
        test_size = max(len(table) - train_size, 0)
        ind = cross_validation.ShuffleSplit(
            len(table), n_iter=1, test_size=test_size,
            train_size=train_size, random_state=random_state
        )
    return next(iter(ind))


def sample_random_p(table, p, stratified=False, replace=False,
                    random_state=None):
    assert 0 <= p <= 1
    n = math.ceil(len(table) * p)
    return sample_random_n(table, n, stratified, replace, random_state)


def test_main():
    app = QtGui.QApplication([])
    data = Table("iris")
    w = OWDataSampler()
    w.setData(data)
    w.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(test_main())
