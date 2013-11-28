from PyQt4 import QtGui
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data.table import Table
from Orange.data import DiscreteVariable
from sklearn import cross_validation


# TODO: Util methods that really shouldn't be here and should be moved out of
# this module
# def makeRandomIndices(number, datasetLength, repetition=False,
#                       randomSeed=None):
#     """
#     :param number: number of indices to be selected from the base pool
#     :param datasetLength: size of the base pool
#     :param repetition: repeat indices bool
#     :param randomSeed: random seed for random number generator
#     :return: tuple with list with selected indices and list with the
#          remainder of the base pool
#     """
#     if randomSeed:
#         random.seed(randomSeed)
#     basePool = range(datasetLength)
#     if repetition:
#         indices = [random.choice(basePool) for _ in range(number)]
#     else:
#         indices = random.sample(basePool, number)
#
#     return (indices, list(set(basePool) - set(indices)))

class OWDataSampler(widget.OWWidget):
    name = "Data Sampler"
    description = "Selects a subset of instances from the data set."
    long_description = ""
    icon = "icons/DataSampler.svg"
    author = "Martin Frlin"
    priority = 100
    category = "Data"
    keywords = ["data", "sample"]
    inputs = [("Data", Table, "setData")]
    outputs = [("Data Sample", Table), ("Remaining Data", Table)]

    want_main_area = False

    stratified = Setting(False)
    randomSeed = Setting("")
    samplingType = Setting(0)
    sampleSizeType = Setting(0)
    sampleSizeNumber = Setting(1)
    sampleSizePercentage = Setting(30)
    numberOfFolds = Setting(10)
    selectedFold = Setting(1)

    def __init__(self):
        super().__init__()

        self.data = None
        self.groups = None
        self.CVSettings = (self.stratified, self.numberOfFolds)

        infoBox = gui.widgetBox(self.controlArea, "Information")
        self.dataInfoLabel = gui.widgetLabel(infoBox, 'No data on input.')
        self.outputInfoLabel = gui.widgetLabel(infoBox, ' ')

        optionsBox = gui.widgetBox(self.controlArea, "Options")
        le = gui.lineEdit(optionsBox, self, "randomSeed", "Random seed: ",
                          orientation=0, controlWidth=60,
                          validator=QtGui.QIntValidator())
        s = le.sizeHint().height() - 2
        b = gui.toolButton(le.box, self, width=s, height=s,
                           callback=lambda: le.setText(""))
        b.setIcon(b.style().standardIcon(b.style().SP_DialogCancelButton))
        gui.rubber(le.box)
        gui.checkBox(optionsBox, self, "stratified", "Stratified, if possible")

        sampling = gui.radioButtons(
            self.controlArea, self, "samplingType", box="Sampling type",
            addSpace=True)

        gui.appendRadioButton(sampling, "Random Sampling:")
        indRndSmpl = gui.indentedBox(sampling)
        sizeType = gui.radioButtons(
            indRndSmpl, self, "sampleSizeType",
            callback=lambda: self.chooseSampling(0))
        gui.appendRadioButton(sizeType, "Sample size:", insertInto=indRndSmpl)
        self.sampleSizeSpin = gui.spin(
            gui.indentedBox(indRndSmpl), self, "sampleSizeNumber",
            1, 1000000000, callback=[lambda: self.chooseSampling(0),
                                     lambda: self.chooseSizeType(0)])
        gui.appendRadioButton(sizeType, "Sample proportions:",
                              insertInto=indRndSmpl)
        gui.hSlider(gui.indentedBox(indRndSmpl), self,
                    "sampleSizePercentage",
                    minValue=1, maxValue=100, ticks=10, labelFormat="%d%%",
                    callback=[lambda: self.chooseSampling(0),
                              lambda: self.chooseSizeType(1)])

        gui.appendRadioButton(sampling, "Cross Validation:")
        crossValidIndent = gui.indentedBox(sampling)
        gui.spin(
            crossValidIndent, self, "numberOfFolds", 2, 100,
            label="Number of folds:",
            callback=[self.updateSelectedFoldSpin,
                      lambda: self.chooseSampling(1)])
        self.selectedFoldSpin = gui.spin(
            crossValidIndent, self, "selectedFold", 1, 100,
            label="Selected fold:",
            callback=[self.updateSelectedFoldSpin,
                      lambda: self.chooseSampling(1)])

        gui.button(self.controlArea, self, "Sample Data",
                   callback=self.sendData)

    def updateSelectedFoldSpin(self):
        self.selectedFoldSpin.setMaximum(self.numberOfFolds)

    def chooseSampling(self, n):
        self.samplingType = n

    def chooseSizeType(self, n):
        self.sampleSizeType = n

    def setData(self, dataset):
        self.data = dataset
        self.dataChanged = True
        if dataset:
            self.dataInfoLabel.setText('%d instances in input data set.' %
                                       len(dataset))
            self.sampleSizeSpin.setMaximum(len(dataset))
        else:
            self.dataInfoLabel.setText('No data on input.')
            self.outputInfoLabel.setText('')
        self.sendData()

    def sendData(self):
        if not self.data:
            self.send("Data Sample", None)
            self.send("Remaining Data", None)
            return
        data_size = len(self.data)
        rnd_seed = None
        if self.randomSeed:
            try:
                rnd_seed = int(self.randomSeed)
            except BaseException:
                pass

        stratify = self.stratified and \
            type(self.data) is Table and \
            isinstance(self.data.domain.class_var, DiscreteVariable)

        # random sampling
        if self.samplingType == 0:
            # size by number
            if self.sampleSizeType == 0:
                n_cases = self.sampleSizeNumber
            # size by percentage
            else:
                n_cases = self.sampleSizePercentage * data_size // 100

            if stratify:
                # n_iter=1 means we compute a single split and iterator holds
                # a single item
                ss = cross_validation.StratifiedShuffleSplit(
                    self.data.Y.flatten(), n_iter=1, test_size=n_cases,
                    random_state=rnd_seed)
            else:
                ss = cross_validation.ShuffleSplit(
                    data_size, n_iter=1, test_size=n_cases,
                    random_state=rnd_seed)

            remainder_indices, sample_indices = \
                [(train_ind, test_ind) for train_ind, test_ind in ss][0]
            self.outputInfoLabel.setText('Outputting %d instances.' %
                                         len(sample_indices))
        # cross validation
        else:
            self.outputInfoLabel.setText('Outputting fold %d.' %
                                         self.selectedFold)

            if self.dataChanged or not self.groups or \
                    (self.CVSettings != (stratify, self.numberOfFolds)):
                if stratify:
                    kf = cross_validation.StratifiedKFold(
                        self.data.Y[:, 0], n_folds=self.numberOfFolds)
                else:
                    kf = cross_validation.KFold(
                        data_size, n_folds=self.numberOfFolds)
                self.groups = [(train_ind, test_ind)
                               for train_ind, test_ind in kf]
            remainder_indices, sample_indices = \
                self.groups[self.selectedFold - 1]

            self.dataChanged = False
            # remember cross validation settings so we recompute only on change
            self.CVSettings = (stratify, self.numberOfFolds)

        self.send("Data Sample", self.data[sample_indices])
        self.send("Remaining Data", self.data[remainder_indices])
