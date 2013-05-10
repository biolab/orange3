from PyQt4 import QtCore
from PyQt4 import QtGui
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data.table import Table


class OWDataSampler(widget.OWWidget):
    _name = "Data Sampler"
    _description = """
    Selects a subset of instances from the data set."""
    _long_description = """
    """
    _icon = "icons/DataSampler.svg"
    _author = "Janez Demsar"
    _maintainer_email = "janez.demsar(@at@)fri.uni-lj.si"
    _priority = 10
    _category = "Data"
    _keywords = ["data", "sample"]
    inputs = [("Data", Table, "setData")]
    outputs = [("Data Sample", Table), ("Remaining Data", Table)]

    want_main_area = False

    # define non contextual settings
    stratified = Setting(False)             # Setting for stratified option
    setRandomSeed = Setting(False)          # Setting for random seed option
    randomSeed = Setting(1)                 # Setting for random seed spin
    samplingType = Setting(1)               # Setting for which RB is enabled at start in sampling types radio button group
    withReplication = Setting(False)        # Setting for data replication option in random sampling
    sampleSizeType = Setting(1)             # Setting for which RB is enabled at start in sample size radio button group
    sampleSizeNumber = Setting(1)           # Setting for the number of examples in a sample in random sampling
    sampleSizePercentage = Setting(30)      # Setting for the percentage of examples in a sample in random sampling
    numberOfFolds = Setting(10)             # Setting for the number of folds in cross validation
    selectedFold = Setting(1)               # Setting for the selected fold in cross validation

    def __init__(self, parent=None, signalManager=None, settings=None):
        super().__init__(parent, signalManager, settings)

        self.data = None

        # self.stratified = False
        # self.setRandomSeed = False
        # self.randomSeed = 1
        # self.samplingType = 1
        # self.withReplication = False
        # self.sampleSizeType = 1
        # self.sampleSizeNumber =1
        # self.sampleSizePercentage = 30
        # self.numberOfFolds = 10
        # self.selectedFold = 1


        # Information Box
        infoBox = gui.widgetBox(self.controlArea, "Information:", addSpace=True)
        self.dataInfoLabel = gui.widgetLabel(infoBox, 'No data on input.')
        self.infob = gui.widgetLabel(infoBox, ' ')
        self.infoc = gui.widgetLabel(infoBox, ' ')


        # Options Box
        optionsBox = gui.widgetBox(self.controlArea, "Options:", addSpace=True)
        self.stratifiedCheckBox = gui.checkBox(optionsBox, self, "stratified", "Stratified (if possible)")
        self.setRndSeedCheckBox = gui.checkWithSpin(optionsBox, self, "Set Random Seed", 1, 32767, "setRandomSeed", "randomSeed")

        # Box that will hold Sampling Types radio buttons and will link them into group
        samplingTypesBox = gui.widgetBox(self.controlArea, "Sampling types:", addSpace=True)

        # Random Sampling
        self.randomSamplingRB = gui.appendRadioButton(samplingTypesBox, self, "samplingType", "Random Sampling:")

        # indent under Random Sampling which also acts as a sample size type radio button group
        self.rndSmplIndent = gui.indentedBox(samplingTypesBox)

        self.replicationCheckBox = gui.checkBox(self.rndSmplIndent, self, "withReplication", "with replication")

        self.sampleSizeRB = gui.appendRadioButton(self.rndSmplIndent, self, "sampleSizeType", "Sample size:")

        # indent level 2 under sample size
        self.smplSizeIndent = gui.indentedBox(self.rndSmplIndent)

        self.sampleSizeSpin = gui.spin(self.smplSizeIndent, self, "sampleSizeNumber", 1, 1000000000)
        # back to level 1 indent

        self.sampleProportionsRB = gui.appendRadioButton(self.rndSmplIndent, self, "sampleSizeType", "Sample proportions:")

        # indent level 2 under sample proportions
        self.smplPropIndent = gui.indentedBox(self.rndSmplIndent)

        self.sampleProportionsSlider = gui.hSlider(self.smplPropIndent, self, "sampleSizePercentage", minValue=1, maxValue=100, ticks=10, labelFormat="%d%%")
        # end of indentation

        # Cross Validation
        self.crossValidationRB = gui.appendRadioButton(samplingTypesBox, self, "samplingType", "Cross Validation:")

        # indent under cross validation
        self.crossValidIndent = gui.indentedBox(samplingTypesBox)

        self.numberOfFoldsSpin = gui.spin(self.crossValidIndent, self, "numberOfFolds", 2, 100, label="Number of folds:")

        self.selectedFoldSpin = gui.spin(self.crossValidIndent, self, "selectedFold", 1, 100, label="Selected fold:")
        # end of indentation

        # Sample Data Button
        gui.button(self.controlArea, self, "Sample Data")

        # When everything is loaded we can connect callbacks that affect gui elements
        self.randomSamplingRB.clicked.connect(self.fadeSamplingTypes)
        self.crossValidationRB.clicked.connect(self.fadeSamplingTypes)

        self.sampleSizeRB.clicked.connect(self.fadeSampleSizeTypes)
        self.sampleProportionsRB.clicked.connect(self.fadeSampleSizeTypes)

        # Fade options out
        self.fadeSampleSizeTypes()
        self.fadeSamplingTypes()

    # GUI METHODS
    def fadeSamplingTypes(self):
        if self.randomSamplingRB.isChecked():
            self.rndSmplIndent.setEnabled(True)
            self.crossValidIndent.setEnabled(False)
        else:
            self.rndSmplIndent.setEnabled(False)
            self.crossValidIndent.setEnabled(True)

    def fadeSampleSizeTypes(self):
        if self.sampleSizeRB.isChecked():
            self.smplSizeIndent.setEnabled(True)
            self.smplPropIndent.setEnabled(False)
        else:
            self.smplSizeIndent.setEnabled(False)
            self.smplPropIndent.setEnabled(True)





    # I/O STREAM ROUTINES
    # handles changes of input stream
    def setData(self, dataset):
        if dataset:
            self.dataInfoLabel.setText('%d instances in input data set.' % len(dataset))
            self.data = dataset
        else:
            self.send("Data Sample", None)
            self.send("Remaining Data", None)
            self.data = None