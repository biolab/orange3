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
    inputs = [("Data", Table, "set_data")]
    outputs = [("Data Sample", Table), ("Remaining Data", Table)]

    want_main_area = False

    # define non contextual settings
    stratified = Setting(False)             # Setting for stratified option
    setRandomSeed = Setting(False)          # Setting for random seed option
    randomSeed = Setting(1)                 # Setting for random seed spin
    samplingType = Setting(1)               # Setting for sampling types radio button group
    withReplication = Setting(False)        # Setting for data replication option in random sampling
    sampleSizeType = Setting(1)             # Setting for sample size radio button group
    sampleSizeNumber = Setting(1)           # Setting for the number of examples in a sample in random sampling
    sampleSizePercentage = Setting(30)      # Setting for the percentage of examples in a sample in random sampling
    numberOfFolds = Setting(10)             # Setting for the number of folds in cross validation
    selectedFold = Setting(1)               # Setting for the selected fold in cross validation

    def __init__(self, parent=None, signalManager=None, settings=None):
        super().__init__(parent, signalManager, settings)


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
        self.rndSamplingRB = gui.appendRadioButton(samplingTypesBox, self, "samplingType", "Random Sampling:")

        # indent under Random Sampling which also acts as a sample size type radio button group
        rndSmplIndent = gui.indentedBox(samplingTypesBox)

        self.replicationCheckBox = gui.checkBox(rndSmplIndent, self, "withReplication", "with replication")

        self.sampleSizeRB = gui.appendRadioButton(rndSmplIndent, self, "sampleSizeType", "Sample size:")

        # indent level 2 under sample size
        smplSizeIndent = gui.indentedBox(rndSmplIndent)

        self.sampleSizeSpin = gui.spin(smplSizeIndent, self, "sampleSizeNumber", 1, 1000000000)

        self.sampleProportionsRB = gui.appendRadioButton(rndSmplIndent, self, "sampleSizeType", "Sample proportions:")

        # indent level 2 under sample proportions
        smplPropIndent = gui.indentedBox(rndSmplIndent)

        self.sampleProportionsSlider = gui.hSlider(smplPropIndent, self, "sampleSizePercentage", minValue=1, maxValue=100, ticks=10, labelFormat="%d%%")

        # Cross Validation
        self.crossValidationRB = gui.appendRadioButton(samplingTypesBox, self, "samplingType", "Cross Validation:")

        # indent under cross validation
        crossValidIndent = gui.indentedBox(samplingTypesBox)

        self.numberOfFoldsSpin = gui.spin(crossValidIndent, self, "numberOfFolds", 2, 100, label="Number of folds:")

        self.selectedFoldSpin = gui.spin(crossValidIndent, self, "selectedFold", 1, 100, label="Selected fold:")

        # Sample Data Button
        gui.button(self.controlArea, self, "Sample Data")