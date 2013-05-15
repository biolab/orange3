import random
from PyQt4 import QtCore
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data.table import Table
from sklearn import cross_validation


# TODO: Util methods that really shouldn't be here and should be moved outside of this module
def makeRandomIndices(number, datasetLength, repetition=False, randomSeed=None):
    """
    :param number: number of indices to be selected from the base pool
    :param datasetLength: size of the base pool
    :param repetition: repeat indices bool
    :param randomSeed: random seed for random number generator
    :return: touple which contains list with selected indices and list with the remainder of the base pool
    """
    if randomSeed:
        random.seed(randomSeed)
    basePool = range(datasetLength)
    if repetition:
        indices = [random.choice(basePool) for _ in range(number)]
    else:
        indices = random.sample(basePool, number)

    return (indices, list(set(basePool) - set(indices)))

def makeRandomGroups(numberOfGroups, datasetLength, randomSeed=None):
    groups = []
    groupSize = datasetLength//numberOfGroups
    remainder = [None] * datasetLength
    for i in range(numberOfGroups-1):
        group, remainder = makeRandomIndices(groupSize, len(remainder), randomSeed=randomSeed)
        groups.append(group)
    if (groupSize - len(remainder)) < 2:
        groups.append(remainder)
    else:
        # append items in last remainder to different groups so we don't get the extraordinarily small group at the end
        for i, item in enumerate(remainder):
            groups[i].append(item)

    return groups

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
    samplingType = Setting(0)               # Setting for which RB is enabled at start in sampling types radio button group
    withReplication = Setting(False)        # Setting for data replication option in random sampling
    sampleSizeType = Setting(0)             # Setting for which RB is enabled at start in sample size radio button group
    sampleSizeNumber = Setting(1)           # Setting for the number of examples in a sample in random sampling
    sampleSizePercentage = Setting(30)      # Setting for the percentage of examples in a sample in random sampling
    numberOfFolds = Setting(10)             # Setting for the number of folds in cross validation
    selectedFold = Setting(1)               # Setting for the selected fold in cross validation


    def __init__(self, parent=None, signalManager=None, settings=None):
        super().__init__(parent, signalManager, settings)

        self.data = None

        # Information Box
        infoBox = gui.widgetBox(self.controlArea, "Information:", addSpace=True)
        self.dataInfoLabel = gui.widgetLabel(infoBox, 'No data on input.')
        self.methodInfoLabel = gui.widgetLabel(infoBox, ' ')
        self.outputInfoLabel = gui.widgetLabel(infoBox, ' ')


        # Options Box
        optionsBox = gui.widgetBox(self.controlArea, "Options:", addSpace=True)
        stratifiedCheckBox = gui.checkBox(optionsBox, self, "stratified", "Stratified (if possible)")
        setRndSeedCheckBox = gui.checkWithSpin(optionsBox, self, "Set Random Seed", 1, 32767, "setRandomSeed", "randomSeed")

        # Box that will hold Sampling Types radio buttons
        samplingTypesBox = gui.widgetBox(self.controlArea, "Sampling types:", addSpace=True)

        # Random Sampling
        samplingTypesBG = gui.radioButtonsInBox(samplingTypesBox, self, "samplingType", [], callback=self.fadeSamplingTypes)
        randomSamplingRB = gui.appendRadioButton(samplingTypesBG, self, "samplingType", "Random Sampling:", insertInto=samplingTypesBox)

        # indent under Random Sampling which also acts as a sample size type radio button group
        self.rndSmplIndent = gui.indentedBox(samplingTypesBox)

        replicationCheckBox = gui.checkBox(self.rndSmplIndent, self, "withReplication", "with replication")

        sampleTypesBG = gui.radioButtonsInBox(self.rndSmplIndent, self, "sampleSizeType", [], callback=self.fadeSampleSizeTypes)
        sampleSizeRB = gui.appendRadioButton(sampleTypesBG, self, "sampleSizeType", "Sample size:", insertInto=self.rndSmplIndent)

        # indent level 2 under sample size
        self.smplSizeIndent = gui.indentedBox(self.rndSmplIndent)

        self.sampleSizeSpin = gui.spin(self.smplSizeIndent, self, "sampleSizeNumber", 1, 1000000000)
        # back to level 1 indent

        sampleProportionsRB = gui.appendRadioButton(sampleTypesBG, self, "sampleSizeType", "Sample proportions:", insertInto=self.rndSmplIndent)

        # indent level 2 under sample proportions
        self.smplPropIndent = gui.indentedBox(self.rndSmplIndent)

        sampleProportionsSlider = gui.hSlider(self.smplPropIndent, self, "sampleSizePercentage", minValue=1, maxValue=100, ticks=10, labelFormat="%d%%")
        # end of indentation

        # Cross Validation
        crossValidationRB = gui.appendRadioButton(samplingTypesBG, self, "samplingType", "Cross Validation:", insertInto=samplingTypesBox)

        # indent under cross validation
        self.crossValidIndent = gui.indentedBox(samplingTypesBox)

        numberOfFoldsSpin = gui.spin(self.crossValidIndent, self, "numberOfFolds", 2, 100, label="Number of folds:", callback=self.updateSelectedFoldSpin)

        self.selectedFoldSpin = gui.spin(self.crossValidIndent, self, "selectedFold", 1, 100, label="Selected fold:")
        # end of indentation

        # Sample Data Button
        gui.button(self.controlArea, self, "Sample Data", callback=self.sendData)


    # GUI METHODS
    def updateSelectedFoldSpin(self):
        self.selectedFoldSpin.setMaximum(self.numberOfFolds)
        pass

    def fadeSamplingTypes(self):
        if self.samplingType == 0:
            self.rndSmplIndent.setEnabled(True)
            self.crossValidIndent.setEnabled(False)
        else:
            self.rndSmplIndent.setEnabled(False)
            self.crossValidIndent.setEnabled(True)

    def fadeSampleSizeTypes(self):
        if self.sampleSizeType == 0:
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
            self.sampleSizeSpin.setMaximum(len(dataset))
            self.sendData()

        # if we get no data forward no data
        else:
            self.dataInfoLabel.setText('No data on input.')
            self.methodInfoLabel.setText('')
            self.infoc.setText('')
            self.send("Data Sample", None)
            self.send("Remaining Data", None)
            self.data = None


    # process data and sends it forward
    def sendData(self):
        if not self.data:
            return
        dataSize = len(self.data)
        rndSeed = None
        if self.setRandomSeed:
            rndSeed = self.randomSeed

        # random sampling
        if self.samplingType == 0:
            # size by number
            if self.sampleSizeType == 0:
                nCases = self.sampleSizeNumber
            # size by percentage
            else:
                nCases = self.sampleSizePercentage*dataSize//100

            if self.withReplication:
                self.methodInfoLabel.setText('Random sampling with repetitions, %d instances.' % nCases)
            else:
                self.methodInfoLabel.setText('Random sampling, %d instances.' % nCases)

            sampleIndices, remainderIndices = makeRandomIndices(nCases, dataSize, repetition=self.withReplication, randomSeed=rndSeed)

            self.outputInfoLabel.setText('Outputting %d instances.' % len(sampleIndices))

        # cross validation
        else:
            if self.selectedFold > self.numberOfFolds:
                self.selectedFold = self.numberOfFolds
            groups = makeRandomGroups(self.numberOfFolds, dataSize, randomSeed=rndSeed)
            sampleIndices = groups.pop(self.selectedFold-1)
            remainderIndices = [].extend(groups)
            self.methodInfoLabel.setText('Cross validation, %d groups.' % self.numberOfFolds)
            self.outputInfoLabel.setText('Outputting group number %d.' % self.selectedFold)

        self.send("Data Sample", self.data[sampleIndices])
        self.send("Remaining Data", self.data[remainderIndices])
