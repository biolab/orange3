from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
from Orange.data.table import Table
from sklearn import cross_validation


# TODO: Util methods that really shouldn't be here and should be moved outside of this module
# def makeRandomIndices(number, datasetLength, repetition=False, randomSeed=None):
#     """
#     :param number: number of indices to be selected from the base pool
#     :param datasetLength: size of the base pool
#     :param repetition: repeat indices bool
#     :param randomSeed: random seed for random number generator
#     :return: touple which contains list with selected indices and list with the remainder of the base pool
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
    #withReplication = Setting(False)        # Setting for data replication option in random sampling
    sampleSizeType = Setting(0)             # Setting for which RB is enabled at start in sample size radio button group
    sampleSizeNumber = Setting(1)           # Setting for the number of examples in a sample in random sampling
    sampleSizePercentage = Setting(30)      # Setting for the percentage of examples in a sample in random sampling
    numberOfFolds = Setting(10)             # Setting for the number of folds in cross validation
    selectedFold = Setting(1)               # Setting for the selected fold in cross validation

    def __init__(self, parent=None, signalManager=None, settings=None):
        super().__init__(parent, signalManager, settings)

        self.data = None
        self.groups = None
        self.CVSettings = [self.stratified, self.numberOfFolds]

        # Information Box
        infoBox = gui.widgetBox(self.controlArea, "Information:", addSpace=True)
        self.dataInfoLabel = gui.widgetLabel(infoBox, 'No data on input.')
        self.methodInfoLabel = gui.widgetLabel(infoBox, ' ')
        self.outputInfoLabel = gui.widgetLabel(infoBox, ' ')

        # Options Box
        optionsBox = gui.widgetBox(self.controlArea, "Options:", addSpace=True)
        # stratified check box
        gui.checkBox(optionsBox, self, "stratified", "Stratified (if possible)", callback=self.settingsChanged)
        # random seed check with spin
        gui.checkWithSpin(
            optionsBox, self, "Random Seed:", 1, 32767, "setRandomSeed", "randomSeed",
            checkCallback=self.settingsChanged, spinCallback=self.settingsChanged)

        # Box that will hold Sampling Types radio buttons
        samplingTypesBox = gui.widgetBox(self.controlArea, "Sampling types:", addSpace=True)

        # Random Sampling
        samplingTypesBG = gui.radioButtonsInBox(
            samplingTypesBox, self, "samplingType", [], callback=[self.fadeSamplingTypes, self.settingsChanged])
        # random sampling radio button
        gui.appendRadioButton(
            samplingTypesBG, self, "samplingType", "Random Sampling:", insertInto=samplingTypesBox)

        # indent under Random Sampling which also acts as a sample size type radio button group
        self.rndSmplIndent = gui.indentedBox(samplingTypesBox)

        #replicationCheckBox = gui.checkBox(self.rndSmplIndent, self, "withReplication", "with replication")

        sampleTypesBG = gui.radioButtonsInBox(
            self.rndSmplIndent, self, "sampleSizeType", [], callback=[self.fadeSampleSizeTypes, self.settingsChanged])
        # sample size radio button
        gui.appendRadioButton(
            sampleTypesBG, self, "sampleSizeType", "Sample size:", insertInto=self.rndSmplIndent)

        # indent level 2 under sample size
        self.smplSizeIndent = gui.indentedBox(self.rndSmplIndent)

        self.sampleSizeSpin = gui.spin(self.smplSizeIndent, self, "sampleSizeNumber", 1, 1000000000)
        # back to level 1 indent
        # sample size type radio button
        gui.appendRadioButton(
            sampleTypesBG, self, "sampleSizeType", "Sample proportions:", insertInto=self.rndSmplIndent)

        # indent level 2 under sample proportions
        self.smplPropIndent = gui.indentedBox(self.rndSmplIndent)
        # sample proportions slider
        gui.hSlider(
            self.smplPropIndent, self, "sampleSizePercentage", minValue=1, maxValue=100, ticks=10, labelFormat="%d%%")
        # end of indentation

        # Cross Validation
        # cross validation radio button
        gui.appendRadioButton(
            samplingTypesBG, self, "samplingType", "Cross Validation:", insertInto=samplingTypesBox)

        # indent under cross validation
        self.crossValidIndent = gui.indentedBox(samplingTypesBox)
        #number of folds spin
        gui.spin(
            self.crossValidIndent, self, "numberOfFolds", 2, 100, label="Number of folds:",
            callback=[self.updateSelectedFoldSpin, self.settingsChanged])

        self.selectedFoldSpin = gui.spin(
            self.crossValidIndent, self, "selectedFold", 1, 100, label="Selected fold:",
            callback=[self.updateSelectedFoldSpin, self.settingsChanged])
        # end of indentation

        # Sample Data Button
        self.sampleDataButton = gui.button(self.controlArea, self, "Sample Data", callback=self.sendData)

        self.fadeSamplingTypes()
        self.fadeSampleSizeTypes()

    # GUI METHODS
    def settingsChanged(self):
        self.sampleDataButton.setEnabled(True)

    def updateSelectedFoldSpin(self):
        """
        Method that we use to set the selected fold maximum so that we can't select a fold that isn't computed.
        """
        self.selectedFoldSpin.setMaximum(self.numberOfFolds)

    def fadeSamplingTypes(self):
        """
        Switches between sampling types by fading one out and enabling the other,
        depends on which radio button is checked.
        """
        if self.samplingType == 0:
            self.rndSmplIndent.setEnabled(True)
            self.crossValidIndent.setEnabled(False)
        else:
            self.rndSmplIndent.setEnabled(False)
            self.crossValidIndent.setEnabled(True)

    def fadeSampleSizeTypes(self):
        """
        Switches between sample size types by fading one out and enabling the other,
        depends on which radio button is checked.
        """
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
            self.dataChanged = True
            self.sendData()
        # if we get no data forward no data
        else:
            self.dataInfoLabel.setText('No data on input.')
            self.methodInfoLabel.setText('')
            self.outputInfoLabel.setText('')
            self.send("Data Sample", None)
            self.send("Remaining Data", None)
            self.data = None

    # process data and sends it forward
    def sendData(self):
        if not self.data:
            return
        data_size = len(self.data)
        rnd_seed = None
        if self.setRandomSeed:
            rnd_seed = self.randomSeed
        # random sampling
        if self.samplingType == 0:
            # size by number
            if self.sampleSizeType == 0:
                n_cases = self.sampleSizeNumber
            # size by percentage
            else:
                n_cases = self.sampleSizePercentage*data_size//100

            self.methodInfoLabel.setText('Random sampling, %d instances.' % n_cases)

            if self.stratified:
                # n_iter=1 means we compute only 1 split and iterator holds only 1 item
                ss = cross_validation.StratifiedShuffleSplit(
                    self.data.Y.flatten(), n_iter=1, test_size=n_cases, random_state=rnd_seed)
            else:
                ss = cross_validation.ShuffleSplit(data_size, n_iter=1, test_size=n_cases, random_state=rnd_seed)

            remainder_indices, sample_indices = [(train_index, test_index) for train_index, test_index in ss][0]
            self.outputInfoLabel.setText('Outputting %d instances.' % len(sample_indices))
        # cross validation
        else:
            self.methodInfoLabel.setText('Cross validation, %d groups.' % self.numberOfFolds)
            self.outputInfoLabel.setText('Outputting group number %d.' % self.selectedFold)

            if self.dataChanged or not self.groups or not (self.CVSettings == [self.stratified, self.numberOfFolds]):
                if self.stratified:
                    kf = cross_validation.StratifiedKFold(self.data.Y.flatten(), n_folds=self.numberOfFolds)
                else:
                    kf = cross_validation.KFold(data_size, n_folds=self.numberOfFolds)
                self.groups = [(train_index, test_index) for train_index, test_index in kf]

            remainder_indices, sample_indices = self.groups[self.selectedFold - 1]

            self.dataChanged = False
            # remember cross validation settings so we compute only if those settings change
            self.CVSettings = [self.stratified, self.numberOfFolds]

        self.send("Data Sample", self.data[sample_indices])
        self.send("Remaining Data", self.data[remainder_indices])
        self.sampleDataButton.setEnabled(False)