import sys
import numpy

import Orange.data
from Orange.widgets import widget, gui, settings

# [start-snippet-1]
class OWDataSamplerB(widget.OWWidget):
    name = "Data Sampler (B)"
    description = "Randomly selects a subset of instances from the data set."
    icon = "icons/DataSamplerB.svg"
    priority = 20

    inputs = [("Data", Orange.data.TableBase, "set_data")]
    outputs = [("Sampled Data", Orange.data.Table)]

    proportion = settings.Setting(50)
    commitOnChange = settings.Setting(0)
# [end-snippet-1]

    want_main_area = False

    def __init__(self):
        super().__init__()

        self.dataset = None
        self.sample = None

        # GUI
# [start-snippet-3]
        box = gui.widgetBox(self.controlArea, "Info")
        self.infoa = gui.widgetLabel(box, 'No data on input yet, waiting to get something.')
        self.infob = gui.widgetLabel(box, '')

        gui.separator(self.controlArea)
        self.optionsBox = gui.widgetBox(self.controlArea, "Options")
        gui.spin(self.optionsBox, self, 'proportion',
                 minv=10, maxv=90, step=10, label='Sample Size [%]:',
                 callback=[self.selection, self.checkCommit])
        gui.checkBox(self.optionsBox, self, 'commitOnChange',
                     'Commit data on selection change')
        gui.button(self.optionsBox, self, "Commit", callback=self.commit)
        self.optionsBox.setDisabled(True)
# [end-snippet-3]

# [start-snippet-4]
    def set_data(self, dataset):
        if dataset is not None:
            self.dataset = dataset
            self.infoa.setText('%d instances in input data set' % len(dataset))
            self.optionsBox.setDisabled(False)
            self.selection()
        else:
            self.dataset = None
            self.sample = None
            self.optionsBox.setDisabled(False)
            self.infoa.setText('No data on input yet, waiting to get something.')
            self.infob.setText('')
        self.commit()

    def selection(self):
        if self.dataset is None:
            return

        n_selected = int(numpy.ceil(len(self.dataset) * self.proportion / 100.))
        indices = numpy.random.permutation(len(self.dataset))
        indices = indices[:n_selected]
        self.sample = self.dataset[indices]
        self.infob.setText('%d sampled instances' % len(self.sample))

    def commit(self):
        self.send("Sampled Data", self.sample)

    def checkCommit(self):
        if self.commitOnChange:
            self.commit()
# [end-snippet-4]

def main(argv=sys.argv):
    from PyQt4.QtGui import QApplication
    app = QApplication(list(argv))
    args = app.argv()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "iris"

    ow = OWDataSamplerB()
    ow.show()
    ow.raise_()

    dataset = Orange.data.Table(filename)
    ow.set_data(dataset)
    ow.handleNewSignals()
    app.exec_()
    ow.set_data(None)
    ow.handleNewSignals()
    return 0

if __name__=="__main__":
    sys.exit(main())
