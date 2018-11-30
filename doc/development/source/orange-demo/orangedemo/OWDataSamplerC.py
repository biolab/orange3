import numpy

import Orange.data
from Orange.widgets.widget import OWWidget, Input, Output, settings
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets import gui


class OWDataSamplerC(OWWidget):
    name = "Data Sampler (C)"
    description = "Randomly selects a subset of instances from the dataset."
    icon = "icons/DataSamplerC.svg"
    priority = 30

    class Inputs:
        data = Input("Data", Orange.data.Table)
# [start-snippet-1]
    class Outputs:
        sample = Output("Sampled Data", Orange.data.Table)
        other = Output("Other Data", Orange.data.Table)
# [end-snippet-1]
    proportion = settings.Setting(50)
    commitOnChange = settings.Setting(0)

    want_main_area = False

    def __init__(self):
        super().__init__()

        self.dataset = None
        self.sample = None
        self.otherdata = None

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.infoa = gui.widgetLabel(box, 'No data on input yet, waiting to get something.')
        self.infob = gui.widgetLabel(box, '')

        gui.separator(self.controlArea)
        self.optionsBox = gui.widgetBox(self.controlArea, "Options")
        gui.spin(self.optionsBox, self, 'proportion',
                 minv=10, maxv=90, step=10,
                 label='Sample Size [%]:',
                 callback=[self.selection, self.checkCommit])
        gui.checkBox(self.optionsBox, self, 'commitOnChange',
                     'Commit data on selection change')
        gui.button(self.optionsBox, self, "Commit", callback=self.commit)
        self.optionsBox.setDisabled(True)

        self.resize(100,50)

    @Inputs.data
    def set_data(self, dataset):
        if dataset is not None:
            self.dataset = dataset
            self.infoa.setText('%d instances in input dataset' % len(dataset))
            self.optionsBox.setDisabled(False)
            self.selection()
        else:
            self.sample = None
            self.otherdata = None
            self.optionsBox.setDisabled(True)
            self.infoa.setText('No data on input yet, waiting to get something.')
            self.infob.setText('')
        self.commit()

    def selection(self):
        if self.dataset is None:
            return

        n_selected = int(numpy.ceil(len(self.dataset) * self.proportion / 100.))
        indices = numpy.random.permutation(len(self.dataset))
        indices_sample = indices[:n_selected]
        indices_other = indices[n_selected:]
        self.sample = self.dataset[indices_sample]
        self.otherdata = self.dataset[indices_other]
        self.infob.setText('%d sampled instances' % len(self.sample))

    def commit(self):
        self.Outputs.sample.send(self.sample)
        self.Outputs.sample.send(self.otherdata)

    def checkCommit(self):
        if self.commitOnChange:
            self.commit()


if __name__ == "__main__":
    WidgetPreview(OWDataSamplerC).run(Orange.data.Table("iris"))
