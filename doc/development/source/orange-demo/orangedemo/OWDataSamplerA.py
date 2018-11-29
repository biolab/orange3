# [start-snippet-1]
import numpy

import Orange.data
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets import gui

class OWDataSamplerA(OWWidget):
    name = "Data Sampler"
    description = "Randomly selects a subset of instances from the dataset"
    icon = "icons/DataSamplerA.svg"
    priority = 10

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        sample = Output("Sampled Data", Orange.data.Table)

    want_main_area = False

    def __init__(self):
        super().__init__()

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.infoa = gui.widgetLabel(box, 'No data on input yet, waiting to get something.')
        self.infob = gui.widgetLabel(box, '')
# [end-snippet-1]

# [start-snippet-2]
    @Inputs.data
    def set_data(self, dataset):
        if dataset is not None:
            self.infoa.setText('%d instances in input dataset' % len(dataset))
            indices = numpy.random.permutation(len(dataset))
            indices = indices[:int(numpy.ceil(len(dataset) * 0.1))]
            sample = dataset[indices]
            self.infob.setText('%d sampled instances' % len(sample))
            self.Outputs.sample.send(sample)
        else:
            self.infoa.setText('No data on input yet, waiting to get something.')
            self.infob.setText('')
            self.Outputs.sample.send("Sampled Data")
# [end-snippet-2]


# [start-snippet-3]
if __name__ == "__main__":
    WidgetPreview(OWDataSamplerA).run(Orange.data.Table("iris"))
# [end-snippet-3]
