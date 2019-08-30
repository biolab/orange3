# [start-snippet-1]
import sys
import numpy

import Orange.data
from Orange.widgets import widget, gui
from Orange.widgets.utils.signals import Input, Output


class OWDataSamplerA(widget.OWWidget):
    name = "Data Sampler"
    description = "Randomly selects a subset of instances from the data set"
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
        self.infoa = gui.widgetLabel(
            box, "No data on input yet, waiting to get something.")
        self.infob = gui.widgetLabel(box, '')
# [end-snippet-1]

# [start-snippet-2]
    @Inputs.data
    def set_data(self, dataset):
        if dataset is not None:
            self.infoa.setText("%d instances in input data set" % len(dataset))
            indices = numpy.random.permutation(len(dataset))
            indices = indices[:int(numpy.ceil(len(dataset) * 0.1))]
            sample = dataset[indices]
            self.infob.setText("%d sampled instances" % len(sample))
            self.Outputs.sample.send(sample)
        else:
            self.infoa.setText(
                "No data on input yet, waiting to get something.")
            self.infob.setText('')
            self.Outputs.sample.send(None)
# [end-snippet-2]

# [start-snippet-3]


def main(argv=sys.argv):
    from AnyQt.QtWidgets import QApplication
    app = QApplication(list(argv))
    args = app.arguments()
    if len(args) > 1:
        filename = args[1]
    else:
        filename = "iris"

    ow = OWDataSamplerA()
    ow.show()
    ow.raise_()

    dataset = Orange.data.Table(filename)
    ow.set_data(dataset)
    ow.handleNewSignals()
    app.exec_()
    ow.set_data(None)
    ow.handleNewSignals()
    return 0


if __name__ == "__main__":
    sys.exit(main())

# [end-snippet-3]
