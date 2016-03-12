import numpy as np
from collections import OrderedDict

from Orange.util import scale
from Orange.misc import DistMatrix
from Orange.widgets import widget, gui, settings


class OWDistanceTransformation(widget.OWWidget):
    name = "Distance Transformation"
    description = "Transform distances according to selected criteria."
    icon = "icons/DistancesTransformation.svg"

    learner_name = settings.Setting("Distance Transformation")

    inputs = [("Distances", DistMatrix, "set_data")]
    outputs = [("Distances", DistMatrix)]

    want_main_area = False
    resizing_enabled = False

    normalization_method = settings.Setting(0)
    inversion_method = settings.Setting(0)
    autocommit = settings.Setting(False)

    normalization_options = (
        ("No normalization", lambda x: x),
        ("To interval [0, 1]", lambda x: scale(x, min=0, max=1)),
        ("To interval [-1, 1]", lambda x: scale(x, min=-1, max=1)),
        ("Sigmoid function: 1/(1+exp(-X))", lambda x: 1/(1+np.exp(-x))),
    )

    inversion_options = (
        ("No inversion", lambda x: x),
        ("-X", lambda x: -x),
        ("1 - X", lambda x: 1-x),
        ("max(X) - X", lambda x: np.max(x) - x),
        ("1/X", lambda x: 1/x),
    )

    def __init__(self):
        super().__init__()

        self.data = None

        box = gui.widgetBox(self.controlArea, "Normalization")
        gui.radioButtons(box, self, "normalization_method",
                         btnLabels=[x[0] for x in self.normalization_options],
                         callback=self._invalidate)

        box = gui.widgetBox(self.controlArea, "Inversion")
        gui.radioButtons(box, self, "inversion_method",
                         btnLabels=[x[0] for x in self.inversion_options],
                         callback=self._invalidate)

        gui.auto_commit(self.controlArea, self, "autocommit", "Apply",
                        checkbox_label="Apply on any change")
        gui.separator(box)
        box.layout().addWidget(self.report_button)

    def set_data(self, data):
        self.data = data
        self.unconditional_commit()

    def commit(self):
        distances = self.data
        if distances is not None:
            # normalize
            norm = self.normalization_options[self.normalization_method][1]
            distances = norm(distances)

            # invert
            inv = self.inversion_options[self.inversion_method][1]
            distances = inv(distances)
        self.send("Distances", distances)

    def send_report(self):
        items = OrderedDict()
        if self.normalization_method == 0:
            if self.inversion_method == 0:
                items['Transformation'] = 'None'
            else:
                items['Transformation'] = 'Inversion (' + \
                    self.inversion_options[self.inversion_method][0] + ')'
        else:
            if self.inversion_method == 0:
                items['Transformation'] = ('Normalization {}'.
                                            format(self.normalization_options
                                            [self.normalization_method][0])
                                            .lower())
            else:
                items['Transformation'] = ('Inversion (' + 
                                            self.inversion_options
                                            [self.inversion_method][0].lower()
                                            + '), Normalization ' + 
                                            self.normalization_options
                                            [self.normalization_method][0]
                                            .lower())
        self.report_items("Model parameters", items)

    def _invalidate(self):
        self.commit()
