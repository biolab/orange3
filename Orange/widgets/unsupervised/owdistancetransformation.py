import numpy as np
from AnyQt.QtCore import Qt

from Orange.util import scale
from Orange.misc import DistMatrix
from Orange.widgets import widget, gui, settings
from Orange.widgets.widget import Input, Output


class OWDistanceTransformation(widget.OWWidget):
    name = "Distance Transformation"
    description = "Transform distances according to selected criteria."
    icon = "icons/DistancesTransformation.svg"

    class Inputs:
        distances = Input("Distances", DistMatrix)

    class Outputs:
        distances = Output("Distances", DistMatrix, dynamic=False)

    want_main_area = False
    resizing_enabled = False
    buttons_area_orientation = Qt.Vertical

    normalization_method = settings.Setting(0)
    inversion_method = settings.Setting(0)
    autocommit = settings.Setting(True)

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

        gui.radioButtons(self.controlArea, self, "normalization_method",
                         box="Normalization",
                         btnLabels=[x[0] for x in self.normalization_options],
                         callback=self._invalidate)

        gui.radioButtons(self.controlArea, self, "inversion_method",
                         box="Inversion",
                         btnLabels=[x[0] for x in self.inversion_options],
                         callback=self._invalidate)

        gui.auto_commit(self.controlArea, self, "autocommit", "Apply")

    @Inputs.distances
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
        self.Outputs.distances.send(distances)

    def send_report(self):
        norm, normopt = self.normalization_method, self.normalization_options
        inv, invopt = self.inversion_method, self.inversion_options
        parts = []
        if inv:
            parts.append('inversion ({})'.format(invopt[inv][0]))
        if norm:
            parts.append('normalization ({})'.format(normopt[norm][0]))
        self.report_items(
            'Model parameters',
            {'Transformation': ', '.join(parts).capitalize() or 'None'})

    def _invalidate(self):
        self.commit()
