from AnyQt.QtWidgets import QSlider, QLabel
from AnyQt.QtCore import Qt, Signal

from Orange.widgets.utils import getdeepattr

from .base import miscellanea
from .boxes import hBox
from .labels import widgetLabel
from .callbacks import connect_control

__all__ = ["hSlider", "valueSlider"]


def hSlider(widget, master, value, box=None, minValue=0, maxValue=10, step=1,
            callback=None, label=None, labelFormat=" %d", ticks=False,
            divideFactor=1.0, vertical=False, createLabel=True, width=None,
            intOnly=True, **misc):
    """
    Construct a slider.

    :param widget: the widget into which the box is inserted
    :type widget: QWidget or None
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param value: the master's attribute with which the value is synchronized
    :type value:  str
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :param label: a label that is inserted into the box
    :type label: str
    :param callback: a function that is called when the value is changed
    :type callback: function

    :param minValue: minimal value
    :type minValue: int or float
    :param maxValue: maximal value
    :type maxValue: int or float
    :param step: step size
    :type step: int or float
    :param labelFormat: the label format; default is `" %d"`
    :type labelFormat: str
    :param ticks: if set to `True`, ticks are added below the slider
    :type ticks: bool
    :param divideFactor: a factor with which the displayed value is divided
    :type divideFactor: float
    :param vertical: if set to `True`, the slider is vertical
    :type vertical: bool
    :param createLabel: unless set to `False`, labels for minimal, maximal
        and the current value are added to the widget
    :type createLabel: bool
    :param width: the width of the slider
    :type width: int
    :param intOnly: if `True`, the slider value is integer (the slider is
        of type :obj:`QSlider`) otherwise it is float
        (:obj:`FloatSlider`, derived in turn from :obj:`QSlider`).
    :type intOnly: bool
    :rtype: :obj:`QSlider` or :obj:`FloatSlider`
    """
    sliderBox = hBox(widget, box, addToLayout=False)
    if label:
        widgetLabel(sliderBox, label)
    sliderOrient = Qt.Vertical if vertical else Qt.Horizontal
    if intOnly:
        slider = QSlider(sliderOrient, sliderBox)
        slider.setRange(minValue, maxValue)
        if step:
            slider.setSingleStep(step)
            slider.setPageStep(step)
            slider.setTickInterval(step)
        signal = slider.valueChanged[int]
    else:
        slider = FloatSlider(sliderOrient, minValue, maxValue, step)
        signal = slider.valueChangedFloat[float]
    sliderBox.layout().addWidget(slider)
    slider.setValue(getdeepattr(master, value))
    if width:
        slider.setFixedWidth(width)
    if ticks:
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(ticks)

    if createLabel:
        label = QLabel(sliderBox)
        sliderBox.layout().addWidget(label)
        label.setText(labelFormat % minValue)
        width1 = label.sizeHint().width()
        label.setText(labelFormat % maxValue)
        width2 = label.sizeHint().width()
        label.setFixedSize(max(width1, width2), label.sizeHint().height())
        txt = labelFormat % (getdeepattr(master, value) / divideFactor)
        label.setText(txt)
        label.setLbl = lambda x: \
            label.setText(labelFormat % (x / divideFactor))
        signal.connect(label.setLbl)

    connect_control(
        master, value, slider,
        signal=signal,
        update_control=lambda val: val is not None and slider.setValue(val),
        callback=callback)

    miscellanea(slider, sliderBox, widget, **misc)
    return slider


def valueSlider(widget, master, value, box=None, label=None,
                values=(), labelFormat=" %d", ticks=False,
                callback=None, vertical=False, width=None, **misc):
    """
    Construct a slider with different values.

    :param widget: the widget into which the box is inserted
    :type widget: QWidget or None
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param value: the master's attribute with which the value is synchronized
    :type value:  str
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :param label: a label that is inserted into the box
    :type label: str
    :param values: values at different slider positions
    :type values: list of int
    :param labelFormat: label format; default is `" %d"`; can also be a function
    :type labelFormat: str or func
    :param callback: a function that is called when the value is changed
    :type callback: function

    :param ticks: if set to `True`, ticks are added below the slider
    :type ticks: bool
    :param vertical: if set to `True`, the slider is vertical
    :type vertical: bool
    :param width: the width of the slider
    :type width: int
    :rtype: :obj:`QSlider`
    """
    if isinstance(labelFormat, str):
        labelFormat = lambda x, f=labelFormat: f % x

    sliderBox = hBox(widget, box, addToLayout=False)
    if label:
        widgetLabel(sliderBox, label)
    slider_orient = Qt.Vertical if vertical else Qt.Horizontal
    slider = QSlider(slider_orient, sliderBox)
    slider.setRange(0, len(values) - 1)
    slider.setSingleStep(1)
    slider.setPageStep(1)
    slider.setTickInterval(1)
    sliderBox.layout().addWidget(slider)
    slider.setValue(values.index(getdeepattr(master, value)))
    if width:
        slider.setFixedWidth(width)
    if ticks:
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(ticks)

    max_label_size = 0
    slider.value_label = value_label = QLabel(sliderBox)
    value_label.setAlignment(Qt.AlignRight)
    sliderBox.layout().addWidget(value_label)
    for lb in values:
        value_label.setText(labelFormat(lb))
        max_label_size = max(max_label_size, value_label.sizeHint().width())
    value_label.setFixedSize(max_label_size, value_label.sizeHint().height())
    value_label.setText(labelFormat(getdeepattr(master, value)))
    value_label.set_label = lambda x: value_label.setText(labelFormat(values[x]))
    slider.valueChanged[int].connect(value_label.set_label)

    connect_control(
        master, value, slider,
        signal=slider.valueChanged[int],
        update_control=lambda val: val is not None and slider.setValue(values.index(val)),
        update_value=lambda val: setattr(master, value, values[val]),
        callback=callback)

    miscellanea(slider, sliderBox, widget, **misc)
    return slider


class FloatSlider(QSlider):
    """
    Slider for continuous values.

    The slider is derived from `QtGui.QSlider`, but maps from its discrete
    numbers to the desired continuous interval.
    """
    valueChangedFloat = Signal(float)

    def __init__(self, orientation, min_value, max_value, step, parent=None):
        super().__init__(orientation, parent)
        self.setScale(min_value, max_value, step)
        self.valueChanged[int].connect(self._send_value)

    def _update(self):
        self.setSingleStep(1)
        if self.min_value != self.max_value:
            self.setEnabled(True)
            self.setMinimum(int(self.min_value / self.step))
            self.setMaximum(int(self.max_value / self.step))
        else:
            self.setEnabled(False)

    def _send_value(self, slider_value):
        value = min(max(slider_value * self.step, self.min_value),
                    self.max_value)
        self.valueChangedFloat.emit(value)

    def setValue(self, value):
        """
        Set current value. The value is divided by `step`

        Args:
            value: new value
        """
        super().setValue(value // self.step)

    def setScale(self, minValue, maxValue, step=0):
        """
        Set slider's ranges (compatibility with qwtSlider).

        Args:
            minValue (float): minimal value
            maxValue (float): maximal value
            step (float): step
        """
        if minValue >= maxValue:
            ## It would be more logical to disable the slider in this case
            ## (self.setEnabled(False))
            ## However, we do nothing to keep consistency with Qwt
            # TODO If it's related to Qwt, remove it
            return
        if step <= 0 or step > (maxValue - minValue):
            if isinstance(maxValue, int) and isinstance(minValue, int):
                step = 1
            else:
                step = float(minValue - maxValue) / 100.0
        self.min_value = float(minValue)
        self.max_value = float(maxValue)
        self.step = step
        self._update()

    def setRange(self, minValue, maxValue, step=1.0):
        """
        Set slider's ranges (compatibility with qwtSlider).

        Args:
            minValue (float): minimal value
            maxValue (float): maximal value
            step (float): step
        """
        # For compatibility with qwtSlider
        # TODO If it's related to Qwt, remove it
        self.setScale(minValue, maxValue, step)
