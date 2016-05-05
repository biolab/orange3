"""This module defines various types of options and their values.
"""
import numpy as np
from PyQt4 import QtGui, QtCore


def textify(obj):
    """Creates readable object representation."""
    return str(obj).capitalize().replace('_', ' ')


class ValidationError(Exception):
    """Base class for validation exceptions."""
    pass


class Value:
    """Value descriptor. Contains an `Option` instance and its value."""

    def __init__(self, value, option):
        self._value = value
        self.option = option
        self.callbacks = []
        self.widget = None

    @property
    def value(self):
        """Current option value."""
        return self._value

    @value.setter
    def value(self, value):
        if self._value != value:
            self._value = value
            self.on_change()

    def set_value(self, value):
        """Alternate value setter (can be used as a callback)."""
        self.value = value

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def on_change(self, *args):
        """Callback on value change event."""
        self.update_gui()
        for callback in self.callbacks:
            callback()

    def validate(self):
        """Checks whether current value is a valid one.

        Raises:
            ValidationError: If value is invalid.
        """
        if self.option.validator:
            self.option.validator(self.value)

    def update_gui(self):
        """Updates gui widget with current value.

        Ensures that values' widget value will be up-to-date in the case value
        was changed in scripting module or the other widget.
        """
        if self.widget:
            self.widget.setText(str(self.value))

    def as_widget(self, parent=None):
        """Creates a gui object (widget). This widget must be connected with
        `self.value` (or `self.set_value`).

        Returns:
            A QtGui.QWidget subclass instance
        """
        self.widget = QtGui.QLabel(text=str(self.value))
        return self.widget

    def add_to_layout(self, layout, parent=None):
        """Adds option's widgets to the `layout`.

        Args:
            layout (QtGui.QGridLayout): A grid layout (should have 12 columns).
        """
        row = layout.rowCount()
        layout.addWidget(self.option.label(), row, 0, 1, 6)
        layout.addWidget(self.as_widget(parent=parent), row, 6, 1, 6)

    def __str__(self):
        return str(self.value)


class BaseOption:
    """Base class for options."""
    ValueClass = Value

    def __init__(self, name=None, *, default=None, verbose_name=None, validator=None,
                 help_text=None):
        """Option is a proxy between gui and wrapped object.

        Arguments:
            name (str): An option identifier.
            default: A default option value.
            verbose_name (Optional[str]): Human readable description of the option.
            validator (Optional[Callable]): An option values validator.
        """
        self.name = name
        self.default = default
        self._verbose_name = verbose_name
        self.validator = validator
        self.help_text = help_text

    @property
    def verbose_name(self):
        return self._verbose_name or textify(self.name)

    @verbose_name.setter
    def verbose_name(self, name):
        self._verbose_name = name

    def __call__(self, *args):
        """Creates a `Value` instance.

        Args:
            value: An initial value.
        """
        return self.ValueClass(args[0] if args else self.default, self)

    def label(self):
        """Creates label (QtGui.QLabel) widget with option name.
        """
        label = QtGui.QLabel(self.verbose_name + ':')
        if self.help_text:
            label.setToolTip(self.help_text)
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        return label


class ObjectOption(BaseOption):
    """Option that either has complex type or its value should'n be changed
    within main widget."""


class BoolValue(Value):
    """Boolean value. Creates CheckBox."""
    checkbox = None

    def as_widget(self, parent=None):
        self.checkbox = QtGui.QCheckBox(parent)
        self.checkbox.stateChanged.connect(self.update_value)
        self.update_gui()
        return self.checkbox

    def update_value(self):
        self.value = self.checkbox.isChecked()

    def update_gui(self):
        if self.checkbox and self.checkbox.isChecked() != self.value:
            self.checkbox.setChecked(self.value)


class BoolOption(BaseOption):
    ValueClass = BoolValue

    def __init__(self, name=None, *, default=False, **kwargs):
        super().__init__(name=name, default=default, **kwargs)


class StringValue(Value):
    def as_widget(self, parent=None):
        self.widget = QtGui.QLineEdit(parent)
        self.widget.setText(self.value)
        self.widget.textChanged.connect(self.set_value)
        return self.widget

    def update_gui(self):
        if self.widget and self.widget.text() != self.value:
            self.widget.setText(self.value)


class StringOption(BaseOption):
    ValueClass = StringValue

    def __init__(self, name=None, *, default='', **kwargs):
        super().__init__(name, default=default, **kwargs)


class IntegerValue(Value):
    def as_widget(self, parent=None):
        self.widget = QtGui.QSpinBox(parent)
        self.widget.setRange(*self.option.range)
        self.widget.setSingleStep(self.option.step)
        self.widget.setValue(self.value)
        self.widget.valueChanged.connect(self.set_value)
        return self.widget

    def update_gui(self):
        if self.widget and self.widget.value() != self.value:
            self.widget.setValue(self.value)


class IntegerOption(BaseOption):
    ValueClass = IntegerValue

    def __init__(self, name=None, *, default=0, range=(0, 100), step=1, **kwargs):
        super().__init__(name, default=default, **kwargs)
        self.step = step
        self.range = range


class FloatValue(Value):
    def as_widget(self, parent=None):
        self.widget = QtGui.QDoubleSpinBox(parent)
        self.widget.setDecimals(self.option.decimals)
        self.widget.setRange(*self.option.range)
        self.widget.setSingleStep(self.option.step)
        self.widget.setValue(self.value)
        self.widget.valueChanged.connect(self.set_value)
        return self.widget

    def update_gui(self):
        if self.widget and self.widget.value() != self.value:
            self.widget.setValue(self.value)


class FloatOption(BaseOption):
    ValueClass = FloatValue

    def __init__(self, name=None, *, default=0., range=(0, 1), step=.01,
                 decimals=None, **kwargs):
        super().__init__(name, default=default, **kwargs)
        self.step = step
        self.range = range
        self.decimals = decimals or np.ceil(abs(np.log10(self.step)))
