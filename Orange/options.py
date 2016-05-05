"""This module defines various types of options and their values.
"""
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
