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


class RatioValue(FloatValue):
    slider = None

    def as_widget(self, parent=None):
        self.widget = QtGui.QGroupBox(parent)
        self.widget.setContentsMargins(0, 0, 0, 0)

        layout = QtGui.QHBoxLayout(self.widget)
        layout.setMargin(0)
        self.widget.setLayout(layout)

        if self.option.left_label:
            layout.addWidget(QtGui.QLabel(self.option.left_label))

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self.widget)
        self.slider.setSingleStep(5)
        # self.slider.setSingleStep(self.option.step)
        self.slider.setTickInterval(100)
        self.slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.slider.valueChanged.connect(lambda x: self.set_value(x * .01))
        layout.addWidget(self.slider)

        if self.option.right_label:
            layout.addWidget(QtGui.QLabel(self.option.right_label))
        self.update_gui()
        return self.widget

    def update_gui(self):
        if self.widget and self.value != self.slider.value() / 100:
            self.slider.setValue(self.value * 100)


class RatioOption(BaseOption):
    ValueClass = RatioValue

    def __init__(self, name=None, *, default=.5, step=.01,
                 left_label='', right_label='', **kwargs):
        super().__init__(name, default=default, **kwargs)
        self.step = step
        self.left_label = left_label
        self.right_label = right_label


class ChoiceValue(Value):
    @property
    def index(self):
        return self.option.choice_values.index(self.value)

    def as_widget(self, parent=None):
        self.widget = QtGui.QComboBox(parent)
        self.widget.addItems(self.option.choice_names)
        self.widget.setCurrentIndex(self.index)
        self.widget.currentIndexChanged.connect(self.update_value)
        return self.widget

    def as_buttons(self):
        layout = QtGui.QVBoxLayout()
        button_group = QtGui.QButtonGroup()
        button_group.buttonClicked[int].connect(
            lambda i: self.update_value(i)
        )

        for i, choice in enumerate(self.option.choices):
            button = QtGui.QRadioButton(text=str(choice))
            button.setChecked(choice.value == self.value)
            button_group.addButton(button, i)
            layout.addWidget(button)

        return button_group, layout

    def update_value(self, index):
        self.value = self.option.choice_values[index]

    def update_gui(self):
        if self.widget and self.widget.currentIndex() != self.index:
            self.widget.setCurrentIndex(self.index)


class Choice:
    """Choice with addition options."""
    def __init__(self, value, verbose_name=None, related_options=None, label=None):
        self.value = value
        self.verbose_name = verbose_name or textify(value)
        self.related_options = related_options or ()
        self.label = label

    def __str__(self):
        return self.verbose_name


class ChoiceOption(BaseOption):
    ValueClass = ChoiceValue

    def __init__(self, name=None, *, choices, default=None, **kwargs):
        self.choices = choices

        if isinstance(choices[0], Choice):
            self.choice_names = [str(c) for c in choices]
            self.choice_values = [c.value for c in choices]
        elif isinstance(choices[0], tuple):
            self.choice_names = [c[1] for c in choices]
            self.choice_values = [c[0] for c in choices]
        else:
            self.choice_names = [textify(c) for c in choices]
            self.choice_values = choices

        if default is None:
            default = self.choice_values[0]

        super().__init__(name, default=default, **kwargs)


class DisableableValue(Value):
    """Value that can be disabled (for instance, has `None` value)
    or enabled (has value from another one."""
    disable_check_box = None
    stacked_layout = None

    def __init__(self, value, option):
        super().__init__(value, option)
        if value == self.option.disable_value:
            self.enabled = False
            self.sub_value = option.sub_option(option.sub_option.default)
        else:
            self.enabled = True
            self.sub_value = option.sub_option(value)

        self.sub_value.add_callback(self.update_value)

    def update_value(self):
        if self.enabled:
            self.value = self.sub_value.value
        else:
            self.value = self.option.disable_value

    def check_state(self):
        if self.enabled != self.disable_check_box.isChecked():
            self.enabled = self.disable_check_box.isChecked()
            self.update_value()

    def update_gui(self):
        if self.disable_check_box:
            self.disable_check_box.setChecked(self.enabled)
        if self.stacked_layout:
            self.stacked_layout.setCurrentIndex(int(self.enabled))

    def as_widget(self, parent=None):
        self.widget = QtGui.QGroupBox(parent)

        layout = QtGui.QHBoxLayout(self.widget)
        layout.setMargin(0)
        self.widget.setContentsMargins(0, 0, 0, 0)
        self.disable_check_box = QtGui.QCheckBox()
        self.disable_check_box.stateChanged.connect(self.check_state)
        layout.addWidget(self.disable_check_box)

        self.stacked_layout = QtGui.QStackedLayout()
        self.stacked_layout.addWidget(QtGui.QLabel(text=self.option.disable_label))
        self.stacked_layout.addWidget(self.sub_value.as_widget(parent=parent))

        layout.addLayout(self.stacked_layout)
        self.widget.setLayout(layout)
        self.update_gui()
        return self.widget


class DisableableOption(BaseOption):
    """Option that holds another one and a constant value."""
    ValueClass = DisableableValue

    def __init__(self, name=None, *, option, disable_value=None, disable_label=None, **kwargs):
        """
        Args:
            option (BaseOption): main option.
            disable_value: value in disabled state
            disable_label (Optional[str]): text to be shown in disabled state
        """
        super().__init__(name, **kwargs)
        self.sub_option = option
        self._verbose_name = self._verbose_name or option._verbose_name
        self.default = disable_value
        self.validator = None
        self.disable_value = disable_value
        self.disable_label = disable_label or textify(disable_value)
        self.help_text = option.help_text


class OptionGroup:
    def __init__(self, name, options):
        self.name = name
        self.options = options

    def add_to_layout(self, layout, values, parent=None):
        title = self.name + ':' if self.name else ''
        box = QtGui.QGroupBox(title=title)
        sub_layout = QtGui.QGridLayout()
        for option in self.options:
            values[option].add_to_layout(sub_layout, parent=parent)
        box.setLayout(sub_layout)
        layout.addWidget(box, layout.rowCount(), 0, 1, 12)


class ChoiceGroup(OptionGroup):
    def add_to_layout(self, layout, values, parent=None):
        choice_value = values[self.name]

        box = QtGui.QGroupBox(title=choice_value.option.verbose_name)
        sub_layout = QtGui.QGridLayout()
        box.setLayout(sub_layout)
        layout.addWidget(box, layout.rowCount(), 0, 1, 12)

        button_group, buttons_layout = choice_value.as_buttons()
        button_group.buttonClicked[int].connect(
            lambda i: self.set_value(i, choice_value)
        )
        choice_value.checked_id = button_group.checkedId()
        sub_layout.addLayout(buttons_layout, 0, 0,
                             len(choice_value.option.choices), 6)

        info_label_layout = QtGui.QStackedLayout()
        sub_layout.addLayout(info_label_layout, 0, 6, 1, 6)
        for choice in choice_value.option.choices:
            info_label_layout.addWidget(QtGui.QLabel(text=choice.label))
        info_label_layout.setCurrentIndex(choice_value.checked_id)

        choice_value.info_label_layout = info_label_layout
        choice_value.button_group = button_group

        choice_value.option_widgets = []
        for i, option in enumerate(self.options):
            value = values[option]
            sub_layout.addWidget(value.option.label(), i + 1, 6, 1, 3)
            widget = value.as_widget(parent=parent)
            sub_layout.addWidget(widget, i + 1, 9, 1, 3)
            choice_value.option_widgets.append(widget)

        choice_value.checked_id = -1
        self.set_value(button_group.checkedId(), choice_value)

    def set_value(self, index, value):
        if value.checked_id != index:
            value.checked_id = value.button_group.checkedId()
            value.info_label_layout.setCurrentIndex(value.checked_id)

            checked = value.option.choices[value.checked_id]
            for i, opt in enumerate(self.options):
                value.option_widgets[i].setEnabled(opt in checked.related_options)
