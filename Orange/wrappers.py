from collections import OrderedDict
from functools import partial

from Orange import options

from PyQt4 import QtGui, QtCore

from Orange.misc import wrapper_meta


class WrapperMeta(wrapper_meta.WrapperMeta):
    """Metaclass for all wrappers."""

    def __new__(mcs, name, bases, attrs, **kwargs):
        return super().__new__(mcs, name, bases, attrs)

    def __init__(cls, name, bases, nmspc):
        super().__init__(name, bases, nmspc)

        def getter(self, option_name):
            return self._values[option_name].value

        def setter(self, value, option_name):
            self._values[option_name].value = value

        cls._options = OrderedDict(cls._options)

        cls.options = list(getattr(cls, 'options', []))
        for key, item in nmspc.items():
            if isinstance(item, options.BaseOption):
                item.name = key
                cls.options.append(item)

        for option in cls.options:
            if not isinstance(option, options.BaseOption):
                raise TypeError("Option '{}' should be a BaseOption instance."
                                .format(option))

            cls._options[option.name] = option
            setattr(cls, option.name, property(
                fget=partial(getter, option_name=option.name),
                fset=partial(setter, option_name=option.name)
            ))

        if 'GUI' not in nmspc:
            cls.GUI = type('GUI', (), {})
        cls.GUI.main_scheme = getattr(cls.GUI, 'main_scheme',
                                      tuple(cls._options.keys()))

    @classmethod
    def __prepare__(mcs, name, bases):
        return OrderedDict()


class BaseWrapper(metaclass=WrapperMeta):
    """Wraps a class and provides simple gui interface.

    Attributes:
        name (str): Name of wrapped object.
        verbose_name (Optional[str]): More readable name.
        instance: An instance of the wrapped class
        options: (List[BaseOption]): A list of all options that will be passed
            to `__wraps__` constructor call.
    """
    __wraps__ = None
    instance = None

    name = ''
    verbose_name = None
    _options = {}

    class GUI:
        main_scheme = None

    def __init__(self, **kwargs):
        self.callback = None

        self._values = {}
        for name, option in self._options.items():
            value = option(kwargs.pop(option.name, option.default))
            value.add_callback(self.on_change)
            self._values[option.name] = value

        if kwargs:
            raise TypeError("Unknown arguments: {}".format(kwargs))

        self.apply_changes()

    @property
    def values(self):
        return self._values

    def __str__(self):
        return self.verbose_name or self.name

    def __repr__(self):
        arguments = ', '.join('{}={}'.format(name, repr(value.value))
                              for name, value in self._values.items())
        return '{}({})'.format(self.__wraps__.__name__, arguments)

    def share_values(self, other):
        """Shares common options (pools equal values)
        Args:
            other (BaseWrapper): wrapper, to share options with
        """
        for name in self.values.keys():
            if name in other.values and \
                    self.values[name].option == other.values[name].option:
                self.values[name] = other.values[name]

    def options_layout(self, parent=None, scheme=None):
        """Creates layout for options configuration.

        Args:
            parent: parent object

        Returns:
            QtGui.QGridLayout: layout with configuration widgets.
        """

        layout = QtGui.QGridLayout(parent)
        for option in (scheme or self.GUI.main_scheme):
            if isinstance(option, str):
                self.values[option].add_to_layout(layout, parent=parent)
            elif isinstance(option, options.BaseOption):
                self.values[option.name].add_to_layout(layout, parent=parent)
            elif isinstance(option, options.OptionGroup):
                option.add_to_layout(layout, self.values, parent=parent)

        return layout

    def on_change(self):
        """Value change callback."""
        if self.callback:
            self.callback()

    def apply_changes(self):
        """Updates `wrapped_object` with new options' values.

        Raises:
            ValidationError: If any of option has an invalid value.
        """
        self.validate()
        if self.__wraps__:
            self.instance = self.__wraps__(**self.state)

    def validate(self):
        for value in self.values.values():
            value.validate()

    @property
    def state(self):
        return {name: v.value for name, v in self.values.items()}


class WrappersMix:
    """Mix multiple wrappers and provides selected one."""

    def __init__(self, wrappers):
        self.wrappers = wrappers
        for i, wrapper in enumerate(self.wrappers):
            for wr in self.wrappers[i+1:]:
                wrapper.share_values(wr)
        self.selected_index = 0
        self._callback = None

        self.button_group = None
        self.wrapper_layout = None

    def options_layout(self, parent=None, orientation=QtCore.Qt.Vertical):
        if orientation == QtCore.Qt.Vertical:
            layout = QtGui.QVBoxLayout(parent)
        else:
            layout = QtGui.QHBoxLayout(parent)

        button_layout = QtGui.QVBoxLayout()
        button_layout.setContentsMargins(10, 0, 0, 0)
        self.button_group = QtGui.QButtonGroup()
        self.button_group.buttonClicked[int].connect(self.index_changed)
        for i, wrapper in enumerate(self.wrappers):
            button = QtGui.QRadioButton(text=wrapper.verbose_name)
            button.setChecked(i == self.selected_index)
            self.button_group.addButton(button, i)
            button_layout.addWidget(button)
        layout.addLayout(button_layout)

        self.wrapper_layout = QtGui.QStackedLayout()
        self.wrapper_layout.setMargin(0)
        for wrapper in self.wrappers:
            box = QtGui.QGroupBox()
            box.setContentsMargins(0, 0, 0, 0)
            wrapper.callback = self.callback
            box.setLayout(wrapper.options_layout())
            self.wrapper_layout.addWidget(box)
        self.wrapper_layout.setCurrentIndex(self.selected_index)
        layout.addLayout(self.wrapper_layout)

        self.selected_index = -1
        self.index_changed(self.button_group.checkedId())
        return layout

    @property
    def checked(self):
        return self.wrappers[self.selected_index]

    def index_changed(self, index):
        if self.selected_index != index:
            self.call_callback()
            self.selected_index = index
            box = QtGui.QGroupBox()
            box.setLayout(self.checked.options_layout())
            self.wrapper_layout.removeWidget(self.wrapper_layout.widget(index))
            self.wrapper_layout.insertWidget(index, box)
            self.wrapper_layout.setCurrentIndex(self.selected_index)

    def call_callback(self):
        if self.callback:
            self.callback()

    @property
    def callback(self):
        return self._callback

    @callback.setter
    def callback(self, callback):
        self._callback = callback
        for wrapper in self.wrappers:
            wrapper.callback = callback
