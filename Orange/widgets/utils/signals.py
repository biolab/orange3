import copy

from Orange.canvas.registry.description import InputSignal, OutputSignal


class _Signal:
    @staticmethod
    def get_flags(multiple, default, explicit, dynamic):
        """Compute flags from arguments"""
        from Orange.canvas.registry.description import \
            Single, Multiple, Default, NonDefault, Explicit, Dynamic
        return (Multiple if multiple else Single) | \
                (Default if default else NonDefault) | \
                (explicit and Explicit) | \
                (dynamic and Dynamic)


class Input(InputSignal, _Signal):
    """
    Description of an input signal.

    The class is used to declare input signals for a widget as follows
    (the example is taken from the widget Test & Score)::

        class Inputs:
            train_data = Input("Data", Table, default=True)
            test_data = Input("Test Data", Table)
            learner = Input("Learner", Learner, multiple=True)
            preprocessor = Input("Preprocessor", Preprocess)

    Every input signal must be used to decorate exactly one method that
    serves as the input handler, for instance::

        @Inputs.train_data
        def set_train_data(self, data):
            ...

    Parameters
    ----------
    name (str): signal name
    type (type): signal type
    id (str): a unique id of the signal
    doc (str, optional): signal documentation
    replaces (list of str): a list with names of signals replaced by this signal
    multiple (bool, optional): if set, multiple signals can be connected
        to this output (default: `False`)
    default (bool, optional): when the widget accepts multiple signals of the
        same type, one of them can set this flag to act as the default
        (default: `False`)
    explicit (bool, optional): if set, this signal is only used when it is
        the only option or when explicitly connected in the dialog
        (default: `False`)
    """
    def __init__(self, name, type, id=None, doc=None, replaces=None, *,
                 multiple=False, default=False, explicit=False):
        flags = self.get_flags(multiple, default, explicit, False)
        super().__init__(name, type, "", flags, id, doc, replaces or [])

    def __call__(self, method):
        """
        Decorator that stores decorated method's name in the signal's
        `handler` attribute. The method is returned unchanged.
        """
        if self.handler:
            raise ValueError("Input {} is already bound to method {}".
                             format(self.name, self.handler))
        self.handler = method.__name__
        return method


class Output(OutputSignal, _Signal):
    """
    Description of an output signal.

    The class is used to declare output signals for a widget as follows
    (the example is taken from the widget Test & Score)::

        class Outputs:
            predictions = Output("Predictions", Table)
            evaluations_results = Output("Evaluation Results", Results)

    The signal is then transmitted by, for instance::

        self.Outputs.predictions.send(predictions)

    Parameters
    ----------
    name (str): signal name
    type (type): signal type
    id (str): a unique id of the signal
    doc (str, optional): signal documentation
    replaces (list of str): a list with names of signals replaced by this signal
    default (bool, optional): when the widget outputs multiple signals of the
        same type, one of them can set this flag to act as the default
        (default: `False`)
    explicit (bool, optional): if set, this signal is only used when it is
        the only option or when explicitly connected in the dialog
        (default: `False`)
    dynamic (bool, optional): Specifies that the instances on the output will
        in general be subtypes of the declared type and that the output can
        be connected to any input signal which can accept a subtype of the
        declared output type.
    """
    def __init__(self, name, type, id=None, doc=None, replaces=None, *,
                 default=False, explicit=False, dynamic=True):
        flags = self.get_flags(False, default, explicit, dynamic)
        super().__init__(name, type, flags, id, doc, replaces or [])
        self.widget = None

    def bound_signal(self, widget):
        """
        Return a copy of the signal bound to a widget.

        Called from `WidgetSignalsMixin.__init__`
        """
        new_signal = copy.copy(self)
        new_signal.widget = widget
        return new_signal

    def send(self, value, id=None):
        """Emit the signal through signal manager."""
        assert self.widget is not None
        signal_manager = self.widget.signalManager
        if signal_manager is not None:
            signal_manager.send(self.widget, self.name, value, id)


class WidgetSignalsMixin:
    """Mixin for managing widget's input and output signals"""
    class Inputs:
        pass

    class Outputs:
        pass

    def __init__(self):
        self._bind_outputs()

    def _bind_outputs(self):
        bound_cls = self.Outputs()
        for name, signal in self.Outputs.__dict__.items():
            if isinstance(signal, Output):
                bound_cls.__dict__[name] = signal.bound_signal(self)
        setattr(self, "Outputs", bound_cls)

    def send(self, signalName, value, id=None):
        """
        Send a `value` on the `signalName` widget output.

        An output with `signalName` must be defined in the class ``outputs``
        list.
        """
        if not any(s.name == signalName for s in self.outputs):
            raise ValueError('{} is not a valid output signal for widget {}'.format(
                signalName, self.name))
        if self.signalManager is not None:
            self.signalManager.send(self, signalName, value, id)

    def handleNewSignals(self):
        """
        Invoked by the workflow signal propagation manager after all
        signals handlers have been called.
        Reimplement this method in order to coalesce updates from
        multiple updated inputs.
        """
        pass

    # Methods used by the meta class
    @classmethod
    def convert_signals(cls):
        """
        Convert tuple descriptions into old-style signals for backwards
        compatibility, and check the input handlers exist.
        The method is called from the meta-class.
        """
        def signal_from_args(args, signal_type):
            if isinstance(args, tuple):
                return signal_type(*args)
            elif isinstance(args, signal_type):
                return copy.copy(args)

        if hasattr(cls, "inputs") and cls.inputs:
            cls.inputs = [signal_from_args(input_, InputSignal)
                          for input_ in cls.inputs]
        if hasattr(cls, "outputs") and cls.outputs:
            cls.outputs = [signal_from_args(output, OutputSignal)
                           for output in cls.outputs]

        cls._check_input_handlers()

    @classmethod
    def _check_input_handlers(cls):
        unbound = [signal.name for signal in cls.Inputs.__dict__.values()
                   if isinstance(signal, Input) and not signal.handler]
        if unbound:
            raise ValueError("unbound signal(s) in {}: {}".
                             format(cls.__name__, ", ".join(unbound)))

        missing_handlers = [signal.handler for signal in cls.inputs
                            if not hasattr(cls, signal.handler)]
        if missing_handlers:
            raise ValueError("missing handlers in {}: {}".
                             format(cls.__name__, ", ".join(missing_handlers)))

    @classmethod
    def get_signals(cls, direction):
        """
        Return a list of `InputSignal` or `OutputSignal` needed for the
        widget description. For old-style signals, the method returns the
        original list. New-style signals are collected into a list.

        Parameters
        ----------
        direction (str): `"inputs"` or `"outputs"`

        Returns
        -------
        list of `InputSignal` or `OutputSignal`
        """
        old_style = cls.__dict__.get(direction, None)
        if old_style:
            return [copy.copy(signal) for signal in old_style]

        signal_class = getattr(cls, direction.title())
        return [signal for signal in signal_class.__dict__.values()
                if isinstance(signal, _Signal)]


class AttributeList(list):
    """Signal type for lists of attributes (variables)"""
