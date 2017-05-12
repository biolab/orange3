import copy

from Orange.canvas.registry.description import \
    InputSignal, OutputSignal, \
    Single, Multiple, Default, NonDefault, Explicit, Dynamic


class Input(InputSignal):
    def __init__(self, name, type, flags=None, id=None, doc=None, replaces=[],
                 single=True, multiple=False, default=False, explicit=False,
                 dynamic=True):
        if flags is None:
            flags = (single and Single) | \
                    (multiple and Multiple) | \
                    (default and Default or NonDefault) | \
                    (explicit and Explicit) | \
                    (dynamic and Dynamic)
        super().__init__(name, type, "", flags, id, doc, replaces)

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


class Output(OutputSignal):
    def __init__(self, name, type, flags=None, id=None, doc=None, replaces=[],
                 single=True, multiple=False, default=False, explicit=False,
                 dynamic=True):
        if flags is None:
            flags = (single and Single) | \
                    (multiple and Multiple) | \
                    (default and Default or NonDefault) | \
                    (explicit and Explicit) | \
                    (dynamic and not multiple and Dynamic)
        super().__init__(name, type, flags, id, doc, replaces)

    def bound_signal(self, widget):
        """Return a copy of the signal bound to a widget."""
        new_signal = copy.copy(self)
        new_signal.widget = widget
        return new_signal

    def send(self, value, id=None):
        assert self.widget is not None
        signal_manager = self.widget.signalManager
        if signal_manager is not None:
            signal_manager.send(self.widget, self.name, value, id)


class WidgetSignalsMixin:
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
        unbound = [signal.name for signal in cls.inputs
                   if not signal.handler]
        if unbound:
            raise ValueError("unbound signal(s) in {}: {}".
                             format(cls.__name__, ", ".join(unbound)))

    @classmethod
    def get_signals(cls, direction):
        old_style = cls.__dict__.get(direction, None)
        if old_style:
            return [copy.copy(signal) for signal in old_style]
        signal_class = getattr(cls, direction.title())
        return [signal for signal in signal_class.__dict__.values()
                if isinstance(signal, (InputSignal, OutputSignal))]


class AttributeList(list):
    """Signal type for lists of attributes (variables)"""
