import copy
import warnings

from Orange.util import OrangeDeprecationWarning


class SignalSpecificationError(Exception):
    pass


# Input/output signal (channel) description

#: Input/Output flags.
#: -------------------
#:
#: The input/output is the default for its type.
#: When there are multiple IO signals with the same type the
#: one with the default flag takes precedence when adding a new
#: link in the canvas.
Default = 8
NonDefault = 16
#: Single input signal (default)
Single = 2
#: Multiple outputs can be linked to this signal.
#: Signal handlers with this flag have (object, id: object) -> None signature.
Multiple = 4
#: Applies to user interaction only.
#: Only connected if specifically requested (in a dedicated "Links" dialog)
#: or it is the only possible connection.
Explicit = 32
#: Dynamic output type (obsolete and ignored).
#: Specifies that the instances on the output will in general be
#: subtypes of the declared type and that the output can be connected
#: to any input signal which can accept a subtype of the declared output
#: type.
Dynamic = 64
#: Only applies to output. Specifies that this output can be connected only
#: to inputs that expect this type of its supertypes. Otherwise, the output
#: can be connected to any input which can accept a subtype of the
#: declared output type.
Strict = 128


class _Signal:
    """
    Description of a signal.

    Parameters
    ----------
    name : str
        Name of the channel.
    type : str or `type`
        Type of the accepted signals.
    flags : int, optional
        Channel flags.
    id : str
        A unique id of the input signal.
    doc : str, optional
        A docstring documenting the channel.
    replaces : List[str]
        A list of names this input replaces.
    """

    def __init__(self, name, type, flags=Single + NonDefault,
                 id=None, doc=None, replaces=[]):
        self.name = name
        self.type = type
        self.id = id
        self.doc = doc
        self.replaces = list(replaces)
        self.widget = None

        if not (flags & Single or flags & Multiple):
            flags += Single
        if not (flags & Default or flags & NonDefault):
            flags += NonDefault

        self.single = flags & Single
        self.default = flags & Default
        self.explicit = flags & Explicit
        self.flags = flags


class Input(_Signal):
    """
    Input has additional argument `handler` for providing a handler in
    old-style signal declarations.
    """
    def __init__(self, name, type, handler="", flags=Single + NonDefault,
                 id=None, doc=None, replaces=[]):
        super().__init__(name, type, flags, id, doc, replaces)
        self.handler = handler

    def __str__(self):
        return "{}(name='{}', type={}, handler={})".format(
            type(self).__name__, self.name, self.type, self.handler)

    __repr__ = __str__

    def __call__(self, method):
        """
        The call implements decorator-like behaviour. The decorator stores
        the decorated method's name in the signal's `handler` attribute.
        The method is returned unchanged.
        """
        if self.handler:
            raise ValueError("Input {} is already bound to method {}".
                             format(self.name, self.handler))
        self.handler = method.__name__
        return method


class Output(_Signal):
    def __init__(self, name, type, flags=Single + NonDefault,
                 id=None, doc=None, replaces=[]):
        super().__init__(name, type, flags, id, doc, replaces)
        self.widget = None
        self.dynamic = not (flags & Strict)

        if self.dynamic and not self.single:
            raise SignalSpecificationError(
                "Output signal can not be 'Multiple' and 'Dynamic'.")
        if flags & Dynamic:
            warnings.warn("all outputs are dynamic; flag Dynamic is deprecated",
                          OrangeDeprecationWarning)

    def __str__(self):
        is_bound = "bound" if self.widget else "unbound"
        return "{} {}(name='{}', type={})".format(
            is_bound, type(self).__name__, self.name, self.type)

    __repr__ = __str__

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

    def send(self, signal_name, value, id=None):
        """
        Send a `value` on the `signalName` widget output.

        An output with `signalName` must be defined in the class ``outputs``
        list.
        """
        signal = getattr(self.Outputs, signal_name, None)
        if signal is None:
            raise ValueError('{} is not a valid output signal for widget {}'.
                             format(signal_name, self.name))
        signal.send(value, id)

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
    def patch_signals(cls):
        cls._migrate_signals(Input)
        cls._check_input_handlers()
        cls._migrate_signals(Output)

    @classmethod
    def _migrate_signals(cls, signal_type):
        """
        Copy signals from `inputs`/`outputs` attribute to group.
        Then, remove the attribute to unshadow the property.

        Args:
            signal_type (type): `Input` or `Output`
        """
        attr_name = signal_type.__name__ + 's'
        signals = cls.__dict__.get(attr_name.lower())
        if isinstance(signals, (list, tuple)):
            group = type(attr_name, (), {})
            setattr(cls, attr_name, group)
            for signal in signals:
                if isinstance(signal, tuple):
                    signal = signal_type(*signal)
                setattr(group, signal.name, signal)
            #delattr(cls, attr_name.lower())

    @classmethod
    def _check_input_handlers(cls):
        unbound = [signal.name for signal in cls.inputs
                   if not signal.handler]
        if unbound:
            raise ValueError("unbound signal(s) in {}: {}".
                             format(cls.__name__, ", ".join(unbound)))


class AttributeList(list):
    """Signal type for lists of attributes (variables)"""

# Temporary compatibility fix
InputSignal = Input
OutputSignal = Output
