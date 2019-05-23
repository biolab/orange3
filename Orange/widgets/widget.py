from orangewidget.gui import OWComponent
from orangewidget.utils.signals import (
    WidgetSignalsMixin,
    InputSignal, OutputSignal,
    Default, NonDefault, Single, Multiple, Dynamic, Explicit
)
from orangewidget.widget import (
    OWBaseWidget, Message, Msg, StateInfo, Input, Output,
)
from Orange.widgets.utils.signals import AttributeList

__all__ = [
    "OWWidget", "Input", "Output", "AttributeList", "Message", "Msg",
    "StateInfo",

    # these are re-exported here for legacy reasons. Use Input/Output instead.
    "InputSignal", "OutputSignal",
    "Default", "NonDefault", "Single", "Multiple", "Dynamic", "Explicit"
]

# these are included in this namespace but are not in __all__
OWComponent = OWComponent
WidgetSignalsMixin = WidgetSignalsMixin

WidgetMetaClass = type(OWBaseWidget)

#: :class:`~OWBaseWidget` re-exposed in this namespace. Orange Widgets
#: should import  This from this module.
OWWidget = OWBaseWidget

#: Input/Output flags (Deprecated).
#: --------------------------------
#:
#: The input/output is the default for its type.
#: When there are multiple IO signals with the same type the
#: one with the default flag takes precedence when adding a new
#: link in the canvas.
Default = Default
NonDefault = NonDefault
#: Single input signal (default)
Single = Single
#: Multiple outputs can be linked to this signal.
#: Signal handlers with this flag have (object, id: object) -> None signature.
Multiple = Multiple
#: Applies to user interaction only.
#: Only connected if specifically requested (in a dedicated "Links" dialog)
#: or it is the only possible connection.
Explicit = Explicit
#: Dynamic output type.
#: Specifies that the instances on the output will in general be
#: subtypes of the declared type and that the output can be connected
#: to any input signal which can accept a subtype of the declared output
#: type.
Dynamic = Dynamic
