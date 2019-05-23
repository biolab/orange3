"""Mixin class for errors, warnings and information

A class derived from `OWWidget` can include member classes `Error`, `Warning`
and `Information`, derived from the same-named `OWWidget` classes. Each of
those contains members that are instances of `UnboundMsg`, which is, for
convenience, also exposed as `Orange.widgets.widget.Msg`. These members
represent all possible errors, like `Error.no_discrete_vars`, with exception
of the deprecated old-style errors.

When the widget is instantiated, classes `Error`, `Warning` and `Information`
are instantiated and bound to the widget: their attribute `widget` is the link
to the widget that instantiated them. Their member messages are replaced with
instances of `_BoundMsg`, which are bound to the group through the `group`
attribute.

A message is shown by calling, e.g. `self.Error.no_discrete_vars()`. The call
formats the message and tells the group to activate it::

    self.formatted = self.format(*args, **kwargs)
    self.group.activate_msg(self)

The group adds it to the dictionary of active messages (attribute `active`)
and emits the signal `messageActivated`. The signal is connected to the
widget's method `update_widget_state`, which shows the message in the bar, and
`WidgetManager`'s `__on_widget_state_changed`, which manages the icons on the
canvas.

Clearing messages work analogously.
"""
from orangewidget.utils.messages import (
    UnboundMsg, MessageGroup, MessagesMixin, WidgetMessagesMixin
)
__all__ = [
    "UnboundMsg", "MessageGroup", "MessagesMixin", "WidgetMessagesMixin"
]
