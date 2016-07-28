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

from operator import attrgetter
from warnings import warn

from PyQt4.QtGui import QApplication, QStyle, QSizePolicy

from Orange.widgets import gui


class UnboundMsg(str):
    """
    The class used for declaring messages in classes derived from
    MessageGroup. When instantiating the message group, instances of this
    class are replaced by instances of `_BoundMsg` that are bound to the group.

    Note: this class is aliased to `Orange.widgets.widget.Msg`.
    """
    def __new__(cls, msg):
        return str.__new__(cls, msg)

    def bind(self, group):
        return _BoundMsg(self, group)

    # The method is implemented in _BoundMsg
    # pylint: disable=unused-variable
    def __call__(self, *args, shown=True, **kwargs):
        """
        Show the message, or hide it if `show` is set `False`
        `*args` and `**kwargs` are passed to the `format` method.

        Args:
            shown (bool): keyword-only argument that can be set to `False` to
                hide the message
            *args: arguments for `format`
            **kwargs: keyword arguments for `format`
        """
        raise RuntimeError("Message is not bound")

    # The method is implemented in _BoundMsg
    def clear(self):
        """Remove the message."""
        raise RuntimeError("Message is not bound")

    # The method is implemented in _BoundMsg
    def is_shown(self):
        """Return True if message is currently displayed."""
        raise RuntimeError("Message is not bound")

    # Ensure that two instance of message are always different
    # In particular, there may be multiple messages "{}".
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class _BoundMsg(UnboundMsg):
    """
    A message that is bound to the group.

    Instances of this class provide the call operator for raising the message,
    and method `clear` for removing it.

    When the message is called, the arguments are passed to the message's
    format` method and the resulting string is stored in the attribute
    `formatted`.

    Attributes:
        group (MessageGroup): the group to which this message belongs
        formatted (str): formatted message

    """
    def __new__(cls, unbound_msg, group):
        self = UnboundMsg.__new__(cls, unbound_msg)
        self.group = group
        self.formatted = ""
        return self

    def __call__(self, *args, shown=True, **kwargs):
        if not shown:
            self.clear()
        else:
            self.formatted = self.format(*args, **kwargs)
            self.group.activate_msg(self)

    def clear(self):
        self.group.deactivate_msg(self)

    def is_shown(self):
        return self in self.group.active

    def __str__(self):
        return self.formatted


class _OldStyleMsg(_BoundMsg):
    """
    Class for handling the old-style messages.

    Instances of this class are instantiated by the old methods `error`,
    `warning` and `information` and added to the list of active messages,
    with their old-style id's as keys. Instance of `_OldStyleMsg` are not
    members of message groups.
    """
    def __new__(cls, text, group):
        self = _BoundMsg.__new__(cls, text, group)
        self.formatted = text
        return self


class MessageGroup:
    """
    A groups of messages, e.g. errors, warnings, information messages

    Widget's `__init__` searches for instances of this class among the widget's
    class member and instantiates them.

    Attributes:
        widget (widget.OWWidget): the widget instance to which the group belongs
        active (dict): active messages

    Note: active messages are stored in the dictionary, in which the key and
    the corresponding value are one and the same object, except for old-style
    classes, for which the key is an (old-style) id. When we remove support for
    old-style messages, this dictionary can be replaced with a set.
    """
    def __init__(self, widget):
        self.widget = widget
        self.active = {}
        self._general = UnboundMsg("{}")
        self._bind_messages()

    def _bind_messages(self):
        # type(self).__dict__ wouldn't return inherited messages, hence dir
        for name in dir(self):
            msg = getattr(self, name)
            if isinstance(msg, UnboundMsg):
                msg = msg.bind(self)
                self.__dict__[name] = msg

    def add_message(self, name, msg="{}"):
        """Add and bind message to a group that is already instantiated
        and bound.

        If the message with that name already exists, the method does nothing.
        The method is used by helpers like this (simplified) one::

            def check_results(results, msg_group):
                msg_group.add_message("invalid_results",
                                      "Results do not include any data")
                msg_group.invalid_results.clear()
                if results.data is None:
                    msg_group.invalid_results()

        The helper is called from several widgets with
        `check_results(results, self.Error)`

        Args:
            name (str): the name of the member with the message
            msg (str or UnboundMsg): message text or instance (default `"{}"`)
        """
        if not isinstance(msg, UnboundMsg):
            msg = UnboundMsg(msg)
        if name not in self.__dict__:
            self.__dict__[name] = msg.bind(self)

    def activate_msg(self, msg, msg_id=None):
        """Activate a message and emit the signal messageActivated

        Args:
            msg (_BoundMsg): the message to activate
            msg_id (int): id for old-style message (to be removed in the future)
        """
        key = msg if msg_id is None else msg_id
        if self.active.get(key) == msg:
            return
        self.active[key] = msg
        self.widget.messageActivated.emit(msg)

    def deactivate_msg(self, msg):
        """Deactivate a message and emit the signal messageDeactivated.

        Args:
            msg (_BoundMsg): the message to deactivate
        """
        if msg not in self.active:
            return
        inst_msg = self.active.pop(msg)
        self.widget.messageDeactivated.emit(
            inst_msg if isinstance(msg, int) else msg)

        # When when we no longer support old-style messages, replace with:
        # if msg not in self.active:
        #     return
        # del self.active[msg]
        # self.widget.messageDeactivated.emit(msg)

    # self has default value to avoid PyCharm warnings when calling
    # self.Error.clear(): PyCharm doesn't know that Error is instantiated
    def clear(self=None):
        """Deactivate all active message from this group."""
        for msg in list(self.active):
            self.deactivate_msg(msg)

    def _add_general(self, id_or_text, text, shown):
        """Handler for methods `error`, `warning` and `information`;
        do not call directly.

        The message is shown as general message. This method also
        handles deprecated messages with id's."""
        if id_or_text is None or id_or_text == "":
            self._general.clear()
        elif isinstance(id_or_text, str):
            self._general(id_or_text, shown=shown)
        # remaining cases handle deprecated messages with id's
        elif text:
            self.activate_msg(_OldStyleMsg(text, self), id_or_text)
        elif isinstance(id_or_text, list):
            for msg_id in id_or_text:
                self.deactivate_msg(msg_id)
        else:
            self.deactivate_msg(id_or_text)


class MessagesMixin:
    """
    Base class for message mixins. The class provides a constructor for
    instantiating and binding message groups.

    Widgets should use `WidgetMessageMixin rather than this class.
    """
    def __init__(self):
        # type(self).__dict__ wouldn't return inherited messages, hence dir
        self.message_groups = []
        for name in dir(self):
            group_class = getattr(self, name)
            if isinstance(group_class, type) and \
                    issubclass(group_class, MessageGroup) and \
                    group_class is not MessageGroup:
                bound_group = group_class(self)
                setattr(self, name, bound_group)
                self.message_groups.append(bound_group)
        self.message_groups.sort(key=attrgetter("severity"), reverse=True)


class WidgetMessagesMixin(MessagesMixin):
    """
    Provide the necessary methods for handling messages in widgets.

    The class defines member classes `Error`, `Warning` and `Information` that
    serve as base classes for these message groups.
    """
    class Error(MessageGroup):
        """Base class for groups of error messages in widgets"""
        severity = 3
        icon_path = gui.resource_filename("icons/error.png")
        bar_background = "#ffc6c6"
        bar_icon = QStyle.SP_MessageBoxCritical

    class Warning(MessageGroup):
        """Base class for groups of warning messages in widgets"""
        severity = 2
        icon_path = gui.resource_filename("icons/warning.png")
        bar_background = "#ffffc9"
        bar_icon = QStyle.SP_MessageBoxWarning

    class Information(MessageGroup):
        """Base class for groups of information messages in widgets"""
        severity = 1
        icon_path = gui.resource_filename("icons/information.png")
        bar_background = "#ceceff"
        bar_icon = QStyle.SP_MessageBoxInformation

    def __init__(self):
        super().__init__()
        self.message_bar = self.message_label = self.message_icon = None
        self.messageActivated.connect(self.update_message_state)
        self.messageDeactivated.connect(self.update_message_state)

    def clear_messages(self):
        """Clear all messages"""
        for group in self.message_groups:
            group.clear()

    def update_message_state(self):
        """Show and update (or hide) the content of the widget's message bar.

        The method is connected to widget's signals `messageActivated` and
        `messageDeactivated`.
        """
        messages = [msg
                    for group in self.message_groups
                    for msg in group.active.values()]
        if not messages:
            self._hide_message_bar()
            return
        else:
            group = messages[0].group
            text = str(messages[0]) if len(messages) == 1 \
                else "{} problems during execution".format(len(messages))
            # TODO: fix tooltip background color - it is not white
            tooltip = '<div style="background-color: #ffffff;">{}</div>'.format(
                ''.join(
                    '<p style="background-color: {}; padding: 5">'
                    '<nobr>{}</nobr></p>'.format(
                        msg.group.bar_background, str(msg))
                    for msg in messages))
            self._set_message_bar(group, text, tooltip)

    def insert_message_bar(self):
        """Insert message bar into the widget.

        This method must be called at the appropriate place in the widget
        layout setup by any widget that is using this mixin."""
        self.message_bar = gui.hBox(self, spacing=0)
        self.message_icon = gui.widgetLabel(self.message_bar, "")
        self.message_label = gui.widgetLabel(self.message_bar, "")
        self.message_label.setStyleSheet("padding-top: 5px")
        self.message_bar.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        gui.rubber(self.message_bar)
        self.message_bar.setVisible(False)

    def _check_has_message_bar(self):
        if self.message_bar is None:
            raise RuntimeError(
                "Did you forget to call WidgetMessagesMixin.insert_message_bar "
                "for {} (or one of its parent classes)?".
                format(type(self).__name__))

    def _hide_message_bar(self):
        self._check_has_message_bar()
        if not self.message_bar.isHidden():
            new_height = self.height() - self.message_bar.height()
            self.message_bar.setVisible(False)
            self.resize(self.width(), new_height)

    def _set_message_bar(self, group, text=None, tooltip=None):
        self._check_has_message_bar()
        current_height = self.height()
        style = QApplication.instance().style()
        self.message_icon.setPixmap(
            style.standardIcon(group.bar_icon).pixmap(14, 14))
        self.message_bar.setStyleSheet(
            "background-color: {}; color: black;"
            "padding: 3px; padding-left: 6px; vertical-align: center".
            format(group.bar_background))
        self.message_label.setText(text)
        self.message_bar.setToolTip(tooltip)
        if self.message_bar.isHidden():
            self.message_bar.setVisible(True)
            new_height = current_height + self.message_bar.height()
            self.resize(self.width(), new_height)

    # pylint doesn't know that Information, Error and Warning are instantiated
    # and thus the methods are bound
    # pylint: disable=no-value-for-parameter
    # This class and classes Information, Error and Warning are friends
    # pylint: disable=protected-access
    @staticmethod
    def _warn_obsolete(text_or_id, what):
        if not isinstance(text_or_id, str) and text_or_id is not None:
            warn("'{}' with id is deprecated; use {} class".
                 format(what, what.title()), stacklevel=3)

    def information(self, text_or_id=None, text=None, shown=True):
        self._warn_obsolete(text_or_id, "information")
        self.Information._add_general(text_or_id, text, shown)

    def warning(self, text_or_id=None, text=None, shown=True):
        self._warn_obsolete(text_or_id, "warning")
        self.Warning._add_general(text_or_id, text, shown)

    def error(self, text_or_id=None, text=None, shown=True):
        self._warn_obsolete(text_or_id, "error")
        self.Error._add_general(text_or_id, text, shown)
