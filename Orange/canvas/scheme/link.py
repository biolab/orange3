"""
===========
Scheme Link
===========

"""

from PyQt4.QtCore import QObject
from PyQt4.QtCore import pyqtSignal as Signal
from PyQt4.QtCore import pyqtProperty as Property

from ..utils import name_lookup
from .errors import IncompatibleChannelTypeError


def compatible_channels(source_channel, sink_channel):
    """
    Do the channels in link have compatible types, i.e. can they be
    connected based on their type.

    """
    source_type = name_lookup(source_channel.type)
    sink_type = name_lookup(sink_channel.type)
    ret = issubclass(source_type, sink_type)
    if source_channel.dynamic:
        ret = ret or issubclass(sink_type, source_type)
    return ret


def can_connect(source_node, sink_node):
    """
    Return True if any output from `source_node` can be connected to
    any input of `sink_node`.

    """
    return bool(possible_links(source_node, sink_node))


def possible_links(source_node, sink_node):
    """
    Return a list of (OutputSignal, InputSignal) tuples, that
    can connect the two nodes.

    """
    possible = []
    for source in source_node.output_channels():
        for sink in sink_node.input_channels():
            if compatible_channels(source, sink):
                possible.append((source, sink))
    return possible


class SchemeLink(QObject):
    """
    A instantiation of a link between two :class:`.SchemeNode` instances
    in a :class:`.Scheme`.

    Parameters
    ----------
    source_node : :class:`.SchemeNode`
        Source node.
    source_channel : :class:`OutputSignal`
        The source widget's signal.
    sink_node : :class:`.SchemeNode`
        The sink node.
    sink_channel : :class:`InputSignal`
        The sink widget's input signal.
    properties : `dict`
        Additional link properties.

    """

    #: The link enabled state has changed
    enabled_changed = Signal(bool)

    #: The link dynamic enabled state has changed.
    dynamic_enabled_changed = Signal(bool)

    def __init__(self, source_node, source_channel,
                 sink_node, sink_channel, enabled=True, properties=None,
                 parent=None):
        QObject.__init__(self, parent)
        self.source_node = source_node

        if isinstance(source_channel, str):
            source_channel = source_node.output_channel(source_channel)
        elif source_channel not in source_node.output_channels():
            raise ValueError("%r not in in nodes output channels." \
                             % source_channel)

        self.source_channel = source_channel

        self.sink_node = sink_node

        if isinstance(sink_channel, str):
            sink_channel = sink_node.input_channel(sink_channel)
        elif sink_channel not in sink_node.input_channels():
            raise ValueError("%r not in in nodes input channels." \
                             % source_channel)

        self.sink_channel = sink_channel

        if not compatible_channels(source_channel, sink_channel):
            raise IncompatibleChannelTypeError(
                    "Cannot connect %r to %r" \
                    % (source_channel.type, sink_channel.type)
                )

        self.__enabled = enabled
        self.__dynamic_enabled = False
        self.__tool_tip = ""
        self.properties = properties or {}

    def source_type(self):
        """
        Return the type of the source channel.
        """
        return name_lookup(self.source_channel.type)

    def sink_type(self):
        """
        Return the type of the sink channel.
        """
        return name_lookup(self.sink_channel.type)

    def is_dynamic(self):
        """
        Is this link dynamic.
        """
        return self.source_channel.dynamic and \
            issubclass(self.sink_type(), self.source_type()) and \
            not (self.sink_type() is self.source_type())

    def set_enabled(self, enabled):
        """
        Enable/disable the link.
        """
        if self.__enabled != enabled:
            self.__enabled = enabled
            self.enabled_changed.emit(enabled)

    def enabled(self):
        """
        Is this link enabled.
        """
        return self.__enabled

    enabled = Property(bool, fget=enabled, fset=set_enabled)

    def set_dynamic_enabled(self, enabled):
        """
        Enable/disable the dynamic link. Has no effect if the link
        is not dynamic.

        """
        if self.is_dynamic() and self.__dynamic_enabled != enabled:
            self.__dynamic_enabled = enabled
            self.dynamic_enabled_changed.emit(enabled)

    def dynamic_enabled(self):
        """
        Is this a dynamic link and is `dynamic_enabled` set to `True`
        """
        return self.is_dynamic() and self.__dynamic_enabled

    dynamic_enabled = Property(bool, fget=dynamic_enabled,
                               fset=set_dynamic_enabled)

    def set_tool_tip(self, tool_tip):
        """
        Set the link tool tip.
        """
        if self.__tool_tip != tool_tip:
            self.__tool_tip = tool_tip

    def tool_tip(self):
        """
        Link tool tip.
        """
        return self.__tool_tip

    tool_tip = Property(str, fget=tool_tip,
                        fset=set_tool_tip)

    def __str__(self):
        return "{0}(({1}, {2}) -> ({3}, {4}))".format(
                    type(self).__name__,
                    self.source_node.title,
                    self.source_channel.name,
                    self.sink_node.title,
                    self.sink_channel.name
                )
