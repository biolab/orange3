"""
===========
Scheme Node
===========

"""

from PyQt4.QtCore import QObject
from PyQt4.QtCore import pyqtSignal as Signal
from PyQt4.QtCore import pyqtProperty as Property


class UserMessage(object):
    """
    A user message that should be displayed in a scheme view.

    Paramaters
    ----------
    contents : str
        Message text.
    severity : int
        Message severity.
    message_id : A hashable object
        Message id.
    data : dict
        A dictionary with optional extra data.

    """
    #: Severity flags
    Info, Warning, Error = 1, 2, 3

    def __init__(self, contents, severity=Info, message_id=None, data={}):
        self.contents = contents
        self.severity = severity
        self.message_id = message_id
        self.data = dict(data)


class SchemeNode(QObject):
    """
    A node in a :class:`.Scheme`.

    Parameters
    ----------
    description : :class:`WidgetDescription`
        Node description instance.
    title : str, optional
        Node title string (if None `description.name` is used).
    position : tuple
        (x, y) tuple of floats for node position in a visual display.
    properties : dict
        Additional extra instance properties (settings, widget geometry, ...)
    parent : :class:`QObject`
        Parent object.

    """

    def __init__(self, description, title=None, position=None,
                 properties=None, parent=None):
        QObject.__init__(self, parent)
        self.description = description

        if title is None:
            title = description.name

        self.__title = title
        self.__position = position or (0, 0)
        self.__progress = -1
        self.__processing_state = 0
        self.__status_message = ""
        self.__state_messages = {}
        self.properties = properties or {}

    def input_channels(self):
        """
        Return a list of input channels (:class:`InputSignal`) for the node.
        """
        return list(self.description.inputs)

    def output_channels(self):
        """
        Return a list of output channels (:class:`OutputSignal`) for the node.
        """
        return list(self.description.outputs)

    def input_channel(self, name):
        """
        Return the input channel matching `name`. Raise a `ValueError`
        if not found.

        """
        for channel in self.input_channels():
            if channel.name == name:
                return channel
        raise ValueError("%r is not a valid input channel name for %r." % \
                         (name, self.description.name))

    def output_channel(self, name):
        """
        Return the output channel matching `name`. Raise an `ValueError`
        if not found.

        """
        for channel in self.output_channels():
            if channel.name == name:
                return channel
        raise ValueError("%r is not a valid output channel name for %r." % \
                         (name, self.description.name))

    #: The title of the node has changed
    title_changed = Signal(str)

    def set_title(self, title):
        """
        Set the node title.
        """
        if self.__title != title:
            self.__title = str(title)
            self.title_changed.emit(self.__title)

    def title(self):
        """
        The node title.
        """
        return self.__title

    title = Property(str, fset=set_title, fget=title)

    #: Position of the node in the scheme has changed
    position_changed = Signal(tuple)

    def set_position(self, pos):
        """
        Set the position (``(x, y)`` tuple) of the node.
        """
        if self.__position != pos:
            self.__position = pos
            self.position_changed.emit(pos)

    def position(self):
        """
        ``(x, y)`` tuple containing the position of the node in the scheme.
        """
        return self.__position

    position = Property(tuple, fset=set_position, fget=position)

    #: Node's progress value has changed.
    progress_changed = Signal(float)

    def set_progress(self, value):
        """
        Set the progress value.
        """
        if self.__progress != value:
            self.__progress = value
            self.progress_changed.emit(value)

    def progress(self):
        """
        The current progress value. -1 if progress is not set.
        """
        return self.__progress

    progress = Property(float, fset=set_progress, fget=progress)

    #: Node's processing state has changed.
    processing_state_changed = Signal(int)

    def set_processing_state(self, state):
        """
        Set the node processing state.
        """
        if self.__processing_state != state:
            self.__processing_state = state
            self.processing_state_changed.emit(state)

    def processing_state(self):
        """
        The node processing state, 0 for not processing, 1 the node is busy.
        """
        return self.__processing_state

    processing_state = Property(int, fset=set_processing_state,
                                fget=processing_state)

    def set_tool_tip(self, tool_tip):
        if self.__tool_tip != tool_tip:
            self.__tool_tip = tool_tip

    def tool_tip(self):
        return self.__tool_tip

    tool_tip = Property(str, fset=set_tool_tip,
                        fget=tool_tip)

    #: The node's status tip has changes
    status_message_changed = Signal(str)

    def set_status_message(self, text):
        if self.__status_message != text:
            self.__status_message = text
            self.status_message_changed.emit(text)

    def status_message(self):
        return self.__status_message

    #: The node's state message has changed
    state_message_changed = Signal(UserMessage)

    def set_state_message(self, message):
        """
        Set a message to be displayed by a scheme view for this node.
        """
        if message.message_id in self.__state_messages and \
                not message.contents:
            del self.__state_messages[message.message_id]

        self.__state_messages[message.message_id] = message
        self.state_message_changed.emit(message)

    def state_messages(self):
        """
        Return a list of all state messages.
        """
        return self.__state_messages.values()

    def __str__(self):
        return "SchemeNode(description_id=%s, title=%r, ...)" % \
                (str(self.description.id), self.title)

    def __repr__(self):
        return str(self)
