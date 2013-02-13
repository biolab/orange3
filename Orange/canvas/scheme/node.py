"""
===========
Scheme Node
===========

"""

from PyQt4.QtCore import QObject
from PyQt4.QtCore import pyqtSignal as Signal
from PyQt4.QtCore import pyqtProperty as Property


class SchemeNode(QObject):
    """A widget node instantiation in the scheme.

    Parameters
    ----------
    description : `WidgetDescription`
        Widget description.
    title : `str`
        Node title string (if None description.name is used).
    position : tuple
        (x, y) two-tuple for node position in a visual display.
    properties : `dict`
        Additional instance properties (settings, widget geometry, ...)
    parent : `QObject`
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
        self.properties = properties or {}

    def input_channels(self):
        """Return the input channels for the node.
        """
        return self.description.inputs

    def output_channels(self):
        """Return the output channels for the node.
        """
        return self.description.outputs

    def input_channel(self, name):
        """Return the input channel matching `name`. Raise an ValueError
        if not found.

        """
        for channel in self.input_channels():
            if channel.name == name:
                return channel
        raise ValueError("%r is not a valid input channel name for %r." % \
                         (name, self.description.name))

    def output_channel(self, name):
        """Return the output channel matching `name`. Raise an ValueError
        if not found.

        """
        for channel in self.output_channels():
            if channel.name == name:
                return channel
        raise ValueError("%r is not a valid output channel name for %r." % \
                         (name, self.description.name))

    def __str__(self):
        return "SchemeNode(description_id=%s, title=%r, ...)" % \
                (str(self.description.id), self.title)

    def __repr__(self):
        return str(self)

    title_changed = Signal(str)
    """The title of the node has changed"""

    def set_title(self, title):
        """Set the node's title
        """
        if self.__title != title:
            self.__title = str(title)
            self.title_changed.emit(self.__title)

    def title(self):
        """Return the nodes title.
        """
        return self.__title

    title = Property(str, fset=set_title, fget=title)

    position_changed = Signal(tuple)
    """Position of the node in the scheme has changed"""

    def set_position(self, pos):
        """Set the position of the node
        """
        if self.__position != pos:
            self.__position = pos
            self.position_changed.emit(pos)

    def position(self):
        """(x, y) tuple containing the position of the node in the scheme.
        """
        return self.__position

    position = Property(tuple, fset=set_position, fget=position)

    progress_changed = Signal(float)
    """Node's progress value has changed."""

    def set_progress(self, value):
        """Set the progress value.
        """
        if self.__progress != value:
            self.__progress = value
            self.progress_changed.emit(value)

    def progress(self):
        """Return the current progress value. -1 if progress is not set.
        """
        return self.__progress

    progress = Property(float, fset=set_progress, fget=progress)

    processing_state_changed = Signal(int)
    """Node's processing state has changed."""

    def set_processing_state(self, state):
        """Set the node's processing state
        """
        if self.__processing_state != state:
            self.__processing_state = state
            self.processing_state_changed.emit(state)

    def processing_state(self):
        """Return the node's processing state, 0 for not processing, 1 the
        node is busy.

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
