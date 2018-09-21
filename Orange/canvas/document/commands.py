"""
Undo/Redo Commands

"""
from typing import Callable

from AnyQt.QtWidgets import QUndoCommand


class AddNodeCommand(QUndoCommand):
    def __init__(self, scheme, node, parent=None):
        QUndoCommand.__init__(self, "Add %s" % node.title, parent)
        self.scheme = scheme
        self.node = node

    def redo(self):
        self.scheme.add_node(self.node)

    def undo(self):
        self.scheme.remove_node(self.node)


class RemoveNodeCommand(QUndoCommand):
    def __init__(self, scheme, node, parent=None):
        QUndoCommand.__init__(self, "Remove %s" % node.title, parent)
        self.scheme = scheme
        self.node = node

        links = scheme.input_links(node) + \
                scheme.output_links(node)

        for link in links:
            RemoveLinkCommand(scheme, link, parent=self)

    def redo(self):
        # redo child commands
        QUndoCommand.redo(self)
        self.scheme.remove_node(self.node)

    def undo(self):
        self.scheme.add_node(self.node)
        # Undo child commands
        QUndoCommand.undo(self)


class AddLinkCommand(QUndoCommand):
    def __init__(self, scheme, link, parent=None):
        QUndoCommand.__init__(self, "Add link", parent)
        self.scheme = scheme
        self.link = link

    def redo(self):
        self.scheme.add_link(self.link)

    def undo(self):
        self.scheme.remove_link(self.link)


class RemoveLinkCommand(QUndoCommand):
    def __init__(self, scheme, link, parent=None):
        QUndoCommand.__init__(self, "Remove link", parent)
        self.scheme = scheme
        self.link = link

    def redo(self):
        self.scheme.remove_link(self.link)

    def undo(self):
        self.scheme.add_link(self.link)


class InsertNodeCommand(QUndoCommand):
    def __init__(self, scheme, new_node, old_link, new_links, parent=None):
        QUndoCommand.__init__(self, "Insert widget into link", parent)
        self.scheme = scheme
        self.inserted_widget = new_node
        self.original_link = old_link
        self.new_links = new_links

    def redo(self):
        self.scheme.add_node(self.inserted_widget)
        self.scheme.remove_link(self.original_link)
        self.scheme.add_link(self.new_links[0])
        self.scheme.add_link(self.new_links[1])

    def undo(self):
        self.scheme.remove_link(self.new_links[0])
        self.scheme.remove_link(self.new_links[1])
        self.scheme.add_link(self.original_link)
        self.scheme.remove_node(self.inserted_widget)


class AddAnnotationCommand(QUndoCommand):
    def __init__(self, scheme, annotation, parent=None):
        QUndoCommand.__init__(self, "Add annotation", parent)
        self.scheme = scheme
        self.annotation = annotation

    def redo(self):
        self.scheme.add_annotation(self.annotation)

    def undo(self):
        self.scheme.remove_annotation(self.annotation)


class RemoveAnnotationCommand(QUndoCommand):
    def __init__(self, scheme, annotation, parent=None):
        QUndoCommand.__init__(self, "Remove annotation", parent)
        self.scheme = scheme
        self.annotation = annotation

    def redo(self):
        self.scheme.remove_annotation(self.annotation)

    def undo(self):
        self.scheme.add_annotation(self.annotation)


class MoveNodeCommand(QUndoCommand):
    def __init__(self, scheme, node, old, new, parent=None):
        QUndoCommand.__init__(self, "Move", parent)
        self.scheme = scheme
        self.node = node
        self.old = old
        self.new = new

    def redo(self):
        self.node.position = self.new

    def undo(self):
        self.node.position = self.old


class ResizeCommand(QUndoCommand):
    def __init__(self, scheme, item, new_geom, parent=None):
        QUndoCommand.__init__(self, "Resize", parent)
        self.scheme = scheme
        self.item = item
        self.new_geom = new_geom
        self.old_geom = item.rect

    def redo(self):
        self.item.rect = self.new_geom

    def undo(self):
        self.item.rect = self.old_geom


class ArrowChangeCommand(QUndoCommand):
    def __init__(self, scheme, item, new_line, parent=None):
        QUndoCommand.__init__(self, "Move arrow", parent)
        self.scheme = scheme
        self.item = item
        self.new_line = new_line
        self.old_line = (item.start_pos, item.end_pos)

    def redo(self):
        self.item.set_line(*self.new_line)

    def undo(self):
        self.item.set_line(*self.old_line)


class AnnotationGeometryChange(QUndoCommand):
    def __init__(self, scheme, annotation, old, new, parent=None):
        QUndoCommand.__init__(self, "Change Annotation Geometry", parent)
        self.scheme = scheme
        self.annotation = annotation
        self.old = old
        self.new = new

    def redo(self):
        self.annotation.geometry = self.new

    def undo(self):
        self.annotation.geometry = self.old


class RenameNodeCommand(QUndoCommand):
    def __init__(self, scheme, node, old_name, new_name, parent=None):
        QUndoCommand.__init__(self, "Rename", parent)
        self.scheme = scheme
        self.node = node
        self.old_name = old_name
        self.new_name = new_name

    def redo(self):
        self.node.set_title(self.new_name)

    def undo(self):
        self.node.set_title(self.old_name)


class TextChangeCommand(QUndoCommand):
    def __init__(self, scheme, annotation,
                 old_content, old_content_type,
                 new_content, new_content_type, parent=None):
        QUndoCommand.__init__(self, "Change text", parent)
        self.scheme = scheme
        self.annotation = annotation
        self.old_content = old_content
        self.old_content_type = old_content_type
        self.new_content = new_content
        self.new_content_type = new_content_type

    def redo(self):
        self.annotation.set_content(self.new_content, self.new_content_type)

    def undo(self):
        self.annotation.set_content(self.old_content, self.old_content_type)


class SetAttrCommand(QUndoCommand):
    def __init__(self, obj, attrname, newvalue, name=None, parent=None):
        if name is None:
            name = "Set %r" % attrname
        QUndoCommand.__init__(self, name, parent)
        self.obj = obj
        self.attrname = attrname
        self.newvalue = newvalue
        self.oldvalue = getattr(obj, attrname)

    def redo(self):
        setattr(self.obj, self.attrname, self.newvalue)

    def undo(self):
        setattr(self.obj, self.attrname, self.oldvalue)


class SimpleUndoCommand(QUndoCommand):
    """
    Simple undo/redo command specified by callable function pair.
    Parameters
    ----------
    redo: Callable[[], None]
        A function expressing a redo action.
    undo : Callable[[], None]
        A function expressing a undo action.
    text : str
        The command's text (see `QUndoCommand.setText`)
    parent : Optional[QUndoCommand]
    """

    def __init__(self, redo, undo, text, parent=None):
        # type: (Callable[[], None], Callable[[], None], ...) -> None
        super().__init__(text, parent)
        self._redo = redo
        self._undo = undo

    def undo(self):
        # type: () -> None
        """Reimplemented."""
        self._undo()

    def redo(self):
        # type: () -> None
        """Reimplemented."""
        self._redo()
