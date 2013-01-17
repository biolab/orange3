"""
Undo/Redo Commands

"""

from PyQt4.QtGui import QUndoCommand


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
    def __init__(self, scheme, annotation, old, new, parent=None):
        QUndoCommand.__init__(self, "Change text", parent)
        self.scheme = scheme
        self.annotation = annotation
        self.old = old
        self.new = new

    def redo(self):
        self.annotation.text = self.new

    def undo(self):
        self.annotation.text = self.old


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
