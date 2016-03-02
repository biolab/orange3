"""
Qt Property Bindings (`propertybindings`)
-----------------------------------------


"""

import sys
import ast

from collections import defaultdict
from operator import add

from AnyQt.QtCore import QObject, QEvent, QT_VERSION
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

from functools import reduce


def find_meta_property(obj, name):
    """
    Return a named (`name`) `QMetaProperty` of a `QObject` instance `obj`.
    If a property by taht name does not exist raise an AttributeError.

    """
    meta = obj.metaObject()
    index = meta.indexOfProperty(name)
    if index == -1:
        raise AttributeError("%s does no have a property named %r." %
                             (meta.className(), name))

    return meta.property(index)


def find_notifier(obj, name):
    """
    Return the notifier signal name (`str`) for the property of
    `object` (instance of `QObject`).

    .. todo: Should it return a QMetaMethod instead?

    """
    prop_meta = find_meta_property(obj, name)
    if not prop_meta.hasNotifySignal():
        raise TypeError("%s does not have a notifier signal." %
                        name)

    notifier = prop_meta.notifySignal()
    if QT_VERSION < 0x50000:
        name = notifier.signature().split("(")[0]
    else:
        name = bytes(notifier.methodSignature()).decode("utf-8").split("(")[0]
    return name


class AbstractBoundProperty(QObject):
    """
    An abstract base class for property bindings.
    """

    changed = Signal([], [object])
    """Emited when the property changes"""

    def __init__(self, obj, propertyName, parent=None):
        QObject.__init__(self, parent)

        self.obj = obj
        self.propertyName = propertyName
        self.obj.destroyed.connect(self._on_destroyed)
        self._source = None

    def set(self, value):
        """
        Set `value` to the property.
        """
        return self.obj.setProperty(self.propertyName, value)

    def get(self):
        """
        Return the property value.
        """
        return self.obj.property(self.propertyName)

    @Slot()
    def notifyChanged(self):
        """
        Notify the binding of a change in the property value.
        The default implementation emits the `changed` signals.

        """
        val = self.get()
        self.changed.emit()
        self.changed[object].emit(val)

    def _on_destroyed(self):
        self.obj = None

    def bindTo(self, source):
        """
        Bind this property to `source` (instance of `AbstractBoundProperty`).
        """
        if self._source != source:
            if self._source:
                self.unbind()

            self._source = source

            source.changed.connect(self.update)
            source.destroyed.connect(self.unbind)
            self.set(source.get())
            self.notifyChanged()

    def unbind(self):
        """
        Unbind the currently bound property (set with `bindTo`).
        """
        self._source.destroyed.disconnect(self.unbind)
        self._source.changed.disconnect(self.update)
        self._source = None

    def update(self):
        """
        Update the property value from `source` property (`bindTo`).
        """
        if self._source:
            source_val = self._source.get()
            curr_val = self.get()
            if source_val != curr_val:
                self.set(source_val)

    def reset(self):
        """
        Reset the property if possible.
        """
        raise NotImplementedError


class PropertyBindingExpr(AbstractBoundProperty):
    def __init__(self, expression, globals={}, locals={}, parent=None):
        QObject.__init__(self, parent)

        self.ast = ast.parse(expression, mode="eval")
        self.code = compile(self.ast, "<unknown>", "eval")

        self.expression = expression
        self.globals = dict(globals)
        self.locals = dict(locals)
        self._sources = {}

        names = self.code.co_names
        for name in names:
            v = locals.get(name, globals.get(name))
            if isinstance(v, AbstractBoundProperty):
                self._sources[name] = v
                v.changed.connect(self.notifyChanged)
                v.destroyed.connect(self._on_destroyed)

    def sources(self):
        """Return all source property bindings appearing in the
        expression namespace.

        """
        return list(self._sources)

    def set(self, value):
        raise NotImplementedError("Cannot set a value of an expression")

    def get(self):
        locals = dict(self.locals)
        locals.update(dict((name, source.get())
                           for name, source in self._sources.items()))
        try:
            value = eval(self.code, self.globals, locals)
        except Exception:
            raise
        return value

    def bindTo(self, source):
        raise NotImplementedError("Cannot bind an expression")

    def _on_destroyed(self):
        source = self.sender()
        self._sources.remove(source)


class PropertyBinding(AbstractBoundProperty):
    """
    A Property binding of a QObject's property registered with Qt's
    meta class object system.

    """
    def __init__(self, obj, propertyName, notifier=None, parent=None):
        AbstractBoundProperty.__init__(self, obj, propertyName, parent)

        if notifier is None:
            notifier = find_notifier(obj, propertyName)

        if notifier is not None:
            signal = getattr(obj, notifier)
            signal.connect(self.notifyChanged)
        else:
            signal = None

        self.notifierSignal = signal

    def _on_destroyed(self):
        self.notifierSignal = None

        AbstractBoundProperty._on_destroyed(self)

    def reset(self):
        meta_prop = find_meta_property(self, self.obj, self.propertyName)
        if meta_prop.isResetable():
            meta_prop.reset(self.obj)
        else:
            return AbstractBoundProperty.reset(self)


class DynamicPropertyBinding(AbstractBoundProperty):
    """
    A Property binding of a QObject's dynamic property.
    """
    def __init__(self, obj, propertyName, parent=None):
        AbstractBoundProperty.__init__(self, obj, propertyName, parent)

        obj.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self.obj and event.type() == QEvent.DynamicPropertyChange:
            if event.propertyName() == self.propertyName:
                self.notifyChanged()

        return AbstractBoundProperty.eventFilter(self, obj, event)


class BindingManager(QObject):
    AutoSubmit = 0
    ManualSubmit = 1

    # Note: This should also apply to Gnome
    Default = 0 if sys.platform == "darwin" else 1

    def __init__(self, parent=None, submitPolicy=Default):
        QObject.__init__(self, parent)
        self._bindings = defaultdict(list)
        self._modified = set()
        self.__submitPolicy = submitPolicy

    def setSubmitPolicy(self, policy):
        if self.__submitPolicy != policy:
            self.__submitPolicy = policy
            if policy == BindingManager.AutoSubmit:
                self.commit()

    def submitPolicy(self):
        return self.__submitPolicy

    def bind(self, target, source):
        if isinstance(target, tuple):
            target = binding_for(*target + (self, ))

        if source is None:
            return UnboundBindingWrapper(target, self)

        else:
            if isinstance(source, tuple):
                source = binding_for(*source + (self,))

            source.changed.connect(self.__on_changed)
            self._bindings[source].append((target, source))
            self.__on_changed(source)

            return None

    def bindings(self):
        """Return (target, source) binding tuples.
        """
        return reduce(add, self._bindings.items(), [])

    def commit(self):
        self.__update()

    def __on_changed(self, sender=None):
        if sender is None:
            sender = self.sender()
        self._modified.add(sender)
        if self.__submitPolicy == BindingManager.AutoSubmit:
            self.__update()

    def __update(self):
        for modified in list(self._modified):
            self._modified.remove(modified)
            for target, source in self._bindings.get(modified, []):
                target.set(source.get())


class UnboundBindingWrapper(object):
    def __init__(self, target, manager):
        self.target = target
        self.manager = manager
        self.__source = None

    def to(self, source):
        if self.__source is None:
            if isinstance(source, tuple):
                source = binding_for(*source + (self.manager,))

            self.manager.bind(self.target, source)
            self.__source = source
        else:
            raise ValueError("Can only call 'to' once.")


def binding_for(obj, name, parent=None):
    """
    Return a suitable binding for property `name` of an `obj`.
    Currently only supports PropertyBinding and DynamicPropertyBinding.

    """
    if isinstance(obj, QObject):
        meta = obj.metaObject()
        index = meta.indexOfProperty(name)
        if index == -1:
            boundprop = DynamicPropertyBinding(obj, name, parent)
        else:
            boundprop = PropertyBinding(obj, name, parent)
    else:
        raise TypeError
    return boundprop
