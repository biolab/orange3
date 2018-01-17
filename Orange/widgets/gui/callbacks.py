from collections import defaultdict, Sequence
from functools import wraps, partial
from operator import attrgetter

from .base import log

__all__ = ["OWComponent"]


class OWComponent:
    """
    Mixin for classes that contain settings and/or attributes that trigger
    callbacks when changed.

    The class initializes the settings handler, provides `__setattr__` that
    triggers callbacks, and provides `controls` attribute for access to
    Qt widgets controling particular attributes.

    Callbacks are exploited by controls (e.g. check boxes, line edits,
    combo boxes...) that are synchronized with attribute values. Changing
    the value of the attribute triggers a call to a function that updates
    the Qt widget accordingly.

    The class is mixed into `widget.OWWidget`, and must also be mixed into
    all widgets not derived from `widget.OWWidget` that contain settings or
    Qt widgets inserted by function in `Orange.widgets.gui` module. See
    `OWScatterPlotGraph` for an example.
    """
    def __init__(self, widget=None):
        # This may look ugly, but the alternative, a class outside the
        # constructor would require storing and accessing self.widget
        # self._get_component... while avoiding __getattr__ and __setattr__.
        # Feel free to refactor if you believe it is going to be nicer.
        class ControlGetter:
            """
            Provide access to GUI elements based on their corresponding attributes
            in widget.

            Every widget has an attribute `controls` that is an instance of this
            class, which uses the `controlled_attributes` dictionary to retrieve the
            control (e.g. `QCheckBox`, `QComboBox`...) corresponding to the attribute.
            For `OWComponents`, it returns its controls so that subsequent
            `__getattr__` will retrieve the control.
            """
            @staticmethod
            def _get_component(name):
                if "." in name:
                    path, name = name.rsplit(".", 1)
                    return attrgetter(path)(self), name
                else:
                    return self, name

            @staticmethod
            def __setattr__(name, control):
                component, name = ControlGetter._get_component(name)
                control_dict = component.controls.__dict__
                if name in control_dict:
                    log.warning("'%s' in '%s' is controlled by two controls",
                                name, self)
                else:
                    control_dict[name] = control

            @staticmethod
            def __getattr__(name):
                control_dict = self.controls.__dict__
                if name not in control_dict and hasattr(self, name):
                    return getattr(self, name).controls
                return control_dict[name]

        self.controlled_attributes = defaultdict(list)
        self.controls = ControlGetter()
        if widget is not None and widget.settingsHandler:
            widget.settingsHandler.initialize(self)

    def connect_control(self, name, func):
        """
        Add `func` to the list of functions called when the value of the
        attribute `name` is set.

        If the name includes a dot, it is assumed that the part the before the
        first dot is a name of an attribute containing an instance of a
        component, and the call is transferred to its `conntect_control`. For
        instance, `calling `obj.connect_control("graph.attr_x", f)` is
        equivalent to `obj.graph.connect_control("attr_x", f)`.

        Args:
            name (str): attribute name
            func (callable): callback function
        """
        if "." in name:
            name, rest = name.split(".", 1)
            sub = getattr(self, name)
            sub.connect_control(rest, func)
        else:
            self.controlled_attributes[name].append(func)

    def __setattr__(self, name, value):
        """Set the attribute value and trigger any attached callbacks.

        For backward compatibility, the name can include dots, e.g.
        `graph.attr_x`. `obj.__setattr__('x.y', v)` is equivalent to
        `obj.x.__setattr__('x', v)`.

        Args:
            name (str): attribute name
            value (object): value to set to the member.
        """
        if "." in name:
            name, rest = name.split(".", 1)
            sub = getattr(self, name)
            setattr(sub, rest, value)
        else:
            super().__setattr__(name, value)
            # First check that the widget is not just being constructed
            if hasattr(self, "controlled_attributes"):
                for callback in self.controlled_attributes.get(name, ()):
                    callback(value)


# TODO: Could this function become a method in OWComponent?
def connect_control(master, value, control, signal,
                    update_control=None, update_value=None, callback=None):
    blocks = [False, False]

    def blockable(direction, func, remove_args=False):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if blocks[not direction]:
                return
            try:
                blocks[direction] = True
                if remove_args:
                    func()
                else:
                    func(*args, **kwargs)
            except BaseException:
                log.error("Error in handler for '%s' of '%s'", value, master,
                          exc_info=True)
            finally:
                blocks[direction] = False
        return wrapper if func else None

    callback_wrapper = partial(blockable, True)
    callfront_wrapper = partial(blockable, False)

    # checking whether `val` is None seems either wrong or redundant,
    # but let's keep it for backward compatibility
    update_value = callback_wrapper(
        update_value or
        (lambda val: val is not None and setattr(master, value, val)))
    update_control = callfront_wrapper(update_control)
    if isinstance(callback, Sequence):
        cfuncs = callback = [callback_wrapper(func) for func in callback]
    else:
        callback = callback_wrapper(callback, remove_args=True)
        cfuncs = [callback] if callback else []

    if signal:
        if update_value:
            signal.connect(update_value)
        for func in cfuncs:
            signal.connect(func)
    if update_control and value:
        master.connect_control(value, update_control)
    setattr(master.controls, value, control)
    return update_control, update_value, callback
