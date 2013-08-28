import os
import time
import copy
import itertools
import pickle
from Orange.canvas.utils import environ
from Orange import data

__all__ = ["Setting", "SettingsHandler",
           "ContextSetting", "ContextHandler",
           "DomainContextHandler", "PerfectDomainContextHandler",
           "ClassValuesContextHandler"]

_immutables = (str, int, bytes, bool, float, tuple)


class Setting:
    """Description of a setting.
    """
    def __init__(self, default, **data):
        self.name = None  # Name gets set in widget's meta class
        self.default = default
        self.__dict__.update(data)


class SettingsHandler:
    """Holds the description of widget's settings, stored as a dict
       whose keys are attribute names and values are instances of Setting
    """

    def __init__(self):
        self.widget_class = None
        self.settings = {}

    def get_settings_filename(self):
        """Return the name of the file with default settings for the widget"""
        return os.path.join(environ.widget_settings_dir,
                            self.widget_class._name + ".ini")

    # noinspection PyBroadException
    def read_defaults(self):
        """Read (global) defaults for this widget class from a file.
        Opens a file and calls :obj:`read_defaults_file`. Derived classes
        should overload the latter."""
        filename = self.get_settings_filename()
        if os.path.exists(filename):
            settings_file = open(filename, "rb")
            try:
                self.read_defaults_file(settings_file)
            except:
                pass
            finally:
                settings_file.close()

    def read_defaults_file(self, settings_file):
        """Read (global) defaults for this widget class from a file."""
        default_settings = pickle.load(settings_file)
        cls = self.widget_class
        for name, setting in default_settings.items():
            if name in self.settings:
                self.settings[name] = setting
                setattr(cls, name, setting.default)

    def write_defaults(self):
        """Write (global) defaults for this widget class to a file.
        Opens a file and calls :obj:`write_defaults_file`. Derived classes
        should overload the latter."""
        filename = self.get_settings_filename()
        settings_file = open(filename, "wb")
        try:
            self.write_defaults_file(settings_file)
        except (EOFError, IOError, pickle.PicklingError):
            settings_file.close()
            os.remove(filename)
        else:
            settings_file.close()

    def write_defaults_file(self, settings_file):
        """Write defaults for this widget class to a file"""
        cls = self.widget_class
        default_settings = {}
        for name, setting in self.settings.items():
            setting.default = getattr(cls, name)
            default_settings[name] = setting
        pickle.dump(default_settings, settings_file, -1)

    def initialize(self, widget, data=None):
        """
        Initialize the widget's settings.

        Before calling this method, the widget instance does not have its
        own settings to shadow the class attributes. (E.g. if widget class
        `MyWidget` has an attribute `point_size`, the class has attribute
        `MyWidget.point_size`, but there is not 'self.point_size`).

        If the widget was loaded from a schema, the schema provides the data
        (as a dictionary or bytes). The instance's attributes (e.g.
        `self.point_size`) are in this case initialized from `data`
        (e.g. `data['point_size']`).

        If there is no data or the data does not include a particular
        setting, the class setting is (shallow-)copied to the instance if it
        is mutable. Immutable settings are kept in the class.

        Derived classes can add or retrieve additional information in the data,
        such as local contexts.

        :param widget: the widget whose settings are initialized
        :type widget: OWWidget
        :param data: Widget-specific default data the overrides the class
                    defaults
        :type data: `dict` or `bytes` that unpickle into a `dict`
        """
        if isinstance(data, bytes):
            data = pickle.loads(data)
        for name, setting in self.settings.items():
            if data and name in data:
                setattr(widget, name, data[name])
            elif not isinstance(setting.default, _immutables):
                setattr(widget, name, copy.copy(setting.default))

    def pack_data(self, widget):
        """
        Pack the settings for the given widget. This method is used when
        saving schema, so that when the schema is reloaded the widget is
        initialized with its proper data and not the class-based defaults.
        See :obj:`SettingsHandler.initialize` for detailed explanation of its
        use.

        Inherited classes add other data, in particular widget-specific
        local contexts.
        """
        data = {}
        for name, setting in self.settings.items():
            data[name] = widget.getattr_deep(name)
        return data

    def update_class_defaults(self, widget):
        """
        Writes widget instance's settings to class defaults. Called when the
        widget is deleted.
        """
        cls = self.widget_class
        for name, setting in self.settings.items():
            # I'm not saving settings that I don't understand
            if type(setting) is Setting:
                setattr(cls, name, widget.getattr_deep(name))
        # this is here only since __del__ is never called
        self.write_defaults()

    # TODO this method has misleading name (method 'initialize' does what
    #      this method's name would indicate. Moreover, the method is never
    #      called by this class but only by ContextHandlers. Perhaps it should
    #      be moved there.
    def settings_to_widget(self, widget):
        widget.retrieveSpecificSettings()

    # TODO similar to settings_to_widget; update_class_defaults does this for
    #      context independent settings
    def settings_from_widget(self, widget):
        widget.storeSpecificSettings()

    # TODO would we like this method to store the changed settings back to
    # class defaults, so the new widgets added to the schema later would have
    # different defaults? I guess so...
    def fast_save(self, widget, name, value):
        """Store the (changed) widget's setting immediatelly to the context."""
        pass



class ContextSetting(Setting):
    OPTIONAL = 0
    IF_SELECTED = 1
    REQUIRED = 2

    # These flags are not general - they assume that the setting has to do
    # something with the attributes. Large majority does, so this greatly
    # simplifies the declaration of settings in widget at not (visible)
    # cost to those settings that don't need it
    def __init__(self, default, not_attribute=False, required=0,
                 exclude_attributes=False, exclude_metas=True, **data):
        super().__init__(default, **data)
        self.not_attribute = not_attribute
        self.exclude_attributes = exclude_attributes
        self.exclude_metas = exclude_metas
        self.required = required


class Context:
    def __init__(self, **argkw):
        self.time = time.time()
        self.values = {}
        self.__dict__.update(argkw)

    def __getstate__(self):
        s = dict(self.__dict__)
        for nc in getattr(self, "no_copy", []):
            if nc in s:
                del s[nc]
        return s


class ContextHandler(SettingsHandler):
    """Base class for setting handlers that can handle contexts."""

    MAX_SAVED_CONTEXTS = 50

    def __init__(self):
        super().__init__()
        self.global_contexts = []

    def initialize(self, widget, data=None):
        """Initialize the widget: call the inherited initialization and
        add an attribute 'context_settings' to the widget. This method
        does not open a context."""
        super().initialize(widget, data)
        if data and "context_settings" in data:
            widget.context_settings = data["context_settings"]
        else:
            widget.context_settings = self.global_contexts

    def read_defaults_file(self, settings_file):
        """Call the inherited method, then read global context from the
           pickle."""
        super().read_defaults_file(settings_file)
        self.global_contexts = pickle.load(settings_file)

    def write_defaults_file(self, settings_file):
        """Call the inherited method, then add global context to the pickle."""
        super().write_defaults_file(settings_file)
        pickle.dump(self.global_contexts, settings_file, -1)

    def pack_data(self, widget):
        """Call the inherited method, then add local contexts to the pickle."""
        data = super().pack_data(widget)
        data["context_settings"] = widget.context_settings
        return data

    def update_class_defaults(self, widget):
        """Call the inherited method, then merge the local context into the
        global contexts. This make sense only when the widget does not use
        global context (i.e. `widget.context_settings is not
        self.global_contexts`); this happens when the widget was initialized by
        an instance-specific data that was passed to :obj:`initialize`."""
        super().update_class_defaults(widget)
        globs = self.global_contexts
        if widget.context_settings is not globs:
            ids = {id(c) for c in globs}
            globs += (c for c in widget.context_settings if id(c) not in ids)
            globs.sort(key=lambda c: -c.time)
            del globs[self.MAX_SAVED_CONTEXTS:]

    def new_context(self):
        """Create a new context."""
        return Context()

    def open_context(self, widget, *arg, **argkw):
        """Open a context by finding one and setting the widget data or
        creating one and fill with the data from the widget."""
        widget.current_context, isNew = \
            self.find_or_create_context(widget, *arg, **argkw)
        if isNew:
            self.settings_from_widget(widget)
        else:
            self.settings_to_widget(widget)

    def match(self, context, *arg):
        """Return the degree to which the stored `context` matches the data
         passed in additional arguments). A match of 0 zero indicates that
         the context cannot be used and 2 means a perfect match, so no further
         search is necessary.

         Derived classes must overload this method."""
        raise SystemError(self.__class__.__name__ + " does not overload match")

    def find_or_create_context(self, widget, *arg):
        """Find the best matching context or create a new one if nothing
        useful is found. The returned context is moved to or added to the top
        of the context list."""
        best_context, best_score = None, 0
        for i, context in enumerate(widget.context_settings):
            score = self.match(context, *arg)
            if score == 2:
                self.move_context_up(widget, i)
                return context, False
            if score > best_score:  # 0 is not OK!
                best_context, best_score = context, score
        if best_context:
            # if cloneIfImperfect should be disabled, change this and the
            # add_context below
            context = self.clone_context(best_context, *arg)
        else:
            context = self.new_context()
        self.add_context(widget, context)
        return context, best_context is None

    def move_context_up(self, widget, index):
        """Move the context to the top of the context list and set the time
        stamp to current."""
        setting = widget.context_settings.pop(index)
        setting.time = time.time()
        widget.context_settings.insert(0, setting)

    def add_context(self, widget, setting):
        """Add the context to the top of the list."""
        s = widget.context_settings
        s.insert(0, setting)
        del s[len(s):]

    def clone_context(self, context, *arg):
        """Construct a copy of the context settings suitable for the context
        described by additional arguments. The method is called by
        find_or_create_context with the same arguments. A class that overloads
        :obj:`match` to accept additional arguments must also overload
        :obj:`clone_context`."""
        return copy.deepcopy(context)

    def close_context(self, widget):
        """Close the context by calling :obj:`settings_from_widget` to write
        any relevant widget settings to the context."""
        self.settings_from_widget(widget)



class DomainContextHandler(ContextHandler):
    MATCH_VALUES_NONE, MATCH_VALUES_CLASS, MATCH_VALUES_ALL = range(3)

    def __init__(self, max_vars_to_pickle=100, match_values=0,
                 reservoir=None, attributes_in_res=True, metas_in_res=False):
        super().__init__()
        self.max_vars_to_pickle = max_vars_to_pickle
        self.match_values = match_values
        self.reservoir = reservoir
        self.attributes_in_res = attributes_in_res
        self.metas_in_res = metas_in_res

        self.has_ordinary_attributes = attributes_in_res
        self.has_meta_attributes = metas_in_res
        for setting in self.settings:
            if isinstance(setting, ContextSetting) and \
                    not setting.not_attribute:
                if not setting.exclude_attributes:
                    self.has_ordinary_attributes = True
                if not setting.exclude_metas:
                    self.has_meta_attributes = True

    def encode_domain(self, domain):
        """
        domain: Orange.data.domain to encode
        return: dict mapping attribute name to type or list of values
                (based on the value of self.match_values attribute)
        """
        def encode(attributes, encode_values):
            if not encode_values:
                return {v.name: v.var_type for v in attributes}

            is_discrete = lambda x: isinstance(x, data.DiscreteVariable)
            return {v.name: v.values if is_discrete(v) else v.var_type
                    for v in attributes}

        match = self.match_values
        if self.has_ordinary_attributes:
            if match == self.MATCH_VALUES_CLASS:
                attributes = encode(domain.attributes, False)
                attributes.update(encode(domain.class_vars, True))
            else:
                attributes = encode(domain, match == self.MATCH_VALUES_ALL)
        else:
            attributes = None

        if self.has_meta_attributes:
            metas = encode(domain.metas, match == self.MATCH_VALUES_ALL)
        else:
            metas = None

        return attributes, metas

    #noinspection PyMethodOverriding,PyTupleAssignmentBalance
    def find_or_create_context(self, widget, domain):
        if not domain:
            return None, False

        if not isinstance(domain, data.Domain):
            domain = domain.domain

        encoded_domain = self.encode_domain(domain)
        context, isNew = \
            super().find_or_create_context(widget, domain, *encoded_domain)

        context.attributes, context.metas = encoded_domain

        if self.has_ordinary_attributes:
            context.ordered_domain = [(v.name, v.var_type) for v in domain]
        else:
            context.ordered_domain = []
        if self.has_meta_attributes:
            context.ordered_domain += [(v.name, v.var_type)
                                      for v in domain.metas]
        if isNew:
            context.values = {}
            context.no_copy = ["ordered_domain"]
        return context, isNew


    def settings_to_widget(self, widget):
        def attrSet(attrs):
            if isinstance(attrs, dict):
                try:
                    return set(attrs.items())
                except TypeError:
                    return list(attrs.items())
            elif isinstance(attrs, bool):
                return {}
            else:
                return set()

        super().settings_to_widget(widget)

        context = widget.current_context
        attr_items_set = attrSet(context.attributes)
        meta_items_set = attrSet(context.metas)
        excluded = set()

        for name, setting in self.settings.items():
            if (not isinstance(setting, ContextSetting) or
                    name not in context.values):
                continue
            # list of tuples (var, type) or a single tuple
            value = context.values[name]

            if isinstance(value, tuple):
                # TODO: is setattr supposed to check that we do not assign
                # values that are optional and do not exist? is context
                # cloning's filter enough to get rid of such attributes?
                setattr(widget, name, value[0])
                if setting.not_attribute:
                    excluded.add(value)
                continue
            else:
                new_labels, new_selected = [], []
                has_selection = hasattr(setting, "selected")
                if has_selection:
                    old_selected = context.values.get(setting.selected, [])
                    for i, saved in enumerate(value):
                        if (not setting.exclude_attributes and (
                                saved in context.attributes or
                                saved in attr_items_set)
                            or not setting.exclude_metas and (
                                saved in context.metas or
                                saved in meta_items_set)):
                            if i in old_selected:
                                new_selected.append(len(new_labels))
                            new_labels.append(saved)
            context.values[name] = new_labels
            setattr(widget, name, value)
            excluded |= set(value)
            if has_selection:
                context.values[setting.selected] = new_selected
                # first 'name', then 'selected' - this gets signalled to Qt
                setattr(widget, setting.selected, new_selected)

        if self.reservoir is not None:
            ll = [a for a in context.ordered_domain if a not in excluded and (
                  self.attributes_in_res and
                  context.attributes.get(a[0], None) == a[1] or
                  self.metas_in_res and context.metas.get(a[0], None) == a[1])]
            setattr(widget, self.reservoir, ll)

    def settings_from_widget(self, widget):
        super().settings_from_widget(widget)
        context = widget.current_context
        context.values = {}
        for name, setting in self.settings.items():
            value = widget.getattr_deep(name)
            context.values[name] = self.encode_setting(widget, setting, value)
            if hasattr(setting, "selected"):
                context.values[setting.selected] = list(
                    widget.getattr_deep(setting.selected))

    def fast_save(self, widget, name, value):
        context = widget.current_context
        if not context:
            return

        if name in self.settings:
            context.values[name] = \
                self.encode_setting(widget, self.settings[name], value)
        else:
            for setting in self.settings.values():
                if name == getattr(setting, "selected", ""):
                    context.values[setting.selected] = list(value)

    def encode_setting(self, widget, setting, value):
        context = widget.current_context
        value = copy.copy(value)
        if isinstance(value, list):
            return value
        elif isinstance(setting, ContextSetting) and isinstance(value, str):
            if not setting.exclude_attributes and value in context.attributes:
                return value, context.attributes[value]
            if not setting.exclude_metas and value in context.metas:
                return value, context.metas[value]

        return value, -2

    def _var_exists(self, setting, value, attributes, metas):
        attr_name, attr_type = value
        return (not setting.exclude_attributes and
                attributes.get(attr_name, -1) == attr_type or
                not setting.exclude_metas and
                metas.get(attr_name, -1) == attr_type)

    #noinspection PyMethodOverriding
    def match(self, context, domain, attrs, metas):
        if (attrs, metas) == (context.attributes, context.metas):
            return 2
        filled = potentially_filled = 0
        for name, setting in self.settings.items():
            if not isinstance(setting, ContextSetting):
                continue
            value = context.values.get(name, None)
            if value is None:
                continue
            if isinstance(value, list):
                if setting.required == ContextSetting.REQUIRED:
                    potentially_filled += len(value)
                    filled += len(value)
                    for item in value:
                        if not self._var_exists(setting, item, attrs, metas):
                            return 0
                else:
                    selected_required = (
                        setting.required == ContextSetting.IF_SELECTED)
                    selected = context.values.get(setting.selected, [])
                    potentially_filled += len(selected)
                    for i in selected:
                        if self._var_exists(setting, value[i], attrs, metas):
                            filled += 1
                        else:
                            if selected_required:
                                return 0
            else:
                potentially_filled += 1
                if value[1] >= 0:
                    if self._var_exists(value, setting, attrs, metas):
                        filled += 1
                    else:
                        if setting.required == ContextSetting.REQUIRED:
                            return 0
        if not potentially_filled:
            return 0.1
        else:
            return filled / potentially_filled


    #noinspection PyMethodOverriding
    def clone_context(self, context, domain, attrs, metas):
        context = copy.deepcopy(context)
        for name, setting in self.settings.items():
            if not isinstance(setting, ContextSetting):
                continue
            value = context.values.get(name, None)
            if value is None:
                continue
            if isinstance(value, list):
                sel_name = getattr(setting, "selected", None)
                if sel_name is not None:
                    selected = context.values.get(sel_name, [])
                    selected.sort()
                    next_sel = selected and selected[0] or -1
                else:
                    selected = None
                    next_sel = -1
                i = j = realI = 0
                while i < len(value):
                    if self._var_exists(setting, value[i], attrs, metas):
                        if next_sel == realI:
                            selected[j] -= realI - i
                            j += 1
                            next_sel = j < len(selected) and selected[j] or -1
                        i += 1
                    else:
                        del value[i]
                        if next_sel == realI:
                            del selected[j]
                            next_sel = j < len(selected) and selected[j] or -1
                    realI += 1
                if sel_name is not None:
                    context.values[sel_name] = selected[:j]
            else:
                if (value[1] >= 0 and
                        not self._var_exists(setting, value, attrs, metas)):
                    del context.values[name]
        context.attributes, context.metas = attrs, metas
        context.ordered_domain = [(attr.name, attr.var_type) for attr in
                                 itertools.chain(domain, domain.metas)]
        return context

    def mergeBack(self, widget):
        glob = self.global_contexts
        mp = self.max_vars_to_pickle
        if widget.context_settings is not glob:
            ids = {id(c) for c in glob}
            glob += (c for c in widget.context_settings if id(c) not in ids and
                    ((c.attributes and len(c.attributes) or 0) +
                     (c.class_vars and len(c.class_vars) or 0) +
                     (c.metas and len(c.metas) or 0)) <= mp)
            glob.sort(key=lambda c: -c.time)
            del glob[self.MAX_SAVED_CONTEXTS:]
        else:
            for i in range(len(glob) - 1, -1, -1):
                c = glob[i]
                n_attrs = ((c.attributes and len(c.attributes) or 0) +
                           (c.class_vars and len(c.class_vars) or 0) +
                           (c.metas and len(c.metas) or 0))
                if n_attrs >= mp:
                        del glob[i]



class ClassValuesContextHandler(ContextHandler):
    #noinspection PyMethodOverriding
    def find_or_create_context(self, widget, classes):
        if isinstance(classes, data.Variable):
            if isinstance(classes, data.DiscreteVariable):
                classes = classes.values
            else:
                classes = None
        context, is_new = super().find_or_create_context(widget, classes)
        context.classes = classes
        if is_new:
            context.values = {}
        return context, is_new

    #noinspection PyMethodOverriding
    def match(self, context, classes):
        if isinstance(classes, data.ContinuousVariable):
            return context.classes is None and 2
        else:
            return context.classes == classes and 2

    def settings_to_widget(self, widget):
        super().settings_to_widget(widget)
        context = widget.current_context
        for name, setting in self.settings.items():
            setattr(widget, name, context.values[name])

    def settings_from_widget(self, widget):
        super().settings_from_widget(widget)
        context = widget.current_context
        values = context.values = {}
        for name, setting in self.settings.items():
            value = widget.getattr_deep(name)
            values[name] = copy.copy(value)

    def fast_save(self, widget, name, value):
        if name in self.settings:
            widget.current_context.values[name] = copy.copy(value)


### Requires the same the same attributes in the same order
### The class overloads domain encoding and matching.
### Due to different encoding, it also needs to overload save_low and
### clone_context (which is the same as the ContextHandler's)
### We could simplify some other methods, but prefer not to replicate the code
class PerfectDomainContextHandler(DomainContextHandler):
    def encode_domain(self, domain):
        if self.match_values == 2:
            def encode(attrs):
                return tuple(
                    (v.name,
                     v.values if isinstance(v, data.DiscreteVariable)
                     else v.var_type)
                    for v in attrs)
        else:
            def encode(attrs):
                return tuple((v.name, v.var_type) for v in attrs)
        return (encode(domain.attributes),
                encode(domain.class_vars),
                encode(domain.metas))


    #noinspection PyMethodOverriding
    def match(self, context, domain, attributes, class_vars, metas):
        return (attributes, class_vars, metas) == (
            context.attributes, context.class_vars, context.metas) and 2

    def encode_setting(self, widget, name, value, setting):
        context = widget.current_context
        if isinstance(value, str):
            atype = -1
            if not setting.exclude_attributes:
                for aname, atype in itertools.chain(context.attributes,
                                                    context.class_vars):
                    if aname == value:
                        break
            if atype == -1 and not setting.exclude_metas:
                for aname, values in itertools.chain(context.attributes,
                                                     context.class_vars):
                    if aname == value:
                        break
            context.values[name] = value, copy.copy(atype)
        else:
            context.values[name] = value, -2


    def clone_context(self, context, _, *__):
        import copy
        return copy.deepcopy(context)
