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

class Setting:
    """Description of a setting.
       The default can be either an (immutable!) object or a callable that is
       used to set the default value.

       Callable settings are not saved as default for widgets; however, when
       a widget is saved within a schema, the data is packed.
    """
    def __init__(self, default, **data):
        self.default = default
        self.__dict__.update(data)


class SettingsHandler:
    """Holds the decription of widget's settings, stored as a dict
       whose keys are attribute names and values are instances of Setting
    """

    def __init__(self):
        self.widget_class = None
        self.settings = {}

    def get_settings_filename(self):
        """Return the name of the file with default settings for the widget"""
        return os.path.join(environ.widget_settings_dir,
                            self.widget_class._name + ".ini")

    def read_defaults(self):
        """Read (global) defaults for this widget class from a file.
        Opens a file and calls :obj:`read_defaults_file`. Derived classes
        should overload the latter."""
        filename = self.get_settings_filename()
        if os.path.exists(filename):
            settings_file = open(filename, "rb")
            try:
                self.read_defaults_file(settings_file)
            except (EOFError, IOError, pickle.UnpicklingError):
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
            if not callable(self.settings[name].default):
                setting.default = getattr(cls, name)
                default_settings[name] = setting
        pickle.dump(default_settings, settings_file)

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
        setting, the method checks whether the default setting (e.g.
        `MyWidget.point_size`) is callable; if it is, it calls it to get the
         value for the setting. Otherwise, the widget instance gets no
         specific attribute that would shadow the class attribute.

        Derived classes can add or retrieve additional information in the data,
        such as local contexts.

        :param widget: the widget whose settings are initialized
        :type widget: OWBaseWidget
        :param data: Widget-specific default data the overrides the class
                    defaults
        :type data: `dict` or `bytes` that unpickle into a `dict`
        """
        if isinstance(data, bytes):
            data = pickle.loads(data)
        for name, setting in self.settings.items():
            if data and name in data:
                setattr(widget, name, data[name])
            elif callable(setting.default):
                setattr(widget, name, setting.default())
            elif not isinstance(setting.default,
                                (str, int, bytes, bool, float, tuple)):
                setattr(widget, name, copy.copy(setting.default))
            # otherwise, keep the class attribute

    def pack_data(self, widget):
        """
        Pack the settings for the given widget. This method is used when
        saving schema, so that when the schema is reloaded the widget is
        initialized with its proper data and not the class-based defaults.
        See :obj:SettingsHandler.initialize for detailed explanation of its
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
            if type(setting) is Setting and not callable(setting.default):
                setattr(cls, name, widget.getattr_deep(name))
        # this is here only since __del__ is never called
        self.write_defaults()

    # TODO this method has misleading name (method 'initialize' does what
    #      this method's name would indicate. Moreover, the method is never
    #      called by this class but only by ContextHandlers. Perhaps it should
    #      be moved there.
    def settingsToWidget(self, widget):
        widget.retrieveSpecificSettings()

    # TODO similar to settingsToWidget; update_class_defaults does this for
    #      context independent settings
    def settingsFromWidget(self, widget):
        widget.storeSpecificSettings()

    # TODO would we like this method to store the changed settings back to
    # class defaults, so the new widgets added to the schema later would have
    # different defaults? I guess so...
    def fastSave(self, widget, name, value):
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
        for nc in getattr(self, "noCopy", []):
            if nc in s:
                del s[nc]
        return s


class ContextHandler(SettingsHandler):
    """Base class for setting handlers that can handle contexts."""

    maxSavedContexts = 50

    def __init__(self):
        super().__init__()
        self.globalContexts = []

    def initialize(self, widget, data=None):
        """Initialize the widget: call the inherited initialization and
        add an attribute 'contextSettings' to the widget. This method
        does not open a context."""
        super().initialize(widget, data)
        if data and "contextSettings" in data:
            widget.contextSettings = data["contextSettings"]
        else:
            widget.contextSettings = self.globalContexts

    def read_defaults_file(self, settings_file):
        """Call the inherited method, then read global context from the
           pickle."""
        super().read_defaults_file(settings_file)
        self.globalContexts = pickle.load(settings_file)

    def write_defaults_file(self, settings_file):
        """Call the inherited method, then add global context to the pickle."""
        super().write_defaults_file(settings_file)
        pickle.dump(self.globalContexts, settings_file)

    def pack_data(self, widget):
        """Call the inherited method, then add local contexts to the pickle."""
        data = super().pack_data(widget)
        data["contextSettings"] = widget.contextSettings
        return data

    def update_class_defaults(self, widget):
        """Call the inherited method, then merge the local context into the
        global contexts. This make sense only when the widget does not use
        global context (i.e. `widget.contextSettings is not
        self.globalContexts`); this happens when the widget was initialized by
        an instance-specific data that was passed to :obj:`initialize`."""
        super().update_class_defaults(widget)
        globs = self.globalContexts
        if widget.contextSettings is not globs:
            ids = {id(c) for c in globs}
            globs += (c for c in widget.contextSettings if id(c) not in ids)
            globs.sort(key=lambda c: -c.time)
            del globs[self.maxSavedContexts:]

    def newContext(self):
        """Create a new context."""
        return Context()

    def openContext(self, widget, *arg, **argkw):
        """Open a context by finding one and setting the widget data or
        creating one and fill with the data from the widget."""
        widget.currentContext, isNew = \
            self.findOrCreateContext(widget, *arg, **argkw)
        if isNew:
            self.settingsFromWidget(widget)
        else:
            self.settingsToWidget(widget)

    def match(self, context, *arg):
        """Return the degree to which the stored `context` matches the data
         passed in additional arguments). A match of 0 zero indicates that
         the context cannot be used and 2 means a perfect match, so no further
         search is necessary.

         Derived classes must overload this method."""
        raise SystemError(self.__class__.__name__ + " does not overload match")

    def findOrCreateContext(self, widget, *arg):
        """Find the best matching context or create a new one if nothing
        useful is found. The returned context is moved to or added to the top
        of the context list."""
        bestContext, bestScore = None, 0
        for i, context in enumerate(widget.contextSettings):
            score = self.match(context, *arg)
            if score == 2:
                self.moveContextUp(widget, i)
                return bestContext, False
            if score > bestScore:  # 0 is not OK!
                bestContext, bestScore = context, score
        if bestContext:
            # if cloneIfImperfect should be disabled, change this and the
            # addContext below
            context = self.cloneContext(bestContext, *arg)
        else:
            context = self.newContext()
        self.addContext(widget, context)
        return context, bestContext is None

    def moveContextUp(self, widget, index):
        """Move the context to the top of the context list and set the time
        stamp to current."""
        setting = widget.contextSettings.pop(index)
        setting.time = time.time()
        widget.contextSettings.insert(0, setting)

    def addContext(self, widget, setting):
        """Add the context to the top of the list."""
        s = widget.contextSettings
        s.insert(0, setting)
        del s[len(s):]

    def cloneContext(self, context, *arg):
        """Construct a copy of the context settings suitable for the context
        described by additional arguments. The method is called by
        findOrCreateContext with the same arguments. Any class that overloads
        :obj:`match` to accept additional arguments must also overload
        :obj:`cloneContext`."""
        return copy.deepcopy(context)

    def closeContext(self, widget):
        """Close the context by calling :obj:`settingsFromWidget` to write
        any relevant widget settings to the context."""
        self.settingsFromWidget(widget)



class DomainContextHandler(ContextHandler):
    MATCH_VALUES_NONE, MATCH_VALUES_CLASS, MATCH_VALUES_ALL = range(3)

    def __init__(self, maxAttributesToPickle=100, matchValues=0,
                 reservoir=None, attributes_in_res=True, metas_in_res=False):
        super().__init__()
        self.maxAttributesToPickle = maxAttributesToPickle
        self.matchValues = matchValues
        self.reservoir = reservoir
        self.attributes_in_res = attributes_in_res
        self.metas_in_res = metas_in_res

        self.hasOrdinaryAttributes = attributes_in_res
        self.hasMetaAttributes = metas_in_res
        for s in self.settings:
            if isinstance(s, ContextSetting) and not s.not_attribute:
                if not s.exclude_attributes:
                    self.hasOrdinaryAttributes = True
                if not s.exclude_metas:
                    self.hasMetaAttributes = True

    def encodeDomain(self, domain):
        def encode(lst, values):
            if values:
                return {v.name:
                        v.values if isinstance(v, data.DiscreteVariable)
                        else v.var_type
                        for v in lst}
            else:
                return {v.name: v.var_type for v in lst}

        match = self.matchValues
        if self.hasOrdinaryAttributes:
            if match == self.MATCH_VALUES_CLASS:
                attributes = encode(domain.attributes, False)
                attributes.update(encode(domain.class_vars, True))
            else:
                attributes = encode(domain, match == self.MATCH_VALUES_ALL)
        else:
            attributes = None

        if self.hasMetaAttributes:
            metas = encode(domain.metas, match == self.MATCH_VALUES_ALL)
        else:
            metas = None

        return attributes, metas


    #noinspection PyMethodOverriding,PyTupleAssignmentBalance
    def findOrCreateContext(self, widget, domain):
        if not domain:
            return None, False

        if not isinstance(domain, data.Domain):
            domain = domain.domain

        encodedDomain = self.encodeDomain(domain)
        context, isNew = super().findOrCreateContext(widget,
                                                     domain, *encodedDomain)
        if len(encodedDomain) == 2:
            context.attributes, context.metas = encodedDomain
        else:
            context.attributes, context.classVar, context.metas = encodedDomain

        if self.hasOrdinaryAttributes:
            context.orderedDomain = [(v.name, v.var_type) for v in domain]
        else:
            context.orderedDomain = []
        if self.hasMetaAttributes:
            context.orderedDomain += [(v.name, v.var_type)
                                      for v in domain.metas]
        if isNew:
            context.values = {}
            context.noCopy = ["orderedDomain"]
        return context, isNew


    def settingsToWidget(self, widget):
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

        super().settingsToWidget(widget)

        context = widget.currentContext
        attrItemsSet = attrSet(context.attributes)
        metaItemsSet = attrSet(context.metas)
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
                newLabels, newSelected = [], []
                has_selection = hasattr(setting, "selected")
                if has_selection:
                    oldSelected = context.values.get(setting.selected, [])
                    for i, saved in enumerate(value):
                        if (not setting.exclude_attributes and (
                                saved in context.attributes or
                                saved in attrItemsSet)
                            or not setting.exclude_metas and (
                                saved in context.metas or
                                saved in metaItemsSet)):
                            if i in oldSelected:
                                newSelected.append(len(newLabels))
                            newLabels.append(saved)
            context.values[name] = newLabels
            setattr(widget, name, value)
            excluded |= set(value)
            if has_selection:
                context.values[setting.selected] = newSelected
                # first 'name', then 'selected' - this gets signalled to Qt
                setattr(widget, setting.selected, newSelected)

        if self.reservoir is not None:
            ll = [a for a in context.orderedDomain if a not in excluded and (
                  self.attributes_in_res and
                  context.attributes.get(a[0], None) == a[1] or
                  self.metas_in_res and context.metas.get(a[0], None) == a[1])]
            setattr(widget, self.reservoir, ll)


    def settingsFromWidget(self, widget):
        super().settingsFromWidget(widget)
        context = widget.currentContext
        context.values = {}
        for name, setting in self.settings.items():
            value = widget.getattr_deep(name)
            if not isinstance(value, list):
                self.saveLow(widget, name, value, setting)
            else:
                context.values[name] = copy.copy(value)  # shallow copy
                if hasattr(setting, "selected"):
                    context.values[setting.selected] = list(
                        widget.getattr_deep(setting.selected))

    def fastSave(self, widget, name, value):
        context = widget.currentContext
        if context:
            for sname, setting in self.settings.items():
                if name == sname:
                    if not isinstance(value, list):
                        context.values[name] = copy.copy(value)  # shallow copy
                    else:
                        self.saveLow(widget, name, value, setting)
                    return
                if name == getattr(setting, "selected", ""):
                    context.values[setting.selected] = list(value)
                    return

    def saveLow(self, widget, name, value, setting):
        context = widget.currentContext
        value = copy.copy(value)
        if isinstance(value, str):
            valtype = -1 if setting.exclude_attributes else \
                context.attributes.get(value, -1)
            if valtype == -1:
                valtype = -1 if setting.exclude_meta_attributes else \
                    context.metas.get(value, -1)
            # if it is still -1, it is not an attribute
            context.values[name] = value, valtype
        else:
            context.values[name] = value, -2

    def __varExists(self, setting, value, attributes, metas):
        return (not setting.exclude_attributes and
                attributes.get(value[0], -1) == value[1] or
                not setting.exlclude_metas and
                metas.get(value[0], -1) == value[1])

    #noinspection PyMethodOverriding
    def match(self, context, domain, attrs, metas):
        if (attrs, metas) == (context.attributes, context.metas):
            return 2
        filled = potentiallyFilled = 0
        for name, setting in self.settings.items():
            if not isinstance(setting, ContextSetting):
                continue
            value = context.values.get(name, None)
            if not value:
                continue
            if isinstance(list, value):
                if setting.required == ContextSetting.REQUIRED:
                    potentiallyFilled += len(value)
                    filled += len(value)
                    for item in value:
                        if not self.__varExists(setting, item, attrs, metas):
                            return 0
                else:
                    selectedRequired = (
                        setting.required == ContextSetting.IF_SELECTED)
                    selected = context.values.get(setting.selected, [])
                    potentiallyFilled += len(selected)
                    for i in selected:
                        if self.__varExists(setting, value[i], attrs, metas):
                            filled += 1
                        else:
                            if selectedRequired:
                                return 0
            else:
                potentiallyFilled += 1
                if value[1] >= 0:
                    if self.__varExists(value, setting, attrs, metas):
                        filled += 1
                    else:
                        if setting.required == ContextSetting.REQUIRED:
                            return 0
        if not potentiallyFilled:
            return 0.1
        else:
            return filled / potentiallyFilled


    #noinspection PyMethodOverriding
    def cloneContext(self, context, domain, attrs, metas):
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
                    nextSel = selected and selected[0] or -1
                else:
                    selected = None
                    nextSel = -1
                i = j = realI = 0
                while i < len(value):
                    if self.__varExists(setting, value[i], attrs, metas):
                        if nextSel == realI:
                            selected[j] -= realI - i
                            j += 1
                            nextSel = j < len(selected) and selected[j] or -1
                        i += 1
                    else:
                        del value[i]
                        if nextSel == realI:
                            del selected[j]
                            nextSel = j < len(selected) and selected[j] or -1
                    realI += 1
                if sel_name is not None:
                    context.values[sel_name] = selected[:j]
            else:
                if (value[1] >= 0 and
                        not self.__varExists(setting, value, attrs, metas)):
                    del context.values[name]
        context.attributes, context.metas = attrs, metas
        context.orderedDomain = [(attr.name, attr.var_type) for attr in
                                 itertools.chain(domain, domain.metas)]
        return context

    def mergeBack(self, widget):
        glob = self.globalContexts
        mp = self.maxAttributesToPickle
        if widget.contextSettings is not glob:
            ids = {id(c) for c in glob}
            glob += (c for c in widget.contextSettings if id(c) not in ids and
                    ((c.attributes and len(c.attributes) or 0) +
                     (c.class_vars and len(c.class_vars) or 0) +
                     (c.metas and len(c.metas) or 0)) <= mp)
            glob.sort(key=lambda c: -c.time)
            del glob[self.maxSavedContexts:]
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
    def findOrCreateContext(self, widget, classes):
        if isinstance(classes, data.Variable):
            if isinstance(classes, data.DiscreteVariable):
                classes = classes.values
            else:
                classes = None
        context, isNew = super().findOrCreateContext(widget, classes)
        context.classes = classes
        if isNew:
            context.values = {}
        return context, isNew

    #noinspection PyMethodOverriding
    def match(self, context, classes):
        if isinstance(classes, data.ContinuousVariable):
            return context.classes is None and 2
        else:
            return context.classes == classes and 2

    def settingsToWidget(self, widget):
        super().settingsToWidget(widget)
        context = widget.currentContext
        for name, setting in self.settings.items():
            setattr(widget, name, context.values[name])

    def settingsFromWidget(self, widget):
        super().settingsFromWidget(widget)
        context = widget.currentContext
        values = context.values = {}
        for name, setting in self.settings.items():
            value = widget.getattr_deep(name)
            values[name] = copy.copy(value)

    def fastSave(self, widget, name, value):
        if name in self.settings:
            widget.currentContext.values[name] = copy.copy(value)


### Requires the same the same attributes in the same order
### The class overloads domain encoding and matching.
### Due to different encoding, it also needs to overload saveLow and
### cloneContext (which is the same as the ContextHandler's)
### We could simplify some other methods, but prefer not to replicate the code
class PerfectDomainContextHandler(DomainContextHandler):
    def encodeDomain(self, domain):
        if self.matchValues == 2:
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

    def saveLow(self, widget, name, value, setting):
        context = widget.currentContext
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


    def cloneContext(self, context, _, *__):
        import copy
        return copy.deepcopy(context)
