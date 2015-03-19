import os
import time
import copy
import itertools
import pickle
import warnings

from Orange.canvas.utils import environ
from Orange.data import DiscreteVariable, Domain, Variable, ContinuousVariable
from Orange.widgets.utils import vartype

__all__ = ["Setting", "SettingsHandler",
           "ContextSetting", "ContextHandler",
           "DomainContextHandler", "PerfectDomainContextHandler",
           "ClassValuesContextHandler"]

_immutables = (str, int, bytes, bool, float, tuple)

class Setting:
    """Description of a setting.
    """

    # A misleading docstring for providing type hints for Settings to PyCharm
    def __new__(cls, default, *args, **kw_args):
        """
        :type: default: T
        :rtype: T
        """
        return super().__new__(cls)

    def __init__(self, default, **data):
        self.name = None  # Name gets set in widget's meta class
        self.default = default
        self.__dict__.update(data)

    def __str__(self):
        return "%s \"%s\"" % (self.__class__.__name__, self.name)

    __repr__ = __str__


class SettingProvider:
    """A hierarchical structure keeping track of settings belonging to
    a class and child setting providers.

    At instantiation, it creates a dict of all Setting and SettingProvider
    members of the class. This dict is used to get/set values of settings
    from/to the instances of the class this provider belongs to.
    """

    def __init__(self, provider_class):
        """ Construct a new instance of SettingProvider.

        Traverse provider_class members and store all instances of
        Setting and SettingProvider.
        """
        self.name = None
        self.provider_class = provider_class
        self.providers = {}
        """:type: dict[str, SettingProvider]"""
        self.settings = {}
        """:type: dict[str, Setting]"""
        self.initialization_data = None

        for name in dir(provider_class):
            value = getattr(provider_class, name, None)
            if isinstance(value, Setting):
                value.name = name
                self.settings[name] = value
            if isinstance(value, SettingProvider):
                value.name = name
                self.providers[name] = value

    def initialize(self, instance, data=None):
        """Initialize instance settings to their default values.

        If default value is mutable, create a shallow copy before assigning it to the instance.

        If data is provided, setting values from data will override defaults.
        """
        if data is None and self.initialization_data is not None:
            data = self.initialization_data

        for name, setting in self.settings.items():
            if data and name in data:
                setattr(instance, name, data[name])
            elif isinstance(setting.default, _immutables):
                setattr(instance, name, setting.default)
            else:
                setattr(instance, name, copy.copy(setting.default))

        for name, provider in self.providers.items():
            if data and name in data:
                if hasattr(instance, name) and not isinstance(getattr(instance, name), SettingProvider):
                    provider.initialize(getattr(instance, name), data[name])
                else:
                    provider.store_initialization_data(data[name])

    def store_initialization_data(self, initialization_data):
        """Store initialization data for later use.

        Used when settings handler is initialized, but member for this provider
        does not exists yet (because handler.initialize is called in __new__, but
        member will be created in __init__.
        """
        self.initialization_data = initialization_data

    def pack(self, instance, packer=None):
        """Pack instance settings in name: value dict.

        packer: optional packing function
                it will be called with setting and instance parameters and should
                yield (name, value) pairs that will be added to the packed_settings.
        """
        if packer is None:
            # noinspection PyShadowingNames
            def packer(setting, instance):
                if hasattr(instance, setting.name):
                    yield setting.name, getattr(instance, setting.name)

        packed_settings = dict(itertools.chain(
            *(packer(setting, instance) for setting in self.settings.values())
        ))

        packed_settings.update({
            name: provider.pack(getattr(instance, name), packer)
            for name, provider in self.providers.items()
            if hasattr(instance, name)
        })
        return packed_settings

    def unpack(self, instance, data):
        """Restore settings from data to the instance.

        instance: instance to restore settings to
        data: dictionary containing packed data
        """
        for setting, data, instance in self.traverse_settings(data, instance):
            if setting.name in data and instance is not None:
                setattr(instance, setting.name, data[setting.name])

    def get_provider(self, provider_class):
        """Return provider for provider_class.

        If this provider matches, return it, otherwise pass
        the call to child providers.
        """
        if issubclass(provider_class, self.provider_class):
            return self

        for subprovider in self.providers.values():
            provider = subprovider.get_provider(provider_class)
            if provider:
                return provider

    def traverse_settings(self, data=None, instance=None):
        """Generator of tuples (setting, data, instance) for each setting
        in this and child providers..

        :param data: dictionary with setting values
        :type data: dict
        :param instance: instance matching setting_provider
        """
        data = data if data is not None else {}
        select_data = lambda x: data.get(x.name, {})
        select_instance = lambda x: getattr(instance, x.name, None)

        for setting in self.settings.values():
            yield setting, data, instance

        for provider in self.providers.values():
            for setting, component_data, component_instance in \
                    provider.traverse_settings(select_data(provider), select_instance(provider)):
                yield setting, component_data, component_instance


class SettingsHandler:
    """Reads widget setting files and passes them to appropriate providers."""

    def __init__(self):
        """Create a setting handler template.

        Used in class definition. Bound instance will be created
        when SettingsHandler.create is called.
        """
        self.widget_class = None
        self.provider = None
        """:type: SettingProvider"""
        self.defaults = {}

    @staticmethod
    def create(widget_class, template=None):
        """Create a new settings handler based on the template and bind it to
        widget_class.

        :type template: SettingsHandler
        :rtype: SettingsHandler
        """

        if template is None:
            template = SettingsHandler()

        setting_handler = copy.copy(template)
        setting_handler.bind(widget_class)
        return setting_handler

    def bind(self, widget_class):
        """Bind settings handler instance to widget_class."""
        self.widget_class = widget_class
        self.provider = SettingProvider(widget_class)
        self.read_defaults()

    # noinspection PyBroadException
    def read_defaults(self):
        """Read (global) defaults for this widget class from a file.
        Opens a file and calls :obj:`read_defaults_file`. Derived classes
        should overload the latter."""
        filename = self._get_settings_filename()
        if os.path.exists(filename):
            settings_file = open(filename, "rb")
            try:
                self.read_defaults_file(settings_file)
            except Exception as e:
                warnings.warn("Could not read defaults for widget %s.\n" % self.widget_class +
                              "The following error occurred:\n\n%s" % e)
            finally:
                settings_file.close()

    def read_defaults_file(self, settings_file):
        """Read (global) defaults for this widget class from a file."""
        defaults = pickle.load(settings_file)
        self.defaults = {
            key: value
            for key, value in defaults.items()
            if not isinstance(value, Setting)
        }

    def write_defaults(self):
        """Write (global) defaults for this widget class to a file.
        Opens a file and calls :obj:`write_defaults_file`. Derived classes
        should overload the latter."""
        filename = self._get_settings_filename()
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
        pickle.dump(self.defaults, settings_file, -1)

    def _get_settings_filename(self):
        """Return the name of the file with default settings for the widget"""
        return os.path.join(environ.widget_settings_dir,
                            self.widget_class.name + ".ini")

    def initialize(self, instance, data=None):
        """
        Initialize widget's settings.

        Replace all instance settings with their default values.

        :param instance: the instance whose settings will be initialized
        :param data: dict of values that will be used instead of defaults.
        :type data: `dict` or `bytes` that unpickle into a `dict`
        """
        provider = self._select_provider(instance)

        if isinstance(data, bytes):
            data = pickle.loads(data)

        if provider is self.provider:
            data = self._add_defaults(data)

        provider.initialize(instance, data)

    def _select_provider(self, instance):
        provider = self.provider.get_provider(instance.__class__)
        if provider is None:
            import warnings

            message = "%s has not been declared as setting provider in %s. " \
                      "Settings will not be saved/loaded properly. Defaults will be used instead." \
                      % (instance.__class__, self.widget_class)
            warnings.warn(message)
            provider = SettingProvider(instance.__class__)
        return provider

    def _add_defaults(self, data):
        if data is None:
            return self.defaults

        new_data = self.defaults.copy()
        new_data.update(data)
        return new_data

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
        return self.provider.pack(widget)

    def update_defaults(self, widget):
        """
        Writes widget instance's settings to class defaults. Called when the
        widget is deleted.
        """
        self.defaults = self.provider.pack(widget)
        self.write_defaults()

    # TODO would we like this method to store the changed settings back to
    # class defaults, so the new widgets added to the schema later would have
    # different defaults? I guess so...
    def fast_save(self, widget, name, value):
        """Store the (changed) widget's setting immediately to the context."""
        pass

    @staticmethod
    def update_packed_data(data, name, value):
        split_name = name.split('.')
        prefixes, name = split_name[:-1], split_name[-1]
        for prefix in prefixes:
            data = data.setdefault(prefix, {})
        data[name] = value

    def reset_settings(self, instance):
        for setting, data, instance in self.provider.traverse_settings(instance=instance):
            if type(setting) == Setting:
                setattr(instance, setting.name, setting.default)


class ContextSetting(Setting):
    OPTIONAL = 0
    IF_SELECTED = 1
    REQUIRED = 2

    # These flags are not general - they assume that the setting has to do
    # something with the attributes. Large majority does, so this greatly
    # simplifies the declaration of settings in widget at no (visible)
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
        self.known_settings = {}

    def initialize(self, instance, data=None):
        """Initialize the widget: call the inherited initialization and
        add an attribute 'context_settings' to the widget. This method
        does not open a context."""
        instance.current_context = None
        super().initialize(instance, data)
        if data and "context_settings" in data:
            instance.context_settings = data["context_settings"]
        else:
            instance.context_settings = self.global_contexts

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

    def update_defaults(self, widget):
        """Call the inherited method, then merge the local context into the
        global contexts. This make sense only when the widget does not use
        global context (i.e. `widget.context_settings is not
        self.global_contexts`); this happens when the widget was initialized by
        an instance-specific data that was passed to :obj:`initialize`."""
        super().update_defaults(widget)
        globs = self.global_contexts
        if widget.context_settings is not globs:
            ids = {id(c) for c in globs}
            globs += (c for c in widget.context_settings if id(c) not in ids)
            globs.sort(key=lambda c: -c.time)
            del globs[self.MAX_SAVED_CONTEXTS:]

    def new_context(self):
        """Create a new context."""
        return Context()

    def open_context(self, widget, *args, **kwargs):
        """Open a context by finding one and setting the widget data or
        creating one and fill with the data from the widget."""
        widget.current_context, is_new = \
            self.find_or_create_context(widget, *args, **kwargs)
        if is_new:
            self.settings_from_widget(widget)
        else:
            self.settings_to_widget(widget)

    def match(self, context, *args, **kwargs):
        """Return the degree to which the stored `context` matches the data
         passed in additional arguments). A match of 0 zero indicates that
         the context cannot be used and 2 means a perfect match, so no further
         search is necessary.

         Derived classes must overload this method."""
        raise TypeError(self.__class__.__name__ + " does not overload match")

    def find_or_create_context(self, widget, *args, **kwargs):
        """Find the best matching context or create a new one if nothing
        useful is found. The returned context is moved to or added to the top
        of the context list."""
        best_context, best_score = None, 0
        for i, context in enumerate(widget.context_settings):
            score = self.match(context, *args, **kwargs)
            if score == 2:
                self.move_context_up(widget, i)
                return context, False
            if score > best_score:  # 0 is not OK!
                best_context, best_score = context, score
        if best_context:
            # if cloneIfImperfect should be disabled, change this and the
            # add_context below
            context = self.clone_context(best_context, *args)
        else:
            context = self.new_context()
        self.add_context(widget, context)
        return context, best_context is None

    @staticmethod
    def move_context_up(widget, index):
        """Move the context to the top of the context list and set the time
        stamp to current."""
        setting = widget.context_settings.pop(index)
        setting.time = time.time()
        widget.context_settings.insert(0, setting)

    @staticmethod
    def add_context(widget, setting):
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
        if widget.current_context is None:
            return

        self.settings_from_widget(widget)
        widget.current_context = None

    # TODO this method has misleading name (method 'initialize' does what
    #      this method's name would indicate.
    def settings_to_widget(self, widget):
        widget.retrieveSpecificSettings()

    # TODO similar to settings_to_widget; update_class_defaults does this for
    #      context independent settings
    def settings_from_widget(self, widget):
        widget.storeSpecificSettings()

    def bind(self, widget_class):
        super().bind(widget_class)
        self.analyze_settings(self.provider, "")

    def analyze_settings(self, provider, prefix):
        for setting in provider.settings.values():
            self.analyze_setting(prefix, setting)

        for name, sub_provider in provider.providers.items():
            new_prefix = '%s%s.' % (prefix, name) if prefix else '%s.' % name
            self.analyze_settings(sub_provider, new_prefix)

    def analyze_setting(self, prefix, setting):
        self.known_settings[prefix + setting.name] = setting
        if isinstance(setting, ContextSetting):
            if hasattr(setting, 'selected'):
                self.known_settings[prefix + setting.selected] = setting


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

        self.known_settings = {}

    def analyze_setting(self, prefix, setting):
        super().analyze_setting(prefix, setting)
        if isinstance(setting, ContextSetting) and not setting.not_attribute:
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

        # noinspection PyShadowingNames
        def encode(attributes, encode_values):
            if not encode_values:
                return {v.name: vartype(v) 
                    for v in attributes}

            is_discrete = lambda x: isinstance(x, DiscreteVariable)
            return {v.name: v.values if is_discrete(v) else vartype(v)
                    for v in attributes}

        match = self.match_values
        if self.has_ordinary_attributes:
            if match == self.MATCH_VALUES_CLASS:
                attributes = encode(domain.attributes, False)
                attributes.update(encode(domain.class_vars, True))
            else:
                attributes = encode(domain, match == self.MATCH_VALUES_ALL)
        else:
            attributes = {}

        if self.has_meta_attributes:
            metas = encode(domain.metas, match == self.MATCH_VALUES_ALL)
        else:
            metas = {}

        return attributes, metas

    def new_context(self):
        """Create a new context."""
        context = super().new_context()
        context.attributes = {}
        context.metas = {}
        context.ordered_domain = []
        context.values = {}
        context.no_copy = ["ordered_domain"]
        return context

    #noinspection PyMethodOverriding,PyTupleAssignmentBalance
    def find_or_create_context(self, widget, domain):
        if not domain:
            return None, False

        if not isinstance(domain, Domain):
            domain = domain.domain

        encoded_domain = self.encode_domain(domain)
        context, is_new = \
            super().find_or_create_context(widget, domain, *encoded_domain)

        context.attributes, context.metas = encoded_domain

        if self.has_ordinary_attributes:
            context.ordered_domain = [(v.name, vartype(v)) for v in domain]
        else:
            context.ordered_domain = []
        if self.has_meta_attributes:
            context.ordered_domain += [(v.name, vartype(v))
                                       for v in domain.metas]
        return context, is_new

    def settings_to_widget(self, widget):
        super().settings_to_widget(widget)

        context = widget.current_context
        if context is None:
            return

        excluded = set()

        for setting, data, instance in \
                self.provider.traverse_settings(data=context.values, instance=widget):
            if not isinstance(setting, ContextSetting) or setting.name not in data:
                continue

            value = self.decode_setting(setting, data[setting.name])
            setattr(instance, setting.name, value)

            if isinstance(value, list):
                excluded |= set(value)
            else:
                if setting.not_attribute:
                    excluded.add(value)

            if hasattr(setting, "selected"):
                new_labels, new_selected = [], []
                old_selected = set(data.get(setting.selected, []))

                # noinspection PyShadowingNames
                def is_attribute(value):
                    return (not setting.exclude_attributes
                            and value in context.attributes)

                # noinspection PyShadowingNames
                def is_meta(value):
                    return (not setting.exclude_metas
                            and value in context.metas)

                for i, old_value in enumerate(value):
                    old_value = self.decode_setting(setting, old_value)
                    if is_attribute(old_value) or is_meta(old_value):
                        if i in old_selected:
                            new_selected.append(len(new_labels))
                        new_labels.append(old_value)

                data[setting.name] = new_labels
                data[setting.selected] = new_selected
                # first 'name', then 'selected' - this gets signalled to Qt
                setattr(instance, setting.name, new_labels)  # labels might have changed
                setattr(instance, setting.selected, new_selected)

        if self.reservoir is not None:
            get_attribute = lambda name: context.attributes.get(name, None)
            get_meta = lambda name: context.metas.get(name, None)
            ll = [a for a in context.ordered_domain if a not in excluded and (
                self.attributes_in_res and get_attribute(a[0]) == a[1] or
                self.metas_in_res and get_meta(a[0]) == a[1])]
            setattr(widget, self.reservoir, ll)

    def settings_from_widget(self, widget):
        super().settings_from_widget(widget)

        context = widget.current_context
        if context is None:
            return

        def packer(setting, instance):
            if hasattr(instance, setting.name):
                value = getattr(instance, setting.name)
                yield setting.name, self.encode_setting(context, setting, value)
                if hasattr(setting, "selected"):
                    yield setting.selected, list(getattr(instance, setting.selected))

        context.values = self.provider.pack(widget, packer=packer)

    def fast_save(self, widget, name, value):
        context = widget.current_context
        if not context:
            return

        if name in self.known_settings:
            setting = self.known_settings[name]

            if name == setting.name or name.endswith(".%s" % setting.name):
                value = self.encode_setting(context, setting, value)
            else:
                value = list(value)

            self.update_packed_data(context.values, name, value)

    def encode_setting(self, context, setting, value):
        value = copy.copy(value)
        if isinstance(value, list):
            return value
        elif isinstance(setting, ContextSetting) and isinstance(value, str):
            if not setting.exclude_attributes and value in context.attributes:
                return value, context.attributes[value]
            if not setting.exclude_metas and value in context.metas:
                return value, context.metas[value]
        return value, -2

    @staticmethod
    def decode_setting(setting, value):
        if isinstance(value, tuple):
            return value[0]
        else:
            return value

    @staticmethod
    def _var_exists(setting, value, attributes, metas):
        if not isinstance(value, tuple) or len(value) != 2:
            return False

        attr_name, attr_type = value
        return (not setting.exclude_attributes and
                attributes.get(attr_name, -1) == attr_type or
                not setting.exclude_metas and
                metas.get(attr_name, -1) == attr_type)

    #noinspection PyMethodOverriding
    def match(self, context, domain, attrs, metas):
        if (attrs, metas) == (context.attributes, context.metas):
            return 2

        matches = []
        try:
            for setting, data, instance in \
                    self.provider.traverse_settings(data=context.values):
                if not isinstance(setting, ContextSetting):
                    continue
                value = data.get(setting.name, None)

                if isinstance(value, list):
                    matches.append(
                        self.match_list(setting, value, context, attrs, metas))
                elif value is not None:
                    matches.append(
                        self.match_value(setting, value, attrs, metas))
        except IncompatibleContext:
            return 0

        matched, available = map(sum, zip(*matches)) if matches else (0, 0)
        if not available:
            return 0.1
        else:
            return matched / available

    def match_list(self, setting, value, context, attrs, metas):
        matched = 0
        if hasattr(setting, 'selected'):
            selected = set(context.values.get(setting.selected, []))
        else:
            selected = set()

        for i, item in enumerate(value):
            if self._var_exists(setting, item, attrs, metas):
                matched += 1
            else:
                if setting.required == ContextSetting.REQUIRED:
                    raise IncompatibleContext()
                if setting.IF_SELECTED and i in selected:
                    raise IncompatibleContext()

        return matched, len(value)

    def match_value(self, setting, value, attrs, metas):
        if value[1] < 0:
            return 0, 0

        if self._var_exists(setting, value, attrs, metas):
            return 1, 1
        else:
            raise IncompatibleContext()

    #noinspection PyMethodOverriding
    def clone_context(self, context, domain, attrs, metas):
        context = copy.deepcopy(context)

        for setting, data, instance in self.provider.traverse_settings(data=context.values):
            if not isinstance(setting, ContextSetting):
                continue

            value = data.get(setting.name, None)
            if isinstance(value, list):
                sel_name = getattr(setting, "selected", None)
                if sel_name is not None:
                    selected = data.get(sel_name, [])
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
                    data[sel_name] = selected[:j]
            elif value is not None:
                if (value[1] >= 0 and
                        not self._var_exists(setting, value, attrs, metas)):
                    del data[setting.name]

        context.attributes, context.metas = attrs, metas
        context.ordered_domain = [(attr.name, vartype(attr)) for attr in
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
            glob.sort(key=lambda context: -context.time)
            del glob[self.MAX_SAVED_CONTEXTS:]
        else:
            for i in range(len(glob) - 1, -1, -1):
                c = glob[i]
                n_attrs = ((c.attributes and len(c.attributes) or 0) +
                           (c.class_vars and len(c.class_vars) or 0) +
                           (c.metas and len(c.metas) or 0))
                if n_attrs >= mp:
                    del glob[i]


class IncompatibleContext(Exception):
    pass


class ClassValuesContextHandler(ContextHandler):
    #noinspection PyMethodOverriding
    def find_or_create_context(self, widget, classes):
        if isinstance(classes, Variable):
            if isinstance(classes, DiscreteVariable):
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
        if isinstance(classes, ContinuousVariable):
            return context.classes is None and 2
        else:
            return context.classes == classes and 2

    def settings_to_widget(self, widget):
        super().settings_to_widget(widget)
        context = widget.current_context
        self.provider.unpack(widget, context.values)

    def settings_from_widget(self, widget):
        super().settings_from_widget(widget)
        context = widget.current_context

        def packer(setting, instance):
            yield setting.name, copy.copy(getattr(instance, setting.name))

        context.values = self.provider.pack(widget, packer=packer)

    def fast_save(self, widget, name, value):
        if widget.current_context is None:
            return

        if name in self.known_settings:
            self.update_packed_data(widget.current_context.values, name, copy.copy(value))


### Requires the same the same attributes in the same order
### The class overloads domain encoding and matching.
### Due to different encoding, it also needs to overload encode_setting and
### clone_context (which is the same as the ContextHandler's)
### We could simplify some other methods, but prefer not to replicate the code
class PerfectDomainContextHandler(DomainContextHandler):
    def encode_domain(self, domain):
        if self.match_values == 2:
            def encode(attrs):
                return tuple(
                    (v.name,
                     v.values if isinstance(v, DiscreteVariable)
                     else vartype(v))
                    for v in attrs)
        else:
            def encode(attrs):
                return tuple((v.name, vartype(v)) for v in attrs)
        return (encode(domain.attributes),
                encode(domain.class_vars),
                encode(domain.metas))

    #noinspection PyMethodOverriding
    def match(self, context, domain, attributes, class_vars, metas):
        return (attributes, class_vars, metas) == (
            context.attributes, context.class_vars, context.metas) and 2

    def encode_setting(self, widget, setting, value):
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
            return value, copy.copy(atype)
        else:
            return super().encode_setting(widget, setting, value)

    def clone_context(self, context, _, *__):
        return copy.deepcopy(context)
