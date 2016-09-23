"""Widget Settings and Settings Handlers

Settings are used to declare widget attributes that persist through sessions.
When widget is removed or saved to a schema file, its settings are packed,
serialized and stored. When a new widget is created, values of attributes
marked as settings are read from disk. When schema is loaded, attribute values
are set to one stored in schema.

Each widget has its own SettingsHandler that takes care of serializing and
storing of settings and SettingProvider that is incharge of reading and
writing the setting values.

All widgets extending from OWWidget use SettingsHandler, unless they
declare otherwise. SettingsHandler ensures that setting attributes
are replaced with default (last used) setting values when the widget is
initialized and stored when the widget is removed.

Widgets with settings whose values depend on the widget inputs use
settings handlers based on ContextHandler. These handlers have two
additional methods, open_context and close_context.

open_context is called when widgets receives new data. It finds a suitable
context and sets the widget attributes to the values stored in context.
If no suitable context exists, a new one is created and values from widget
are copied to it.

close_context stores values that were last used on the widget to the context
so they can be used alter. It should be called before widget starts modifying
(initializing) the value of the setting attributes.
"""

import copy
import itertools
import os
import pickle
import time
import warnings

from Orange.data import Domain, Variable
from Orange.misc.environ import widget_settings_dir
from Orange.widgets.utils import vartype

__all__ = ["Setting", "SettingsHandler",
           "ContextSetting", "ContextHandler",
           "DomainContextHandler", "PerfectDomainContextHandler",
           "ClassValuesContextHandler", "widget_settings_dir"]

_IMMUTABLES = (str, int, bytes, bool, float, tuple)


class Setting:
    """Description of a setting.
    """

    # Settings are automatically persisted to disk
    packable = True

    # Setting is only persisted to schema (default value does not change)
    schema_only = False

    def __new__(cls, default, *args, **kwargs):
        """A misleading docstring for providing type hints for Settings

        :type: default: T
        :rtype: T
        """
        return super().__new__(cls)

    def __init__(self, default, **data):
        self.name = None  # Name gets set in widget's meta class
        self.default = default
        self.__dict__.update(data)

    def __str__(self):
        return '{0} "{1}"'.format(self.__class__.__name__, self.name)

    __repr__ = __str__

    def __getnewargs__(self):
        return (self.default, )


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

        Parameters
        ----------
        provider_class : class
            class containing settings definitions
        """
        self.name = ""
        self.provider_class = provider_class
        self.providers = {}
        """:type: dict[str, SettingProvider]"""
        self.settings = {}
        """:type: dict[str, Setting]"""
        self.initialization_data = None

        for name in dir(provider_class):
            value = getattr(provider_class, name, None)
            if isinstance(value, Setting):
                value = copy.deepcopy(value)
                value.name = name
                self.settings[name] = value
            if isinstance(value, SettingProvider):
                value = copy.deepcopy(value)
                value.name = name
                self.providers[name] = value

    def initialize(self, instance, data=None):
        """Initialize instance settings to their default values.

        Mutable values are (shallow) copied before they are assigned to the
        widget. Immutable are used as-is.

        Parameters
        ----------
        instance : OWWidget
            widget instance to initialize
        data : Optional[dict]
            optional data used to override the defaults
            (used when settings are loaded from schema)
        """
        if data is None and self.initialization_data is not None:
            data = self.initialization_data

        self._initialize_settings(instance, data)
        self._initialize_providers(instance, data)

    def _initialize_settings(self, instance, data):
        for name, setting in self.settings.items():
            if data and name in data:
                setattr(instance, name, data[name])
            elif isinstance(setting.default, _IMMUTABLES):
                setattr(instance, name, setting.default)
            else:
                setattr(instance, name, copy.copy(setting.default))

    def _initialize_providers(self, instance, data):
        if not data:
            return

        for name, provider in self.providers.items():
            if name not in data:
                continue

            member = getattr(instance, name, None)
            if member is None or isinstance(member, SettingProvider):
                provider.store_initialization_data(data[name])
            else:
                provider.initialize(member, data[name])

    def store_initialization_data(self, initialization_data):
        """Store initialization data for later use.

        Used when settings handler is initialized, but member for this
        provider does not exists yet (because handler.initialize is called in
        __new__, but member will be created in __init__.

        Parameters
        ----------
        initialization_data : dict
            data to be used for initialization when the component is created
        """
        self.initialization_data = initialization_data

    @staticmethod
    def _default_packer(setting, instance):
        """A simple packet that yields setting name and value.

        Parameters
        ----------
        setting : Setting
        instance : OWWidget
        """
        if setting.packable:
            if hasattr(instance, setting.name):
                yield setting.name, getattr(instance, setting.name)
            else:
                warnings.warn("{0} is declared as setting on {1} "
                              "but not present on instance."
                              .format(setting.name, instance))

    def pack(self, instance, packer=None):
        """Pack instance settings in a name:value dict.

        Parameters
        ----------
        instance : OWWidget
            widget instance
        packer: callable (Setting, OWWidget) -> Generator[(str, object)]
            optional packing function
            it will be called with setting and instance parameters and
            should yield (name, value) pairs that will be added to the
            packed_settings.
        """
        if packer is None:
            packer = self._default_packer

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

        Parameters
        ----------
        instance : OWWidget
            instance to restore settings to
        data : dict
            packed data
        """
        for setting, data, instance in self.traverse_settings(data, instance):
            if setting.name in data and instance is not None:
                setattr(instance, setting.name, data[setting.name])

    def get_provider(self, provider_class):
        """Return provider for provider_class.

        If this provider matches, return it, otherwise pass
        the call to child providers.

        Parameters
        ----------
        provider_class : class
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

        Parameters
        ----------
        data : dict
            dictionary with setting values
        instance : OWWidget
            instance matching setting_provider
        """
        data = data if data is not None else {}

        for setting in self.settings.values():
            yield setting, data, instance

        for provider in self.providers.values():
            data_ = data.get(provider.name, {})
            instance_ = getattr(instance, provider.name, None)
            for setting, component_data, component_instance in \
                    provider.traverse_settings(data_, instance_):
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
        self.known_settings = {}

    @staticmethod
    def create(widget_class, template=None):
        """Create a new settings handler based on the template and bind it to
        widget_class.

        Parameters
        ----------
        widget_class : class
        template : SettingsHandler
            SettingsHandler to copy setup from

        Returns
        -------
        SettingsHandler
        """

        if template is None:
            template = SettingsHandler()

        setting_handler = copy.copy(template)
        setting_handler.defaults = {}
        setting_handler.bind(widget_class)
        return setting_handler

    def bind(self, widget_class):
        """Bind settings handler instance to widget_class.

        Parameters
        ----------
        widget_class : class
        """
        self.widget_class = widget_class
        self.provider = SettingProvider(widget_class)
        self.known_settings = {}
        self.analyze_settings(self.provider, "")
        self.read_defaults()

    def analyze_settings(self, provider, prefix):
        """Traverse through all settings known to the provider
        and analyze each of them.

        Parameters
        ----------
        provider : SettingProvider
        prefix : str
            prefix the provider is registered to handle
        """
        for setting in provider.settings.values():
            self.analyze_setting(prefix, setting)

        for name, sub_provider in provider.providers.items():
            new_prefix = '{0}{1}.'.format(prefix or '', name)
            self.analyze_settings(sub_provider, new_prefix)

    def analyze_setting(self, prefix, setting):
        """Perform any initialization task related to setting.

        Parameters
        ----------
        prefix : str
        setting : Setting
        """
        self.known_settings[prefix + setting.name] = setting

    def read_defaults(self):
        """Read (global) defaults for this widget class from a file.
        Opens a file and calls :obj:`read_defaults_file`. Derived classes
        should overload the latter."""
        filename = self._get_settings_filename()
        if os.path.isfile(filename):
            settings_file = open(filename, "rb")
            try:
                self.read_defaults_file(settings_file)
            # Unpickling exceptions can be of any type
            # pylint: disable=broad-except
            except Exception as ex:
                warnings.warn("Could not read defaults for widget {0}\n"
                              "The following error occurred:\n\n{1}"
                              .format(self.widget_class, ex))
            finally:
                settings_file.close()

    def read_defaults_file(self, settings_file):
        """Read (global) defaults for this widget class from a file.

        Parameters
        ----------
        settings_file : file-like object
        """
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
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        settings_file = open(filename, "wb")
        try:
            self.write_defaults_file(settings_file)
        except (EOFError, IOError, pickle.PicklingError):
            settings_file.close()
            os.remove(filename)
        else:
            settings_file.close()

    def write_defaults_file(self, settings_file):
        """Write defaults for this widget class to a file

        Parameters
        ----------
        settings_file : file-like object
        """
        pickle.dump(self.defaults, settings_file, -1)

    def _get_settings_filename(self):
        """Return the name of the file with default settings for the widget"""
        return os.path.join(widget_settings_dir(),
                            "{0.__module__}.{0.__qualname__}.pickle"
                            .format(self.widget_class))

    def initialize(self, instance, data=None):
        """
        Initialize widget's settings.

        Replace all instance settings with their default values.

        Parameters
        ----------
        instance : OWWidget
        data : dict or bytes that unpickle into a dict
            values used to override the defaults
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
            message = "{0} has not been declared as setting provider in {1}. " \
                      "Settings will not be saved/loaded properly. Defaults will be used instead." \
                      .format(instance.__class__, self.widget_class)
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

        Parameters
        ----------
        widget : OWWidget
        """
        return self.provider.pack(widget)

    def update_defaults(self, widget):
        """
        Writes widget instance's settings to class defaults. Called when the
        widget is deleted.

        Parameters
        ----------
        widget : OWWidget
        """
        self.defaults = self.provider.pack(widget)
        for name, setting in self.known_settings.items():
            if setting.schema_only:
                self.defaults.pop(name, None)
        self.write_defaults()

    def fast_save(self, widget, name, value):
        """Store the (changed) widget's setting immediately to the context.

        Parameters
        ----------
        widget : OWWidget
        name : str
        value : object

        """
        if name in self.known_settings:
            setting = self.known_settings[name]
            if not setting.schema_only:
                setting.default = value

    def reset_settings(self, instance):
        """Reset widget settings to defaults

        Parameters
        ----------
        instance : OWWidget
        """
        for setting, _, instance in self.provider.traverse_settings(instance=instance):
            if setting.packable:
                setattr(instance, setting.name, setting.default)


class ContextSetting(Setting):
    """Description of a context dependent setting"""

    OPTIONAL = 0
    IF_SELECTED = 1
    REQUIRED = 2

    # Context settings are not persisted, but are stored in context instead.
    packable = False

    # These flags are not general - they assume that the setting has to do
    # something with the attributes. Large majority does, so this greatly
    # simplifies the declaration of settings in widget at no (visible)
    # cost to those settings that don't need it
    # TODO: exclude_metas should be disabled by default
    def __init__(self, default, not_attribute=False, required=0,
                 exclude_attributes=False, exclude_metas=True, **data):
        super().__init__(default, **data)
        self.not_attribute = not_attribute
        self.exclude_attributes = exclude_attributes
        self.exclude_metas = exclude_metas
        self.required = required


class Context:
    """Class for data thad defines context and
    values that should be applied to widget if given context
    is encountered."""
    def __init__(self, **argkw):
        self.time = time.time()
        self.values = {}
        self.__dict__.update(argkw)

    def __getstate__(self):
        state = dict(self.__dict__)
        for nc in getattr(self, "no_copy", []):
            if nc in state:
                del state[nc]
        return state


class ContextHandler(SettingsHandler):
    """Base class for setting handlers that can handle contexts.

    Classes deriving from it need to implement method `match`.
    """

    NO_MATCH = 0
    PERFECT_MATCH = 2

    MAX_SAVED_CONTEXTS = 50

    def __init__(self):
        super().__init__()
        self.global_contexts = []
        self.known_settings = {}

    def analyze_setting(self, prefix, setting):
        super().analyze_setting(prefix, setting)
        if isinstance(setting, ContextSetting):
            if hasattr(setting, 'selected'):
                self.known_settings[prefix + setting.selected] = setting

    def initialize(self, instance, data=None):
        """Initialize the widget: call the inherited initialization and
        add an attribute 'context_settings' to the widget. This method
        does not open a context."""
        instance.current_context = None
        super().initialize(instance, data)
        if data and "context_settings" in data:
            instance.context_settings = data["context_settings"]
        else:
            instance.context_settings = []

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
        self.settings_from_widget(widget)
        data["context_settings"] = widget.context_settings
        return data

    def update_defaults(self, widget):
        """Call the inherited method, then merge the local context into the
        global contexts. This make sense only when the widget does not use
        global context (i.e. `widget.context_settings is not
        self.global_contexts`); this happens when the widget was initialized by
        an instance-specific data that was passed to :obj:`initialize`."""
        self.settings_from_widget(widget)

        super().update_defaults(widget)
        globs = self.global_contexts
        if widget.context_settings is not globs:
            ids = {id(c) for c in globs}
            globs += (c for c in widget.context_settings if id(c) not in ids)
            globs.sort(key=lambda c: -c.time)
            del globs[self.MAX_SAVED_CONTEXTS:]

    def new_context(self, *args):
        """Create a new context."""
        return Context()

    def open_context(self, widget, *args):
        """Open a context by finding one and setting the widget data or
        creating one and fill with the data from the widget."""
        widget.current_context, is_new = \
            self.find_or_create_context(widget, *args)
        if is_new:
            self.settings_from_widget(widget, *args)
        else:
            self.settings_to_widget(widget, *args)

    def match(self, context, *args):
        """Return the degree to which the stored `context` matches the data
        passed in additional arguments).
        When match returns 0 (ContextHandler.NO_MATCH), the context will not
        be used. When it returns ContextHandler.PERFECT_MATCH, the context
        is a perfect match so no further search is necessary.

        If imperfect matching is not desired, match should only
        return ContextHandler.NO_MATCH or ContextHandler.PERFECT_MATCH.

        Derived classes must overload this method.
        """
        raise NotImplementedError

    def find_or_create_context(self, widget, *args):
        """Find the best matching context or create a new one if nothing
        useful is found. The returned context is moved to or added to the top
        of the context list."""

        # First search the contexts that were already used in this widget instance
        best_context, best_score = self.find_context(widget.context_settings, args, move_up=True)
        # If the exact data was used, reuse the context
        if best_score == self.PERFECT_MATCH:
            return best_context, False

        # Otherwise check if a better match is available in global_contexts
        best_context, best_score = self.find_context(self.global_contexts, args,
                                                     best_score, best_context)
        if best_context:
            context = self.clone_context(best_context, *args)
        else:
            context = self.new_context(*args)
        # Store context in widget instance. It will be pushed to global_contexts
        # when (if) update defaults is called.
        self.add_context(widget.context_settings, context)
        return context, best_context is None

    def find_context(self, known_contexts, args, best_score=0, best_context=None, move_up=False):
        """Search the given list of contexts and return the context
         which best matches the given args.

        best_score and best_context can be used to provide base_values.
        """

        for i, context in enumerate(known_contexts):
            score = self.match(context, *args)
            if score == self.PERFECT_MATCH:
                if move_up:
                    self.move_context_up(known_contexts, i)
                return context, score
            if score > best_score:  # NO_MATCH is not OK!
                best_context, best_score = context, score
        return best_context, best_score

    @staticmethod
    def move_context_up(contexts, index):
        """Move the context to the top of the list and set
        the timestamp to current."""
        setting = contexts.pop(index)
        setting.time = time.time()
        contexts.insert(0, setting)

    def add_context(self, contexts, setting):
        """Add the context to the top of the list."""
        contexts.insert(0, setting)
        del contexts[self.MAX_SAVED_CONTEXTS:]

    def clone_context(self, old_context, *args):
        """Construct a copy of the context settings suitable for the context
        described by additional arguments. The method is called by
        find_or_create_context with the same arguments. A class that overloads
        :obj:`match` to accept additional arguments must also overload
        :obj:`clone_context`."""
        context = self.new_context(*args)
        context.values = copy.deepcopy(old_context.values)

        traverse = self.provider.traverse_settings
        for setting, data, _ in traverse(data=context.values):
            if not isinstance(setting, ContextSetting):
                continue

            self.filter_value(setting, data, *args)
        return context

    @staticmethod
    def filter_value(setting, data, *args):
        """Remove values related to setting that are invalid given args."""

    def close_context(self, widget):
        """Close the context by calling :obj:`settings_from_widget` to write
        any relevant widget settings to the context."""
        if widget.current_context is None:
            return

        self.settings_from_widget(widget)
        widget.current_context = None

    def settings_to_widget(self, widget, *args):
        """Apply context settings stored in currently opened context
        to the widget.
        """
        context = widget.current_context
        if context is None:
            return

        widget.retrieveSpecificSettings()

        for setting, data, instance in \
                self.provider.traverse_settings(data=context.values, instance=widget):
            if not isinstance(setting, ContextSetting) or setting.name not in data:
                continue

            value = self.decode_setting(setting, data[setting.name])
            setattr(instance, setting.name, value)
            if hasattr(setting, "selected") and setting.selected in data:
                setattr(instance, setting.selected, data[setting.selected])

    def settings_from_widget(self, widget, *args):
        """Update the current context with the setting values from the widget.
        """

        context = widget.current_context
        if context is None:
            return

        widget.storeSpecificSettings()

        def packer(setting, instance):
            if isinstance(setting, ContextSetting) and hasattr(instance, setting.name):
                value = getattr(instance, setting.name)
                yield setting.name, self.encode_setting(context, setting, value)
                if hasattr(setting, "selected"):
                    yield setting.selected, list(getattr(instance, setting.selected))

        context.values = self.provider.pack(widget, packer=packer)

    def fast_save(self, widget, name, value):
        """Update value of `name` setting in the current context to `value`
        """
        setting = self.known_settings.get(name)
        if isinstance(setting, ContextSetting):
            context = widget.current_context
            if context is None:
                return

            value = self.encode_setting(context, setting, value)
            self.update_packed_data(context.values, name, value)
        else:
            super().fast_save(widget, name, value)

    @staticmethod
    def update_packed_data(data, name, value):
        """Updates setting value stored in data dict"""

        *prefixes, name = name.split('.')
        for prefix in prefixes:
            data = data.setdefault(prefix, {})
        data[name] = value

    def encode_setting(self, context, setting, value):
        """Encode value to be stored in settings dict"""
        return copy.copy(value)

    def decode_setting(self, setting, value):
        """Decode settings value from the setting dict format"""
        return value


class DomainContextHandler(ContextHandler):
    """Context handler for widgets with settings that depend on
    the input dataset. Suitable settings are selected based on the
    data domain."""

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

        match = self.match_values
        encode = self.encode_variables
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

    @staticmethod
    def encode_variables(attributes, encode_values):
        """Encode variables to a list mapping name to variable type
        or a list of values."""

        if not encode_values:
            return {v.name: vartype(v) for v in attributes}

        return {v.name: v.values if v.is_discrete else vartype(v)
                for v in attributes}

    def new_context(self, domain, attributes, metas):
        """Create a new context."""
        context = super().new_context()
        context.attributes = attributes
        context.metas = metas
        context.ordered_domain = []
        if self.has_ordinary_attributes:
            context.ordered_domain += [(attr.name, vartype(attr))
                                       for attr in domain]
        if self.has_meta_attributes:
            context.ordered_domain += [(attr.name, vartype(attr))
                                       for attr in domain.metas]
        return context

    def open_context(self, widget, domain):
        if not domain:
            return None, False

        if not isinstance(domain, Domain):
            domain = domain.domain

        super().open_context(widget, domain, *self.encode_domain(domain))

    def filter_value(self, setting, data, domain, attrs, metas):
        value = data.get(setting.name, None)
        if isinstance(value, list):
            sel_name = getattr(setting, "selected", None)
            selected = set(data.pop(sel_name, []))
            new_selected, new_value = [], []
            for i, item in enumerate(value):
                if self.is_valid_item(setting, item, attrs, metas):
                    if i in selected:
                        new_selected.append(len(new_value))
                    new_value.append(item)

            data[setting.name] = new_value
            if hasattr(setting, 'selected'):
                data[setting.selected] = new_selected
        elif value is not None:
            if (value[1] >= 0 and
                    not self._var_exists(setting, value, attrs, metas)):
                del data[setting.name]

    def settings_to_widget(self, widget, domain, *args):
        context = widget.current_context
        if context is None:
            return

        widget.retrieveSpecificSettings()
        excluded = set()

        for setting, data, instance in \
                self.provider.traverse_settings(data=context.values, instance=widget):
            if not isinstance(setting, ContextSetting) or setting.name not in data:
                continue

            value = self.decode_setting(setting, data[setting.name], domain)
            setattr(instance, setting.name, value)
            if hasattr(setting, "selected") and setting.selected in data:
                setattr(instance, setting.selected, data[setting.selected])

            if isinstance(value, list):
                excluded |= set(value)
            else:
                if setting.not_attribute:
                    excluded.add(value)

        if self.reservoir is not None:
            get_attribute = lambda name: context.attributes.get(name, None)
            get_meta = lambda name: context.metas.get(name, None)
            ll = [a for a in context.ordered_domain if a not in excluded and (
                self.attributes_in_res and get_attribute(a[0]) == a[1] or
                self.metas_in_res and get_meta(a[0]) == a[1])]
            setattr(widget, self.reservoir, ll)

    def encode_setting(self, context, setting, value):
        value = copy.copy(value)
        if isinstance(value, list):
            return value
        elif isinstance(setting, ContextSetting):
            if isinstance(value, str):
                if not setting.exclude_attributes and value in context.attributes:
                    return value, context.attributes[value]
                if not setting.exclude_metas and value in context.metas:
                    return value, context.metas[value]
            elif isinstance(value, Variable):
                return value.name, 100 + vartype(value)
        return value, -2

    def decode_setting(self, setting, value, domain=None):
        if isinstance(value, tuple):
            if 100 <= value[1]:
                if not domain:
                    raise ValueError("Cannot decode variable without domain")
                return domain[value[0]]
            return value[0]
        else:
            return value

    @staticmethod
    def _var_exists(setting, value, attributes, metas):
        if not isinstance(value, tuple) or len(value) != 2:
            return False

        attr_name, attr_type = value
        if 100 <= attr_type:
            attr_type -= 100
        return (not setting.exclude_attributes and
                attributes.get(attr_name, -1) == attr_type or
                not setting.exclude_metas and
                metas.get(attr_name, -1) == attr_type)

    def match(self, context, domain, attrs, metas):
        if (attrs, metas) == (context.attributes, context.metas):
            return self.PERFECT_MATCH

        matches = []
        try:
            for setting, data, _ in \
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
            return self.NO_MATCH

        matches.append((0, 0))
        matched, available = [sum(m) for m in zip(*matches)]

        return matched / available if available else 0.1

    def match_list(self, setting, value, context, attrs, metas):
        """Match a list of values with the given context.
        returns a tuple containing number of matched and all values.
        """
        matched = 0
        if hasattr(setting, 'selected'):
            selected = set(context.values.get(setting.selected, []))
        else:
            selected = set()

        for i, item in enumerate(value):
            if self.is_valid_item(setting, item, attrs, metas):
                matched += 1
            else:
                if setting.required == ContextSetting.REQUIRED:
                    raise IncompatibleContext()
                if setting.IF_SELECTED and i in selected:
                    raise IncompatibleContext()

        return matched, len(value)

    def match_value(self, setting, value, attrs, metas):
        """Match a single value """
        if value[1] < 0:
            return 0, 0

        if self._var_exists(setting, value, attrs, metas):
            return 1, 1
        else:
            raise IncompatibleContext()

    def is_valid_item(self, setting, item, attrs, metas):
        """Return True if given item can be used with attrs and metas

        Subclasses can override this method to checks data in alternative
        representations.
        """
        if not isinstance(item, tuple):
            return True
        return self._var_exists(setting, item, attrs, metas)


class IncompatibleContext(Exception):
    """Raised when a required variable in context is not available in data."""
    pass


class ClassValuesContextHandler(ContextHandler):
    """Context handler used for widgets that work with
    a single discrete variable"""

    def open_context(self, widget, classes):
        if isinstance(classes, Variable):
            if classes.is_discrete:
                classes = classes.values
            else:
                classes = None

        super().open_context(widget, classes)

    def new_context(self, classes):
        context = super().new_context()
        context.classes = classes
        return context

    def match(self, context, classes):
        if isinstance(classes, Variable) and classes.is_continuous:
            return (self.PERFECT_MATCH if context.classes is None
                    else self.NO_MATCH)
        else:
            return (self.PERFECT_MATCH if context.classes == classes
                    else self.NO_MATCH)


class PerfectDomainContextHandler(DomainContextHandler):
    """Context handler that matches a context only when
    the same domain is available.

    It uses a different encoding than the DomainContextHandler.
    """

    def new_context(self, domain, attributes, class_vars, metas):
        """Same as DomainContextHandler, but also store class_vars"""
        context = super().new_context(domain, attributes, metas)
        context.class_vars = class_vars
        return context

    def clone_context(self, old_context, *args):
        """Copy of context is always valid, since widgets are using
        the same domain."""
        context = self.new_context(*args)
        context.values = copy.deepcopy(old_context.values)
        return context

    def encode_domain(self, domain):
        """Encode domain into tuples (name, type)
        A tuple is returned for each of attributes, class_vars and metas.
        """

        if self.match_values == self.MATCH_VALUES_ALL:
            def _encode(attrs):
                return tuple((v.name, v.values if v.is_discrete else vartype(v))
                             for v in attrs)
        else:
            def _encode(attrs):
                return tuple((v.name, vartype(v)) for v in attrs)
        return (_encode(domain.attributes),
                _encode(domain.class_vars),
                _encode(domain.metas))

    def match(self, context, domain, attributes, class_vars, metas):
        """Context only matches when domains are the same"""

        return (self.PERFECT_MATCH
                if (context.attributes == attributes and
                    context.class_vars == class_vars and
                    context.metas == metas)
                else self.NO_MATCH)

    def encode_setting(self, context, setting, value):
        """Same as is domain context handler, but handles separately stored
        class_vars."""

        if isinstance(setting, ContextSetting) and isinstance(value, str):

            def _candidate_variables():
                if not setting.exclude_attributes:
                    yield from itertools.chain(context.attributes,
                                               context.class_vars)
                if not setting.exclude_metas:
                    yield from context.metas

            for aname, atype in _candidate_variables():
                if aname == value:
                    return value, atype

            return value, -1
        else:
            return super().encode_setting(context, setting, value)
