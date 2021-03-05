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

# Overriden methods in these classes add arguments
# pylint: disable=arguments-differ,unused-argument,no-value-for-parameter

import copy
import itertools
import logging
import warnings

from orangewidget.settings import (
    Setting, SettingProvider, SettingsHandler, ContextSetting,
    ContextHandler, Context, IncompatibleContext, SettingsPrinter,
    rename_setting, widget_settings_dir
)
from orangewidget.settings import _apply_setting

from Orange.data import Domain, Variable
from Orange.util import OrangeDeprecationWarning
from Orange.widgets.utils import vartype

log = logging.getLogger(__name__)

__all__ = [
    # re-exported from orangewidget.settings
    "Setting", "SettingsHandler", "SettingProvider",
    "ContextSetting", "Context", "ContextHandler", "IncompatibleContext",
    "rename_setting", "widget_settings_dir",
    # defined here
    "DomainContextHandler", "PerfectDomainContextHandler",
    "ClassValuesContextHandler", "SettingsPrinter",
    "migrate_str_to_variable",
]


class DomainContextHandler(ContextHandler):
    """Context handler for widgets with settings that depend on
    the input dataset. Suitable settings are selected based on the
    data domain."""

    MATCH_VALUES_NONE, MATCH_VALUES_CLASS, MATCH_VALUES_ALL = range(3)

    def __init__(self, *, match_values=0, first_match=True, **kwargs):
        super().__init__()
        self.match_values = match_values
        self.first_match = first_match

        for name in kwargs:
            warnings.warn(
                "{} is not a valid parameter for DomainContextHandler"
                .format(name), OrangeDeprecationWarning
            )

    def encode_domain(self, domain):
        """
        domain: Orange.data.domain to encode
        return: dict mapping attribute name to type or list of values
                (based on the value of self.match_values attribute)
        """

        match = self.match_values
        encode = self.encode_variables
        if match == self.MATCH_VALUES_CLASS:
            attributes = encode(domain.attributes, False)
            attributes.update(encode(domain.class_vars, True))
        else:
            attributes = encode(domain.variables, match == self.MATCH_VALUES_ALL)

        metas = encode(domain.metas, match == self.MATCH_VALUES_ALL)

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
        return context

    def open_context(self, widget, domain):
        if domain is None:
            return
        if not isinstance(domain, Domain):
            domain = domain.domain
        super().open_context(widget, domain, *self.encode_domain(domain))

    def filter_value(self, setting, data, domain, attrs, metas):
        value = data.get(setting.name, None)
        if isinstance(value, list):
            new_value = [item for item in value
                         if self.is_valid_item(setting, item, attrs, metas)]
            data[setting.name] = new_value
        elif isinstance(value, dict):
            new_value = {item: val for item, val in value.items()
                         if self.is_valid_item(setting, item, attrs, metas)}
            data[setting.name] = new_value
        elif self.is_encoded_var(value) \
                and not self._var_exists(setting, value, attrs, metas):
            del data[setting.name]

    @staticmethod
    def encode_variable(var):
        return var.name, 100 + vartype(var)

    @classmethod
    def encode_setting(cls, context, setting, value):
        if isinstance(value, list):
            if all(e is None or isinstance(e, Variable) for e in value) \
                    and any(e is not None for e in value):
                return ([None if e is None else cls.encode_variable(e)
                         for e in value],
                        -3)
            else:
                return copy.copy(value)

        elif isinstance(value, dict) \
                and all(isinstance(e, Variable) for e in value):
            return ({cls.encode_variable(e): val for e, val in value.items()},
                    -4)

        if isinstance(value, Variable):
            if isinstance(setting, ContextSetting):
                return cls.encode_variable(value)
            else:
                raise ValueError("Variables must be stored as ContextSettings; "
                                 f"change {setting.name} to ContextSetting.")

        return copy.copy(value), -2

    # backward compatibility, pylint: disable=keyword-arg-before-vararg
    def decode_setting(self, setting, value, domain=None, *args):
        def get_var(name):
            if domain is None:
                raise ValueError("Cannot decode variable without domain")
            return domain[name]

        if isinstance(value, tuple):
            data, dtype = value
            if dtype == -3:
                return[None if name_type is None else get_var(name_type[0])
                       for name_type in data]
            if dtype == -4:
                return {get_var(name): val for (name, _), val in data.items()}
            if dtype >= 100:
                return get_var(data)
            return value[0]
        else:
            return value

    @classmethod
    def _var_exists(cls, setting, value, attributes, metas):
        if not cls.is_encoded_var(value):
            return False

        attr_name, attr_type = value
        # attr_type used to be either 1-4 for variables stored as string
        # settings, and 101-104 for variables stored as variables. The former is
        # no longer supported, but we play it safe and still handle both here.
        attr_type %= 100
        return (not setting.exclude_attributes and
                attributes.get(attr_name, -1) == attr_type or
                not setting.exclude_metas and
                metas.get(attr_name, -1) == attr_type)

    def match(self, context, domain, attrs, metas):
        if context.attributes == attrs and context.metas == metas:
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
                # type check is a (not foolproof) check in case of a pair that
                # would, by conincidence, have -3 or -4 as the second element
                elif isinstance(value, tuple) and len(value) == 2 \
                       and (value[1] == -3 and isinstance(value[0], list)
                            or (value[1] == -4 and isinstance(value[0], dict))):
                    matches.append(self.match_list(setting, value[0], context,
                                                   attrs, metas))
                elif value is not None:
                    matches.append(
                        self.match_value(setting, value, attrs, metas))
        except IncompatibleContext:
            return self.NO_MATCH

        if self.first_match and matches and sum(m[0] for m in matches):
            return self.MATCH

        matches.append((0, 0))
        matched, available = [sum(m) for m in zip(*matches)]
        return matched / available if available else 0.1

    def match_list(self, setting, value, context, attrs, metas):
        """Match a list of values with the given context.
        returns a tuple containing number of matched and all values.
        """
        matched = 0
        for item in value:
            if self.is_valid_item(setting, item, attrs, metas):
                matched += 1
            elif setting.required == ContextSetting.REQUIRED:
                raise IncompatibleContext()
        return matched, len(value)

    def match_value(self, setting, value, attrs, metas):
        """Match a single value """
        if value[1] < 0:
            return 0, 0

        if self._var_exists(setting, value, attrs, metas):
            return 1, 1
        elif setting.required == setting.OPTIONAL:
            return 0, 1
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

    @staticmethod
    def is_encoded_var(value):
        return isinstance(value, tuple) \
            and len(value) == 2 \
            and isinstance(value[0], str) and isinstance(value[1], int) \
            and value[1] >= 0

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
            # variable.values used to be a list, and so were context.classes
            # cast to tuple for compatibility with past contexts
            if context.classes is not None and tuple(context.classes) == classes:
                return self.PERFECT_MATCH
            else:
                return self.NO_MATCH


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
                return tuple((v.name, list(v.values) if v.is_discrete else vartype(v))
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


def migrate_str_to_variable(settings, names=None, none_placeholder=None):
    """
    Change variables stored as `(str, int)` to `(Variable, int)`.

    Args:
        settings (Context): context that is being migrated
        names (sequence): names of settings to be migrated. If omitted,
            all settings with values `(str, int)` are migrated.
    """
    def _fix(name):
        var, vtype = settings.values[name]
        if 0 <= vtype <= 100:
            settings.values[name] = (var, 100 + vtype)
        elif var == none_placeholder and vtype == -2:
            settings.values[name] = None

    if names is None:
        for name, setting in settings.values.items():
            if DomainContextHandler.is_encoded_var(setting):
                _fix(name)
    elif isinstance(names, str):
        _fix(names)
    else:
        for name in names:
            _fix(name)
