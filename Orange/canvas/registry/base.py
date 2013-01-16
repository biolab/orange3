"""
WidgetRegistry Base

"""

import logging
import bisect

from operator import attrgetter

from . import description

log = logging.getLogger(__name__)

# Registry hex version
VERSION_HEX = 0x000101


class WidgetRegistry(object):
    """A container for widget and category descriptions.

    This class is most often used with WidgetDiscovery class but can
    be used separately.

    >>> reg = WidgetRegistry()
    >>> file_desc = description.WidgetDescription.from_module(
    ...     "Orange.OrangeWidgets.Data.OWFile"
    ... )
    ...
    >>> reg.register_widget(file_desc)
    >>> reg.widgets()
    [...

    """

    def __init__(self, other=None):
        # A list of (category, widgets_list) tuples ordered by priority.
        self.registry = []
        # tuples from 'registry' indexed by name
        self._categories_dict = {}
        # WidgetDscriptions by qualified name
        self._widgets_dict = {}

        if other is not None:
            if not isinstance(other, WidgetRegistry):
                raise TypeError("Expected a 'WidgetRegistry' got %r." \
                                % type(other).__name__)

            self.registry = list(other.registry)
            self._categories_dict = dict(other._categories_dict)
            self._widgets_dict = dict(other._widgets_dict)

    def categories(self):
        """List all top level widget categories ordered by priority.
        """
        return [c for c, _ in self.registry]

    def category(self, name):
        """Return category with `name`.

        .. note:: Categories are identified by `name` attribute in contrast
            with widgets which are identified by `qualified_name`.

        """
        return self._categories_dict[name][0]

    def has_category(self, name):
        """Does the category with `name` exist in this registry.
        """
        return name in self._categories_dict

    def widgets(self, category=None):
        """List all widgets in a `category`. If no category is
        specified list widgets in all categories.

        """
        if category is None:
            categories = self.categories()
        elif isinstance(category, basestring):
            categories = [self.category(category)]
        else:
            categories = [category]

        widgets = []
        for cat in categories:
            if isinstance(cat, basestring):
                cat = self.category(cat)
            cat_widgets = self._categories_dict[cat.name][1]
            widgets.extend(sorted(cat_widgets,
                                  key=attrgetter("priority")))
        return widgets

    def widget(self, qualified_name):
        """Return widget description for `qualified_name`.
        Raise KeyError if the description does not exist.

        """
        return self._widgets_dict[qualified_name]

    def has_widget(self, qualified_name):
        """Does the widget with `qualified_name` exist in this registry.
        """
        return qualified_name in self._widgets_dict

    def register_widget(self, desc):
        """Register a widget by its description.
        """
        if not isinstance(desc, description.WidgetDescription):
            raise TypeError("Expected a 'WidgetDescription' got %r." \
                            % type(desc).__name__)

        if self.has_widget(desc.qualified_name):
            raise ValueError("%r already exists in the registry." \
                             % desc.qualified_name)

        category = desc.category
        if category is None:
            category = "Unspecified"

        if self.has_category(category):
            cat_desc = self.category(category)
        else:
            log.warning("Creating a default category %r.", category)
            cat_desc = description.CategoryDescription(name=category)
            self.register_category(cat_desc)

        self._insert_widget(cat_desc, desc)

    def register_category(self, desc):
        """Register category by its description.

        .. note:: It is always best to register the category
            before the widgets belonging to it.

        """
        if not isinstance(desc, description.CategoryDescription):
            raise TypeError("Expected a 'CategoryDescription' got %r." \
                            % type(desc).__name__)

        name = desc.name
        if not name:
            log.warning("Creating a default category name.")
            name = "default"

        if any(name == c.name for c in self.categories()):
            log.warning("A category with %r name already exists" % name)
            return

        self._insert_category(desc)

    def _insert_category(self, desc):
        """Insert category description into 'registry' list
        """
        priority = desc.priority
        priorities = [c.priority for c, _ in self.registry]
        insertion_i = bisect.bisect_right(priorities, priority)

        item = (desc, [])
        self.registry.insert(insertion_i, item)
        self._categories_dict[desc.name] = item

    def _insert_widget(self, category, desc):
        """Insert widget description `desc` into `category`.
        """
        assert(isinstance(category, description.CategoryDescription))
        _, widgets = self._categories_dict[category.name]

        priority = desc.priority
        priorities = [w.priority for w in widgets]
        insertion_i = bisect.bisect_right(priorities, priority)
        widgets.insert(insertion_i, desc)
        self._widgets_dict[desc.qualified_name] = desc
