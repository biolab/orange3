"""
Qt Model classes for widget registry.

"""
import bisect

from xml.sax.saxutils import escape
from urllib.parse import urlencode

from PyQt4.QtGui import (
    QStandardItemModel, QStandardItem, QColor, QBrush, QAction
)

from PyQt4.QtCore import QObject, Qt, QVariant
from PyQt4.QtCore import pyqtSignal as Signal

from .discovery import WidgetDiscovery
from .description import WidgetDescription, CategoryDescription
from .base import WidgetRegistry

from ..resources import icon_loader

from . import cache, NAMED_COLORS, DEFAULT_COLOR


class QtWidgetDiscovery(QObject, WidgetDiscovery):
    """
    Qt interface class for widget discovery.
    """
    # Discovery has started
    discovery_start = Signal()
    # Discovery has finished
    discovery_finished = Signal()
    # Processing widget with name
    discovery_process = Signal(str)
    # Found a widget with description
    found_widget = Signal(WidgetDescription)
    # Found a category with description
    found_category = Signal(CategoryDescription)

    def __init__(self, parent=None, registry=None, cached_descriptions=None):
        QObject.__init__(self, parent)
        WidgetDiscovery.__init__(self, registry, cached_descriptions)

    def run(self, entry_points_iter):
        self.discovery_start.emit()
        WidgetDiscovery.run(self, entry_points_iter)
        self.discovery_finished.emit()

    def handle_widget(self, description):
        self.discovery_process.emit(description.name)
        self.found_widget.emit(description)

    def handle_category(self, description):
        self.found_category.emit(description)


class QtWidgetRegistry(QObject, WidgetRegistry):
    """
    A QObject wrapper for `WidgetRegistry`

    A QStandardItemModel instance containing the widgets in
    a tree (of depth 2). The items in a model can be quaries using standard
    roles (DisplayRole, BackgroundRole, DecorationRole ToolTipRole).
    They also have QtWidgetRegistry.CATEGORY_DESC_ROLE,
    QtWidgetRegistry.WIDGET_DESC_ROLE, which store Category/WidgetDescription
    respectfully. Furthermore QtWidgetRegistry.WIDGET_ACTION_ROLE stores an
    default QAction which can be used for widget creation action.

    """

    CATEGORY_DESC_ROLE = Qt.UserRole + 1
    """Category Description Role"""

    WIDGET_DESC_ROLE = Qt.UserRole + 2
    """Widget Description Role"""

    WIDGET_ACTION_ROLE = Qt.UserRole + 3
    """Widget Action Role"""

    BACKGROUND_ROLE = Qt.UserRole + 4
    """Background color for widget/category in the canvas
    (different from Qt.BackgroundRole)
    """

    category_added = Signal(str, CategoryDescription)
    """signal: category_added(name: str, desc: CategoryDescription)
    """

    widget_added = Signal(str, str, WidgetDescription)
    """signal widget_added(category_name: str, widget_name: str,
                           desc: WidgetDescription)
    """

    reset = Signal()
    """signal: reset()
    """

    def __init__(self, other_or_parent=None, parent=None):
        if isinstance(other_or_parent, QObject) and parent is None:
            parent, other_or_parent = other_or_parent, None
        QObject.__init__(self, parent)
        WidgetRegistry.__init__(self, other_or_parent)

        # Should  the QStandardItemModel be subclassed?
        self.__item_model = QStandardItemModel(self)

        for i, desc in enumerate(self.categories()):
            cat_item = self._cat_desc_to_std_item(desc)
            self.__item_model.insertRow(i, cat_item)

            for j, wdesc in enumerate(self.widgets(desc.name)):
                widget_item = self._widget_desc_to_std_item(wdesc, desc)
                cat_item.insertRow(j, widget_item)

    def model(self):
        """
        Return the widget descriptions in a Qt Item Model instance
        (QStandardItemModel).

        .. note:: The model should not be modified outside of the registry.

        """
        return self.__item_model

    def item_for_widget(self, widget):
        """Return the QStandardItem for the widget.
        """
        if isinstance(widget, str):
            widget = self.widget(widget)
        cat = self.category(widget.category)
        cat_ind = self.categories().index(cat)
        cat_item = self.model().item(cat_ind)
        widget_ind = self.widgets(cat).index(widget)
        return cat_item.child(widget_ind)

    def action_for_widget(self, widget):
        """
        Return the QAction instance for the widget (can be a string or
        a WidgetDescription instance).

        """
        item = self.item_for_widget(widget)
        return item.data(self.WIDGET_ACTION_ROLE)

    def create_action_for_item(self, item):
        """
        Create a QAction instance for the widget description item.
        """
        name = item.text()
        tooltip = item.toolTip()
        whatsThis = item.whatsThis()
        icon = item.icon()
        if icon:
            action = QAction(icon, name, self, toolTip=tooltip,
                             whatsThis=whatsThis,
                             statusTip=name)
        else:
            action = QAction(name, self, toolTip=tooltip,
                             whatsThis=whatsThis,
                             statusTip=name)

        widget_desc = item.data(self.WIDGET_DESC_ROLE)
        action.setData(widget_desc)
        action.setProperty("item", item)
        return action

    def _insert_category(self, desc):
        """
        Override to update the item model and emit the signals.
        """
        priority = desc.priority
        priorities = [c.priority for c, _ in self.registry]
        insertion_i = bisect.bisect_right(priorities, priority)

        WidgetRegistry._insert_category(self, desc)

        cat_item = self._cat_desc_to_std_item(desc)
        self.__item_model.insertRow(insertion_i, cat_item)

        self.category_added.emit(desc.name, desc)

    def _insert_widget(self, category, desc):
        """
        Override to update the item model and emit the signals.
        """
        assert(isinstance(category, CategoryDescription))
        categories = self.categories()
        cat_i = categories.index(category)
        _, widgets = self._categories_dict[category.name]
        priorities = [w.priority for w in widgets]
        insertion_i = bisect.bisect_right(priorities, desc.priority)

        WidgetRegistry._insert_widget(self, category, desc)

        cat_item = self.__item_model.item(cat_i)
        widget_item = self._widget_desc_to_std_item(desc, category)

        cat_item.insertRow(insertion_i, widget_item)

        self.widget_added.emit(category.name, desc.name, desc)

    def _cat_desc_to_std_item(self, desc):
        """
        Create a QStandardItem for the category description.
        """
        item = QStandardItem()
        item.setText(desc.name)

        if desc.icon:
            icon = desc.icon
        else:
            icon = "icons/default-category.svg"

        icon = icon_loader.from_description(desc).get(icon)
        item.setIcon(icon)

        if desc.background:
            background = desc.background
        else:
            background = DEFAULT_COLOR

        background = NAMED_COLORS.get(background, background)

        brush = QBrush(QColor(background))
        item.setData(brush, self.BACKGROUND_ROLE)

        tooltip = desc.description if desc.description else desc.name

        item.setToolTip(tooltip)
        item.setFlags(Qt.ItemIsEnabled)
        item.setData(desc, self.CATEGORY_DESC_ROLE)
        return item

    def _widget_desc_to_std_item(self, desc, category):
        """
        Create a QStandardItem for the widget description.
        """
        item = QStandardItem(desc.name)
        item.setText(desc.name)

        if desc.icon:
            icon = desc.icon
        else:
            icon = "icons/default-widget.svg"

        icon = icon_loader.from_description(desc).get(icon)
        item.setIcon(icon)

        # This should be inherited from the category.
        background = None
        if desc.background:
            background = desc.background
        elif category.background:
            background = category.background
        else:
            background = DEFAULT_COLOR

        if background is not None:
            background = NAMED_COLORS.get(background, background)
            brush = QBrush(QColor(background))
            item.setData(brush, self.BACKGROUND_ROLE)

        tooltip = tooltip_helper(desc)
        style = "ul { margin-top: 1px; margin-bottom: 1px; }"
        tooltip = TOOLTIP_TEMPLATE.format(style=style, tooltip=tooltip)
        item.setToolTip(tooltip)
        item.setWhatsThis(whats_this_helper(desc))
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        item.setData(desc, self.WIDGET_DESC_ROLE)

        # Create the action for the widget_item
        action = self.create_action_for_item(item)
        item.setData(action, self.WIDGET_ACTION_ROLE)
        return item


TOOLTIP_TEMPLATE = """\
<html>
<head>
<style type="text/css">
{style}
</style>
</head>
<body>
{tooltip}
</body>
</html>
"""


def tooltip_helper(desc):
    """Widget tooltip construction helper.

    """
    tooltip = []
    tooltip.append("<b>{name}</b>".format(name=escape(desc.name)))

    if desc.project_name and desc.project_name != "Orange":
        tooltip[0] += " (from {0})".format(desc.project_name)

    if desc.description:
        tooltip.append("{0}".format(
                            escape(desc.description)))

    inputs_fmt = "<li>{name}</li>"

    if desc.inputs:
        inputs = "".join(inputs_fmt.format(name=inp.name)
                         for inp in desc.inputs)
        tooltip.append("Inputs:<ul>{0}</ul>".format(inputs))
    else:
        tooltip.append("No inputs")

    if desc.outputs:
        outputs = "".join(inputs_fmt.format(name=out.name)
                          for out in desc.outputs)
        tooltip.append("Outputs:<ul>{0}</ul>".format(outputs))
    else:
        tooltip.append("No outputs")

    return "<hr/>".join(tooltip)


def whats_this_helper(desc, include_more_link=False):
    """
    A `What's this` text construction helper. If `include_more_link` is
    True then the text will include a `more...` link.

    """
    title = desc.name
    help_url = desc.help

    if not help_url:
        help_url = "help://search?" + urlencode({"id": desc.id})

    description = desc.description
    long_description = desc.long_description

    template = ["<h3>{0}</h3>".format(escape(title))]

    if description:
        template.append("<p>{0}</p>".format(escape(description)))

    if long_description:
        template.append("<p>{0}</p>".format(escape(long_description[:100])))

    if help_url and include_more_link:
        template.append("<a href='{0}'>more...</a>".format(escape(help_url)))

    return "\n".join(template)


def run_discovery(entry_points_iter, cached=False):
    """
    Run the default discovery and return an instance of
    :class:`QtWidgetRegistry`.

    """
    reg_cache = {}
    if cached:
        reg_cache = cache.registry_cache()

    discovery = QtWidgetDiscovery(cached_descriptions=reg_cache)
    registry = QtWidgetRegistry()
    discovery.found_category.connect(registry.register_category)
    discovery.found_widget.connect(registry.register_widget)
    discovery.run()
    if cached:
        cache.save_registry_cache(reg_cache)
    return registry
