import copy
from itertools import chain

from orangecanvas.registry import WidgetDescription
from orangecanvas.registry import discovery


def widget_desc_from_module(module):
    """
    Get the widget description from a module.

    The module is inspected for classes that have a method
    `get_widget_description`. The function calls this method and expects
    a dictionary, which is used as keyword arguments for
    :obj:`WidgetDescription`. This method also converts all signal types
    into qualified names to prevent import problems when cached
    descriptions are unpickled (the relevant code using this lists should
    be able to handle missing types better).

    Parameters
    ----------
    module (`module` or `str`): a module to inspect

    Returns
    -------
    An instance of :obj:`WidgetDescription`
    """
    if isinstance(module, str):
        module = __import__(module, fromlist=[""])

    if module.__package__:
        package_name = module.__package__.rsplit(".", 1)[-1]
    else:
        package_name = None

    default_cat_name = package_name if package_name else ""

    for widget_class in module.__dict__.values():
        if not hasattr(widget_class, "get_widget_description"):
            continue
        description = widget_class.get_widget_description()
        if description is None:
            continue

        description = WidgetDescription(**description)
        description.package = module.__package__
        description.category = widget_class.category or default_cat_name
        return description

    raise discovery.WidgetSpecificationError


class WidgetDiscovery(discovery.WidgetDiscovery):

    def widget_description(self, module, widget_name=None, category_name=None,
                           distribution=None):
        """
        Return widget description from a module.
        """
        module = discovery.asmodule(module)
        desc = widget_desc_from_module(module)

        if widget_name is not None:
            desc.name = widget_name

        if category_name is not None:
            desc.category = category_name

        if distribution is not None:
            desc.project_name = distribution.project_name

        return desc
