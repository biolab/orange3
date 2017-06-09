"""
Widget meta description classes
===============================

"""

import sys
import copy

# Exceptions
from itertools import chain


class DescriptionError(Exception):
    pass


class WidgetSpecificationError(DescriptionError):
    pass


class SignalSpecificationError(DescriptionError):
    pass


class CategorySpecificationError(DescriptionError):
    pass


###############
# Channel flags
###############

# A single signal
Single = 2

# Multiple signal (more then one input on the channel)
Multiple = 4

# Default signal (default or primary input/output)
Default = 8
NonDefault = 16

# Explicit - only connected if specifically requested or the only possibility
Explicit = 32

# Dynamic type output signal
Dynamic = 64


# Input/output signal (channel) description


class InputSignal(object):
    """
    Description of an input channel.

    Parameters
    ----------
    name : str
        Name of the channel.
    type : str or `type`
        Type of the accepted signals.
    handler : str
        Name of the handler method for the signal.
    flags : int, optional
        Channel flags.
    id : str
        A unique id of the input signal.
    doc : str, optional
        A docstring documenting the channel.
    replaces : List[str]
        A list of names this input replaces.
    """
    def __init__(self, name, type, handler, flags=Single + NonDefault,
                 id=None, doc=None, replaces=()):
        self.name = name
        self.type = type
        self.handler = handler
        self.id = id
        self.doc = doc
        self.replaces = list(replaces)

        if not (flags & Single or flags & Multiple):
            flags += Single

        if not (flags & Default or flags & NonDefault):
            flags += NonDefault

        self.single = flags & Single
        self.default = flags & Default
        self.explicit = flags & Explicit
        self.flags = flags

    def __str__(self):
        fmt = ("{0.__name__}(name={name!r}, type={type!s}, "
               "handler={handler}, ...)")
        return fmt.format(type(self), **self.__dict__)

    __repr__ = __str__


class OutputSignal(object):
    """
    Description of an output channel.

    Parameters
    ----------
    name : str
        Name of the channel.
    type : str or `type`
        Type of the output signals.
    flags : int, optional
        Channel flags.
    id : str
        A unique id of the output signal.
    doc : str, optional
        A docstring documenting the channel.
    replaces : List[str]
        A list of names this output replaces.
    """
    def __init__(self, name, type, flags=Single + NonDefault,
                 id=None, doc=None, replaces=()):
        self.name = name
        self.type = type
        self.id = id
        self.doc = doc
        self.replaces = list(replaces)

        if not (flags & Single or flags & Multiple):
            flags += Single

        if not (flags & Default or flags & NonDefault):
            flags += NonDefault

        self.single = flags & Single
        self.default = flags & Default
        self.explicit = flags & Explicit
        self.dynamic = flags & Dynamic
        self.flags = flags

        if self.dynamic and not self.single:
            raise SignalSpecificationError(
                "Output signal can not be 'Multiple' and 'Dynamic'."
                )

    def __str__(self):
        fmt = ("{0.__name__}(name={name!r}, type={type!s}, "
               "...)")
        return fmt.format(type(self), **self.__dict__)

    __repr__ = __str__


class WidgetDescription(object):
    """
    Description of a widget.

    Parameters
    ----------
    name : str
        A human readable name of the widget.
    id : str
        A unique identifier of the widget (in most situations this should
        be the full module name).
    category : str, optional
        A name of the category in which this widget belongs.
    version : str, optional
        Version of the widget. By default the widget inherits the project
        version.
    description : str, optional
        A short description of the widget, suitable for a tool tip.
    qualified_name : str
        A qualified name (import name) of the class implementing the widget.
    package : str, optional
        A package name where the widget is implemented.
    project_name : str, optional
        The distribution name that provides the widget.
    inputs : list of :class:`InputSignal`, optional
        A list of input channels provided by the widget.
    outputs : list of :class:`OutputSignal`, optional
        A list of output channels provided by the widget.
    help : str, optional
        URL or an Resource template of a detailed widget help page.
    help_ref : str, optional
        A text reference id that can be used to identify the help
        page, for instance an intersphinx reference.
    keywords : list-of-str, optional
        A list of keyword phrases.
    priority : int, optional
        Widget priority (the order of the widgets in a GUI presentation).
    icon : str, optional
        A filename of the widget icon (in relation to the package).
    background : str, optional
        Widget's background color (in the canvas GUI).
    replaces : list-of-str, optional
        A list of `id`s this widget replaces (optional).

    """
    def __init__(self, name, id, category=None, version=None,
                 description=None,
                 qualified_name=None, package=None, project_name=None,
                 inputs=(), outputs=(),
                 help=None, help_ref=None, url=None, keywords=None,
                 priority=sys.maxsize,
                 icon=None, background=None,
                 replaces=None):

        if not qualified_name:
            # TODO: Should also check that the name is real.
            raise ValueError("'qualified_name' must be supplied.")

        self.name = name
        self.id = id
        self.category = category
        self.version = version
        self.description = description
        self.qualified_name = qualified_name
        self.package = package
        self.project_name = project_name
        self.inputs = inputs
        self.outputs = outputs
        self.help = help
        self.help_ref = help_ref
        self.url = url
        self.keywords = keywords
        self.priority = priority
        self.icon = icon
        self.background = background
        self.replaces = replaces

    def __str__(self):
        return ("WidgetDescription(name=%(name)r, id=%(id)r), "
                "category=%(category)r, ...)") % self.__dict__

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_module(cls, module):
        # False positive for widget_class: undefined var is prevented by
        # raising exception in else after the for
        # pylint: disable=undefined-loop-variable
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
            description = copy.deepcopy(description)
            for s in chain(description["inputs"], description["outputs"]):
                s.type = "%s.%s" % (s.type.__module__, s.type.__name__)
            description = WidgetDescription(**description)

            description.package = module.__package__
            description.category = widget_class.category or default_cat_name
            return description

        raise WidgetSpecificationError

class CategoryDescription(object):
    """
    Description of a widget category.

    Parameters
    ----------

    name : str
        A human readable name.
    version : str, optional
        Version string.
    description : str, optional
        A short description of the category, suitable for a tool tip.
    long_description : str, optional
        A longer description.
    qualified_name : str,
        Qualified name
    project_name : str
        A project name providing the category.
    priority : int
        Priority (order in the GUI).
    icon : str
        An icon filename (a resource name retrievable using `pkg_resources`
        relative to `qualified_name`).
    background : str
        An background color for widgets in this category.

    """
    def __init__(self, name=None, version=None,
                 description=None, long_description=None,
                 qualified_name=None, package=None,
                 project_name=None,
                 url=None, help=None, keywords=None,
                 widgets=None, priority=sys.maxsize,
                 icon=None, background=None):

        self.name = name
        self.version = version
        self.description = description
        self.long_description = long_description
        self.qualified_name = qualified_name
        self.package = package
        self.project_name = project_name
        self.url = url
        self.help = help
        self.keywords = keywords
        self.widgets = widgets or []
        self.priority = priority
        self.icon = icon
        self.background = background

    def __str__(self):
        return "CategoryDescription(name=%(name)r, ...)" % self.__dict__

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_package(cls, package):
        """
        Get the CategoryDescription from a package.

        Parameters
        ----------
        package : `module` or `str`
            A package containing the category.

        """
        if isinstance(package, str):
            package = __import__(package, fromlist=[""])
        package_name = package.__name__
        qualified_name = package_name
        default_name = package_name.rsplit(".", 1)[-1]

        name = getattr(package, "NAME", default_name)
        description = getattr(package, "DESCRIPTION", None)
        long_description = getattr(package, "LONG_DESCRIPTION", None)
        url = getattr(package, "URL", None)
        help = getattr(package, "HELP", None)
        keywords = getattr(package, "KEYWORDS", None)
        widgets = getattr(package, "WIDGETS", None)
        priority = getattr(package, "PRIORITY", sys.maxsize - 1)
        icon = getattr(package, "ICON", None)
        background = getattr(package, "BACKGROUND", None)

        if priority == sys.maxsize - 1 and name.lower() == "prototypes":
            priority = sys.maxsize

        return CategoryDescription(
            name=name,
            qualified_name=qualified_name,
            description=description,
            long_description=long_description,
            help=help,
            url=url,
            keywords=keywords,
            widgets=widgets,
            priority=priority,
            icon=icon,
            background=background)
