from collections import OrderedDict
import time
import itertools
import time
from PyQt4.QtCore import QByteArray, QBuffer, QIODevice
from Orange.widgets.io import PngFormat


def plural(s, number, suffix="s"):
    """
    Insert the number into the string, and make plural where marked, if needed.

    The string should use `{number}` to mark the place(s) where the number is
    inserted and `{s}` where an "s" needs to be added if the number is not 1.

    For instance, a string could be "I saw {number} dog{s} in the forest".

    Argument `suffix` can be used for some forms or irregular plural, like:

        plural("I saw {number} fox{s} in the forest", x, "es")
        plural("I say {number} child{s} in the forest", x, "ren")

    :param s: string
    :type s: str
    :param number: number
    :type number: int
    :param suffix: the suffix to use; default is "s"
    :type suffix: str
    :rtype: str
    """
    return s.format(number=number, s=suffix if number % 100 != 1 else "")


def plural_w(s, number, suffix="s", capitalize=False):
    """
    Insert the number into the string, and make plural where marked, if needed.

    If the number is smaller or equal to ten, a word is used instead of a
    numeric representation.

    The string should use `{number}` to mark the place(s) where the number is
    inserted and `{s}` where an "s" needs to be added if the number is not 1.

    For instance, a string could be "I saw {number} dog{s} in the forest".

    Argument `suffix` can be used for some forms or irregular plural, like:

        plural("I saw {number} fox{s} in the forest", x, "es")
        plural("I say {number} child{s} in the forest", x, "ren")

    :param s: string
    :type s: str
    :param number: number
    :type number: int
    :param suffix: the suffix to use; default is "s"
    :type suffix: str
    :rtype: str
    """
    numbers = ("zero", "one", "two", "three", "four", "five", "six", "seven",
               "nine", "ten")
    number_str = numbers[number] if number < len(numbers) else str(number)
    if capitalize:
        number_str = number_str.capitalize()
    return s.format(number=number_str, s=suffix if number % 100 != 1 else "")


def clip_string(s, limit=1000, sep=None):
    """
    Clip a string at a given character and add "..." if the string was clipped.

    If a separator is specified, the string is not clipped at the given limit
    but after the last occurence of the separator below the limit.

    :param s: string to clip
    :type s: str
    :param limit: number of characters to retain (including "...")
    :type limit: int
    :param sep: separator
    :type sep: str
    :rtype: str
    """
    if len(s) < limit:
        return s
    s = s[:limit - 3]
    if sep is None:
        return s
    sep_pos = s.rfind(sep)
    if sep_pos == -1:
        return s
    return s[:sep_pos + len(sep)] + "..."


def clipped_list(items, limit=1000, less_lookups=False, total_min=10, total=""):
    """
    Return a clipped comma-separated representation of the list.

    If `less_lookups` is `True`, clipping will use a generator across the first
    `(limit + 2) // 3` items only, which suffices even if each item is only a
    single character long. This is useful in case when retrieving items is
    expensive, while it is generally slower.

    If there are at least `total_lim` items, and argument `total` is present,
    the string `total.format(len(items))` is added to the end of string.
    Argument `total` can be, for instance `"(total: {} variables)"`.

    If `total` is given, `s` cannot be a generator.

    :param items: list
    :type items: list or another iterable object
    :param limit: number of characters to retain (including "...")
    :type limit: int
    :param total_min: the minimal number of items that triggers adding `total`
    :type total_min: int
    :param total: the string that is added if `len(items) >= total_min`
    :type total: str
    :param less_lookups: minimize the number of lookups
    :type less_lookups: bool
    :return:
    """
    if less_lookups:
        s = ", ".join(itertools.islice(items, (limit + 2) // 3))
    else:
        s = ", ".join(items)
    s = clip_string(s, limit, ", ")
    if total and len(items) >= total_min:
        s += " " + total.format(len(items))
    return s


def get_html_section(name):
    """
    Return a new section as HTML, with the given name and a time stamp.

    :param name: section name
    :type name: str
    :rtype: str
    """
    datetime = time.strftime("%a %b %d %y, %H:%M:%S")
    return "<h1>{} <span class='timestamp'>{}</h1>".format(name, datetime)


def get_html_subsection(name):
    """
    Return a subsection as HTML, with the given name

    :param name: subsection name
    :type name: str
    :rtype: str
    """
    return "<h2>{}</h2>".format(name)


def render_items(items, order=None, exclude=None):
    """
    Render a list of pairs or an (ordered) dictionary as a HTML list.

    The function skips the items whose values are `None` or `False`, and items
    that are listed in `exclude`.

    If argument `order` is present, the items must be given as dictionary.

    :param items: a list or dictionary of items
    :type items: dict or list
    :param order: a list of item names to render
    :type order: list
    :param exclude: a list or set of items to exclude
    :type exclude: list or set
    :return: rendered content
    :rtype: str
    """
    if order is not None:
        gen = ((key, items[key]) for key in order)
    elif isinstance(items, dict):
        gen = items.items()
    else:
        gen = items
    return "<ul>" + "".join(
        "<b>{}:</b> {}</br>".format(key, value) for key, value in gen
        if (exclude is None or key not in exclude) and
        (value is not None and value is not False)
    ) + "</ul>"


def get_html_img(scene):
    byte_array = QByteArray()
    filename = QBuffer(byte_array)
    filename.open(QIODevice.WriteOnly)
    writer = PngFormat()
    writer.write(filename, scene)
    img_encoded = byte_array.toBase64().data().decode("utf-8")
    return "<ul><img src='data:image/png;base64,%s'/></ul>" % img_encoded


def describe_domain(domain):
    """
    Return an :obj:`OrderedDict` describing a domain

    Description contains keys "Features", "Meta attributes" and "Targets"
    with the corresponding clipped lists of names. If the domain contains no
    meta attributes or targets, the value is `False`, which prevents it from
    being rendered by :obj:`~Orange.canvas.report.render_items`.

    :param domain: domain
    :type domain: Orange.data.Domain
    :rtype: OrderedDict
    """
    def clip_attrs(items, s):
        return clipped_list([a.name for a in items], 1000,
                            total_min=10, total=" (total: {{}} {})".format(s))

    return OrderedDict(
        [("Features", clip_attrs(domain.attributes, "features")),
         ("Meta attributes", bool(domain.metas) and
             clip_attrs(domain.metas, "meta attributes")),
         ("Target", bool(domain.class_vars) and
             clip_attrs(domain.class_vars, "targets variables"))])


def describe_data(data):
    """
    Return an :obj:`OrderedDict` describing the data

    Description contains keys "Data instances" (with the number of instances)
    and "Features", "Meta attributes" and "Targets" with the corresponding
    clipped lists of names. If the domain contains no meta attributes or
    targets, the value is `False`, which prevents it from being rendered.

    :param data: data
    :type data: Orange.data.Table
    :rtype: OrderedDict
    """
    if data is None:
        return OrderedDict()
    items = OrderedDict()
    items["Data instances"] = len(data)
    items.update(describe_domain(data.domain))
    return items


def describe_data_brief(data):
    domain = data.domain
    items = OrderedDict([
        ("Data instances", len(data)),
        ("Features", len(domain.attributes) or "None"),
        ("Meta attributes", len(domain.metas) or "None")])
    if domain.has_discrete_class:
        items["Target"] = "Class '{}'".format(domain.class_var.name)
    elif domain.has_continuous_class:
        items["Target"] = "Numeric variable '{}'". \
            format(domain.class_var.name)
    elif domain.class_vars:
        items["Targets"] = len(domain.class_vars)
    else:
        items["Targets"] = False
    return items
