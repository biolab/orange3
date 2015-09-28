import itertools
import time
from collections import OrderedDict
from itertools import chain
from PyQt4.QtCore import Qt, QAbstractItemModel, QByteArray, QBuffer, QIODevice
from PyQt4.QtGui import QGraphicsScene, QStandardItemModel
from Orange.widgets.io import PngFormat


class Report:
    report_html = ""
    name = ""

    def show_report(self):
        from Orange.canvas.report.owreport import OWReport

        report = OWReport.get_instance()
        self.create_report_html()
        report.make_report(self)
        report.show()
        report.raise_()

    def get_widget_name_extension(self):
        return None

    def create_report_html(self):
        self.report_html = get_html_section(self.name)
        self.report_html += '<div class="content">\n'
        self.send_report()
        self.report_html += '</div>\n\n'

    @staticmethod
    def _fix_args(name, items):
        if items is None:
            return "", name
        else:
            return name, items

    def report_items(self, name, items=None):
        name, items = self._fix_args(name, items)
        self.report_name(name)
        self.report_html += render_items(items)

    def report_name(self, name):
        if name != "":
            self.report_html += get_html_subsection(name)

    def report_data(self, name, data=None):
        name, data = self._fix_args(name, data)
        self.report_items(name, describe_data(data))

    def report_domain(self, name, domain=None):
        name, domain = self._fix_args(name, domain)
        self.report_items(name, describe_domain(domain))

    def report_data_brief(self, name, data=None):
        name, data = self._fix_args(name, data)
        self.report_items(name, describe_data_brief(data))

    def report_plot(self, name, plot=None):
        name, plot = self._fix_args(name, plot)
        from pyqtgraph import PlotWidget, PlotItem, GraphicsWidget
        self.report_name(name)
        if isinstance(plot, QGraphicsScene):
            self.report_html += get_html_img(plot)
        elif isinstance(plot, PlotItem):
            self.report_html += get_html_img(plot)
        elif isinstance(plot, PlotWidget):
            self.report_html += get_html_img(plot.plotItem)
        elif isinstance(plot, GraphicsWidget):
            self.report_html += get_html_img(plot.scene())

    # noinspection PyBroadException
    def report_table(self, name, table=None, header_rows=0, header_columns=0,
                     num_format=None):
        name, table = self._fix_args(name, table)
        join = "".join

        def report_standard_model(model):
            content = ((model.item(row, col).data(Qt.DisplayRole)
                        for col in range(model.columnCount())
                        ) for row in range(model.rowCount()))
            has_header = not hasattr(table, "isHeaderHidden") or \
                not table.isHeaderHidden()
            if has_header:
                try:
                    header = (model.horizontalHeaderItem(col).data(Qt.DisplayRole)
                              for col in range(model.columnCount())),
                    content = chain(header, content)
                except:
                    has_header = False
            return report_list(content, header_rows + has_header)

        # noinspection PyBroadException
        def report_abstract_model(model):
            content = ((model.data(model.index(row, col))
                        for col in range(model.columnCount())
                        ) for row in range(model.rowCount()))
            try:
                header = [model.headerData(col, Qt.Horizontal, Qt.DisplayRole)
                          for col in range(model.columnCount())]
            except:
                header = None
            if header:
                content = chain([header], content)
            return report_list(content, header_rows + bool(header))

        if num_format:
            def fmtnum(s):
                try:
                    return num_format.format(float(s))
                except:
                    return s
        else:
            def fmtnum(s):
                return s

        def report_list(data,
                        header_rows=header_rows, header_columns=header_columns):
            cells = ["<td>{}</td>", "<th>{}</th>"]
            return join("  <tr>\n    {}</tr>\n".format(
                join(cells[rowi < header_rows or coli < header_columns]
                     .format(fmtnum(elm)) for coli, elm in enumerate(row))
            ) for rowi, row in enumerate(data))

        self.report_name(name)
        if isinstance(table, QAbstractItemModel):
            model = table
        else:
            try:
                model = table.model()
            except:
                model = None
        if isinstance(model, QStandardItemModel):
            body = report_standard_model(model)
        elif isinstance(model, QAbstractItemModel):
            body = report_abstract_model(model)
        elif isinstance(table, list):
            body = report_list(table)
        else:
            body = None
        if body:
            self.report_html += "<table>\n" + body + "</table>"

    # noinspection PyBroadException
    def report_list(self, name, data=None, limit=1000):
        name, data = self._fix_args(name, data)

        def report_abstract_model(model):
            content = (model.data(model.index(row, 0))
                       for row in range(model.rowCount()))
            return clipped_list(content, limit, less_lookups=True)

        self.report_name(name)
        try:
            model = data.model()
        except:
            model = None
        if isinstance(model, QAbstractItemModel):
            txt = report_abstract_model(model)
        else:
            txt = ""
        self.report_html += txt

    def report_paragraph(self, name, text=None):
        name, text = self._fix_args(name, text)
        self.report_name(name)
        self.report_html += "<p>{}</p>".format(text)

    def report_caption(self, text):
        self.report_html += "<p class='caption'>{}</p>".format(text)

    def report_raw(self, name, html=None):
        name, html = self._fix_args(name, html)
        self.report_name(name)
        self.report_html += html

    def combo_value(self, combo):
        text = combo.currentText()
        if text != combo.emptyString:
            return text


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


def render_items(items):
    """
    Render a sequence of pairs or an `OrderedDict` as a HTML list.

    The function skips the items whose values are `None` or `False`.

    :param items: a list or dictionary of items
    :type items: dict or list
    :return: rendered content
    :rtype: str
    """
    if isinstance(items, dict):
        items = items.items()
    return "<ul>" + "".join(
        "<b>{}:</b> {}</br>".format(key, value) for key, value in items
        if value is not None and value is not False) + "</ul>"


def get_html_img(scene):
    byte_array = QByteArray()
    filename = QBuffer(byte_array)
    filename.open(QIODevice.WriteOnly)
    writer = PngFormat()
    writer.write(filename, scene)
    img_encoded = byte_array.toBase64().data().decode("utf-8")
    return "<img src='data:image/png;base64,%s'/>" % img_encoded


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
    items = OrderedDict()
    if data is None:
        return items
    items["Data instances"] = len(data)
    items.update(describe_domain(data.domain))
    return items


def describe_domain_brief(domain):
    """
    Return an :obj:`OrderedDict` with the number of features, metas and classes

    Description contains "Features" and "Meta attributes" with the number of
    featuers, and "Targets" that contains either a name, if there is a single
    target, or the number of targets if there are multiple.

    :param domain: data
    :type domain: Orange.data.Domain
    :rtype: OrderedDict
    """
    items = OrderedDict()
    if domain is None:
        return items
    items["Features"] = len(domain.attributes) or "None"
    items["Meta attributes"] = len(domain.metas) or "None"
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


def describe_data_brief(data):
    """
    Return an :obj:`OrderedDict` with a brief description of data.

    Description contains keys "Data instances" with the number of instances,
    "Features" and "Meta attributes" with the corresponding numbers, and
    "Targets", which contains a name, if there is a single target, or the
    number of targets if there are multiple.

    :param data: data
    :type data: Orange.data.Table
    :rtype: OrderedDict
    """
    items = OrderedDict()
    if data is None:
        return items
    items["Data instances"] = len(data)
    items.update(describe_domain_brief(data.domain))
    return items
