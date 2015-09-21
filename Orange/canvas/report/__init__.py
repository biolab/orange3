import itertools
import time
from PyQt4.QtCore import QByteArray, QBuffer, QIODevice
from Orange.widgets.io import PngFormat


def plural(s, number):
    return s.format(number=number, s="s" if number % 100 != 1 else "")


def plural_w(s, number, capitalize=False):
    numbers = ("zero", "one", "two", "three", "four", "five", "six", "seven",
               "nine", "ten")
    number_str = numbers[number] if number < len(numbers) else str(number)
    if capitalize:
        number_str = number_str.capitalize()
    return s.format(number=number_str, s="s" if number % 100 != 1 else "")


def clip_string(s, limit=1000, sep=None):
    if len(s) < limit:
        return s
    s = s[:limit - 3]
    if sep is None:
        return s
    sep_pos = s.rfind(sep)
    if sep_pos == -1:
        return s
    return s[:sep_pos + len(sep)] + "..."


def clipped_list(s, limit=1000, less_lookups=False):
    if less_lookups:
        s = ", ".join(itertools.islice(s, (limit + 2) // 3))
    else:
        s = ", ".join(s)
    return clip_string(s, limit, ", ")


def get_html_section(name):
    datetime = time.strftime("%a %b %d %y, %H:%M:%S")
    return "<h1>%s <span class='timestamp'>%s</h1>" % (name, datetime)


def get_html_subsection(name):
    return "<h2>%s</h2>" % name


def render_items(items):
    return "<ul>" + "".join("<b>%s:</b> %s</br>" % i for i in items) + "</ul>"


def get_html_img(scene):
    byte_array = QByteArray()
    filename = QBuffer(byte_array)
    filename.open(QIODevice.WriteOnly)
    writer = PngFormat()
    writer.write(filename, scene)
    img_encoded = byte_array.toBase64().data().decode("utf-8")
    return "<ul><img src='data:image/png;base64,%s'/></ul>" % img_encoded
