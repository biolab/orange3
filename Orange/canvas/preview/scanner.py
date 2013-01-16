"""
Scheme file preview parser.

"""
import sys

import logging

from xml.sax import make_parser, handler, saxutils, SAXParseException

log = logging.getLogger(__name__)


class PreviewHandler(handler.ContentHandler):
    def __init__(self):
        self._in_name = False
        self._in_description = False
        self._in_thumbnail = False
        self.name_data = []
        self.title = None
        self.description = None
        self.description_data = []
        self.thumbnail_data = []

    def startElement(self, name, attrs):
        if name == "scheme":
            if attrs.get("version", "1.0") >= "2.0":
                self.title = attrs.get("title", None)
                self.description = attrs.get("description", None)

        elif name == "thumbnail":
            self._in_thumbnail = True

    def endElement(self, name):
        if name == "name":
            self._in_name = False
        elif name == "description":
            self._in_description = False
        elif name == "thumbnail":
            self._in_thumbnail = False

    def characters(self, content):
        if self._in_name:
            self.name_data.append(content)
        elif self._in_description:
            self.description_data.append(content)
        elif self._in_thumbnail:
            self._thumbnail_data.append(content)


def preview_parse(scheme_file):
    """Return the title, description, and thumbnail svg image data from a
    `scheme_file` (can be a file path or a file-like object).

    """
    parser = make_parser()
    handler = PreviewHandler()
    parser.setContentHandler(handler)
    parser.parse(scheme_file)

    name_data = handler.title or ""
    description_data = handler.description or ""
    svg_data = "".join(handler.thumbnail_data)

    return (saxutils.unescape(name_data),
            saxutils.unescape(description_data),
            saxutils.unescape(svg_data))


def scheme_svg_thumbnail(scheme_file):
    """Load the scheme scheme from a file and return it's svg image
    representation.

    """
    from .. import scheme
    from ..canvas import scene
    from ..registry import global_registry

    scheme = scheme.Scheme()
    scheme.load_from(scheme_file)
    tmp_scene = scene.CanvasScene()
    tmp_scene.set_registry(global_registry())
    tmp_scene.set_scheme(scheme)

    # Force the anchor point layout.
    tmp_scene.anchor_layout().activate()

    svg = scene.grab_svg(tmp_scene)
    tmp_scene.clear()
    tmp_scene.deleteLater()
    return svg


def scan_update(item):
    """Given a preview item, scan the scheme file ('item.path') and update the
    items contents.

    """

    path = unicode(item.path())

    # workaround for bugs.python.org/issue11159
    path = path.encode(sys.getfilesystemencoding())

    try:
        title, desc, svg = preview_parse(path)
    except SAXParseException, ex:
        log.error("%r is malformed (%r)", path, ex)
        item.setEnabled(False)
        item.setSelectable(False)
        return

    if not svg:
        try:
            svg = scheme_svg_thumbnail(path)
        except Exception:
            log.error("Could not render scheme preview for %r", title,
                      exc_info=True)

    if item.name() != title:
        item.setName(title)

    if item.description() != desc:
        item.setDescription(desc)

    if svg:
        item.setThumbnail(svg)
