"""
Scheme save/load routines.

"""
import base64
import sys

from xml.etree.ElementTree import TreeBuilder, Element, ElementTree, parse

from collections import defaultdict
from itertools import chain

import pickle as pickle
import json
import pprint

import ast
from ast import literal_eval

import logging

from . import SchemeNode, SchemeLink
from .annotations import SchemeTextAnnotation, SchemeArrowAnnotation
from .errors import IncompatibleChannelTypeError

from .. import registry

log = logging.getLogger(__name__)


class UnknownWidgetDefinition(Exception):
    pass


def string_eval(source):
    """
    Evaluate a python string literal `source`. Raise ValueError if
    `source` is not a string literal.

    >>> string_eval("'a string'")
    a string

    """
    node = ast.parse(source, "<source>", mode="eval")
    if not isinstance(node.body, ast.Str):
        raise ValueError("%r is not a string literal" % source)
    return node.body.s


def tuple_eval(source):
    """
    Evaluate a python tuple literal `source` where the elements are
    constrained to be int, float or string. Raise ValueError if not
    a tuple literal.

    >>> tuple_eval("(1, 2, "3")")
    (1, 2, '3')

    """
    node = ast.parse(source, "<source>", mode="eval")

    if not isinstance(node.body, ast.Tuple):
        raise ValueError("%r is not a tuple literal" % source)

    if not all(isinstance(el, (ast.Str, ast.Num))
               for el in node.body.elts):
        raise ValueError("Can only contain numbers or strings")

    return literal_eval(source)


def terminal_eval(source):
    """
    Evaluate a python 'constant' (string, number, None, True, False)
    `source`. Raise ValueError is not a terminal literal.

    >>> terminal_eval("True")
    True

    """
    node = ast.parse(source, "<source>", mode="eval")

    try:
        return _terminal_value(node.body)
    except ValueError:
        raise
        raise ValueError("%r is not a terminal constant" % source)


def _terminal_value(node):
    if isinstance(node, ast.Str):
        return node.s
    elif isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.Name) and \
            node.id in ["True", "False", "None"]:
        return __builtins__[node.id]

    raise ValueError("Not a terminal")


def sniff_version(stream):
    """
    Parse a scheme stream and return the scheme's serialization
    version string.

    """
    doc = parse(stream)
    scheme_el = doc.getroot()
    version = scheme_el.attrib.get("version", None)
    # Fallback: check for "widgets" tag.
    if scheme_el.find("widgets") is not None:
        version = "1.0"
    else:
        version = "2.0"

    return version


def parse_scheme(scheme, stream, error_handler=None,
                 allow_pickle_data=False):
    """
    Parse a saved scheme from `stream` and populate a `scheme`
    instance (:class:`Scheme`).
    `error_handler` if given will be called with an exception when
    a 'recoverable' error occurs. By default the exception is simply
    raised.

    Parameters
    ----------
    scheme : :class:`.Scheme`
        A scheme instance to populate with the contents of `stream`.
    stream : file-like object
        A file like object opened for reading.
    error_hander : function, optional
        A function to call with an exception instance when a `recoverable`
        error occurs.
    allow_picked_data : bool, optional
        Specifically allow parsing of picked data streams.

    """
    doc = parse(stream)
    scheme_el = doc.getroot()
    version = scheme_el.attrib.get("version", None)
    if version is None:
        # Fallback: check for "widgets" tag.
        if scheme_el.find("widgets") is not None:
            version = "1.0"
        else:
            version = "2.0"

    if error_handler is None:
        def error_handler(exc):
            raise exc

    if version == "1.0":
        parse_scheme_v_1_0(doc, scheme, error_handler=error_handler,
                           allow_pickle_data=allow_pickle_data)
        return scheme
    else:
        parse_scheme_v_2_0(doc, scheme, error_handler=error_handler,
                           allow_pickle_data=allow_pickle_data)
        return scheme


def scheme_node_from_element(node_el, registry):
    """
    Create a SchemeNode from an `Element` instance.
    """
    try:
        widget_desc = registry.widget(node_el.get("qualified_name"))
    except KeyError as ex:
        raise UnknownWidgetDefinition(*ex.args)

    title = node_el.get("title")
    pos = node_el.get("position")

    if pos is not None:
        pos = tuple_eval(pos)

    return SchemeNode(widget_desc, title=title, position=pos)


def parse_scheme_v_2_0(etree, scheme, error_handler, widget_registry=None,
                       allow_pickle_data=False):
    """
    Parse an `ElementTree` instance.
    """
    if widget_registry is None:
        widget_registry = registry.global_registry()

    nodes_not_found = []

    nodes = []
    links = []

    id_to_node = {}

    scheme_node = etree.getroot()
    scheme.title = scheme_node.attrib.get("title", "")
    scheme.description = scheme_node.attrib.get("description", "")

    # Load and create scheme nodes.
    for node_el in etree.findall("nodes/node"):
        try:
            node = scheme_node_from_element(node_el, widget_registry)
        except UnknownWidgetDefinition as ex:
            # description was not found
            error_handler(ex)
            node = None
        except Exception:
            raise

        if node is not None:
            nodes.append(node)
            id_to_node[node_el.get("id")] = node
        else:
            nodes_not_found.append(node_el.get("id"))

    # Load and create scheme links.
    for link_el in etree.findall("links/link"):
        source_id = link_el.get("source_node_id")
        sink_id = link_el.get("sink_node_id")

        if source_id in nodes_not_found or sink_id in nodes_not_found:
            continue

        source = id_to_node.get(source_id)
        sink = id_to_node.get(sink_id)

        source_channel = link_el.get("source_channel")
        sink_channel = link_el.get("sink_channel")
        enabled = link_el.get("enabled") == "true"

        try:
            link = SchemeLink(source, source_channel, sink, sink_channel,
                              enabled=enabled)
        except (ValueError, IncompatibleChannelTypeError) as ex:
            error_handler(ex)
        else:
            links.append(link)

    # Load node properties
    for property_el in etree.findall("node_properties/properties"):
        node_id = property_el.attrib.get("node_id")

        if node_id in nodes_not_found:
            continue

        node = id_to_node[node_id]

        format = property_el.attrib.get("format", "pickle")

        if "data" in property_el.attrib:
            # data string is 'encoded' with 'repr' i.e. unicode and
            # nonprintable characters are \u or \x escaped.
            # Could use 'codecs' module?
            data = string_eval(property_el.attrib.get("data"))
        else:
            data = property_el.text

        properties = None
        if format != "pickle" or allow_pickle_data:
            try:
                properties = loads(data, format)
            except Exception:
                log.error("Could not load properties for %r.", node.title,
                          exc_info=True)

        if properties is not None:
            node.properties = properties

    annotations = []
    for annot_el in etree.findall("annotations/*"):
        if annot_el.tag == "text":
            rect = annot_el.attrib.get("rect", "(0, 0, 20, 20)")
            rect = tuple_eval(rect)

            font_family = annot_el.attrib.get("font-family", "").strip()
            font_size = annot_el.attrib.get("font-size", "").strip()

            font = {}
            if font_family:
                font["family"] = font_family
            if font_size:
                font["size"] = int(font_size)

            annot = SchemeTextAnnotation(rect, annot_el.text or "", font=font)
        elif annot_el.tag == "arrow":
            start = annot_el.attrib.get("start", "(0, 0)")
            end = annot_el.attrib.get("end", "(0, 0)")
            start, end = map(tuple_eval, (start, end))

            color = annot_el.attrib.get("fill", "red")
            annot = SchemeArrowAnnotation(start, end, color=color)
        annotations.append(annot)

    for node in nodes:
        scheme.add_node(node)

    for link in links:
        scheme.add_link(link)

    for annot in annotations:
        scheme.add_annotation(annot)


def parse_scheme_v_1_0(etree, scheme, error_handler, widget_registry=None,
                       allow_pickle_data=False):
    """
    ElementTree Instance of an old .ows scheme format.
    """
    if widget_registry is None:
        widget_registry = registry.global_registry()

    widgets_not_found = []

    widgets = widget_registry.widgets()
    widgets_by_name = [(d.qualified_name.rsplit(".", 1)[-1], d)
                       for d in widgets]
    widgets_by_name = dict(widgets_by_name)

    nodes_by_caption = {}
    nodes = []
    links = []
    for widget_el in etree.findall("widgets/widget"):
        caption = widget_el.get("caption")
        name = widget_el.get("widgetName")
        x_pos = widget_el.get("xPos")
        y_pos = widget_el.get("yPos")

        if name in widgets_by_name:
            desc = widgets_by_name[name]
        else:
            error_handler(UnknownWidgetDefinition(name))
            widgets_not_found.append(caption)
            continue

        node = SchemeNode(desc, title=caption,
                          position=(int(x_pos), int(y_pos)))
        nodes_by_caption[caption] = node
        nodes.append(node)

    for channel_el in etree.findall("channels/channel"):
        in_caption = channel_el.get("inWidgetCaption")
        out_caption = channel_el.get("outWidgetCaption")

        if in_caption in widgets_not_found or \
                out_caption in widgets_not_found:
            continue

        source = nodes_by_caption[out_caption]
        sink = nodes_by_caption[in_caption]
        enabled = channel_el.get("enabled") == "1"
        signals = literal_eval(channel_el.get("signals"))

        for source_channel, sink_channel in signals:
            try:
                link = SchemeLink(source, source_channel, sink, sink_channel,
                                  enabled=enabled)
            except (ValueError, IncompatibleChannelTypeError) as ex:
                error_handler(ex)
            else:
                links.append(link)

    settings = etree.find("settings")
    properties = {}
    if settings is not None:
        data = settings.attrib.get("settingsDictionary", None)
        if data and allow_pickle_data:
            try:
                properties = literal_eval(data)
            except Exception:
                log.error("Could not load properties for the scheme.",
                          exc_info=True)

    for node in nodes:
        if node.title in properties:
            try:
                node.properties = pickle.loads(properties[node.title])
            except Exception:
                log.error("Could not unpickle properties for the node %r.",
                          node.title, exc_info=True)

        scheme.add_node(node)

    for link in links:
        scheme.add_link(link)


def inf_range(start=0, step=1):
    """Return an infinite range iterator.
    """
    while True:
        yield start
        start += step


def scheme_to_etree(scheme, data_format="literal", pickle_fallback=False):
    """
    Return an `xml.etree.ElementTree` representation of the `scheme.
    """
    builder = TreeBuilder(element_factory=Element)
    builder.start("scheme", {"version": "2.0",
                             "title": scheme.title or "",
                             "description": scheme.description or ""})

    ## Nodes
    node_ids = defaultdict(inf_range().__next__)
    builder.start("nodes", {})
    for node in scheme.nodes:
        desc = node.description
        attrs = {"id": str(node_ids[node]),
                 "name": desc.name,
                 "qualified_name": desc.qualified_name,
                 "project_name": desc.project_name or "",
                 "version": desc.version or "",
                 "title": node.title,
                 }
        if node.position is not None:
            attrs["position"] = str(node.position)

        if type(node) is not SchemeNode:
            attrs["scheme_node_type"] = "%s.%s" % (type(node).__name__,
                                                   type(node).__module__)
        builder.start("node", attrs)
        builder.end("node")

    builder.end("nodes")

    ## Links
    link_ids = defaultdict(inf_range().__next__)
    builder.start("links", {})
    for link in scheme.links:
        source = link.source_node
        sink = link.sink_node
        source_id = node_ids[source]
        sink_id = node_ids[sink]
        attrs = {"id": str(link_ids[link]),
                 "source_node_id": str(source_id),
                 "sink_node_id": str(sink_id),
                 "source_channel": link.source_channel.name,
                 "sink_channel": link.sink_channel.name,
                 "enabled": "true" if link.enabled else "false",
                 }
        builder.start("link", attrs)
        builder.end("link")

    builder.end("links")

    ## Annotations
    annotation_ids = defaultdict(inf_range().__next__)
    builder.start("annotations", {})
    for annotation in scheme.annotations:
        annot_id = annotation_ids[annotation]
        attrs = {"id": str(annot_id)}
        data = None
        if isinstance(annotation, SchemeTextAnnotation):
            tag = "text"
            attrs.update({"rect": repr(annotation.rect)})

            # Save the font attributes
            font = annotation.font
            attrs.update({"font-family": font.get("family", None),
                          "font-size": font.get("size", None)})
            attrs = [(key, value) for key, value in attrs.items()
                     if value is not None]
            attrs = dict((key, str(value)) for key, value in attrs)

            data = annotation.text

        elif isinstance(annotation, SchemeArrowAnnotation):
            tag = "arrow"
            attrs.update({"start": repr(annotation.start_pos),
                          "end": repr(annotation.end_pos)})

            # Save the arrow color
            try:
                color = annotation.color
                attrs.update({"fill": color})
            except AttributeError:
                pass

            data = None
        else:
            log.warning("Can't save %r", annotation)
            continue
        builder.start(tag, attrs)
        if data is not None:
            builder.data(data)
        builder.end(tag)

    builder.end("annotations")

    builder.start("thumbnail", {})
    builder.end("thumbnail")

    # Node properties/settings
    builder.start("node_properties", {})
    for node in scheme.nodes:
        data = None
        if node.properties:
            try:
                data, format = dumps(node.properties, format=data_format,
                                     pickle_fallback=pickle_fallback)
            except Exception:
                log.error("Error serializing properties for node %r",
                          node.title, exc_info=True)
            if data is not None:
                builder.start("properties",
                              {"node_id": str(node_ids[node]),
                               "format": format})
                builder.data(data)
                builder.end("properties")

    builder.end("node_properties")
    builder.end("scheme")
    root = builder.close()
    tree = ElementTree(root)
    return tree


def scheme_to_ows_stream(scheme, stream, pretty=False, pickle_fallback=False):
    """
    Write scheme to a a stream in Orange Scheme .ows (v 2.0) format.

    Parameters
    ----------
    scheme : :class:`.Scheme`
        A :class:`.Scheme` instance to serialize.
    stream : file-like object
        A file-like object opened for writing.
    pretty : bool, optional
        If `True` the output xml will be pretty printed (indented).
    pickle_fallback : bool, optional
        If `True` allow scheme node properties to be saves using pickle
        protocol if properties cannot be saved using the default
        notation.

    """
    tree = scheme_to_etree(scheme, data_format="literal",
                           pickle_fallback=pickle_fallback)

    if pretty:
        indent(tree.getroot(), 0)

    if sys.version_info < (2, 7):
        # in Python 2.6 the write does not have xml_declaration parameter.
        tree.write(stream, encoding="utf-8")
    else:
        tree.write(stream, encoding="utf-8", xml_declaration=True)


def indent(element, level=0, indent="\t"):
    """
    Indent an instance of a :class:`Element`. Based on
    (http://effbot.org/zone/element-lib.htm#prettyprint).

    """
    def empty(text):
        return not text or not text.strip()

    def indent_(element, level, last):
        child_count = len(element)

        if child_count:
            if empty(element.text):
                element.text = "\n" + indent * (level + 1)

            if empty(element.tail):
                element.tail = "\n" + indent * (level + (-1 if last else 0))

            for i, child in enumerate(element):
                indent_(child, level + 1, i == child_count - 1)

        else:
            if empty(element.tail):
                element.tail = "\n" + indent * (level + (-1 if last else 0))

    return indent_(element, level, True)


def dumps(obj, format="literal", prettyprint=False, pickle_fallback=False):
    """
    Serialize `obj` using `format` ('json' or 'literal') and return its
    string representation and the used serialization format ('literal',
    'json' or 'pickle').

    If `pickle_fallback` is True and the serialization with `format`
    fails object's pickle representation will be returned

    """
    if format == "literal":
        try:
            return (literal_dumps(obj, prettyprint=prettyprint, indent=1),
                    "literal")
        except (ValueError, TypeError) as ex:
            if not pickle_fallback:
                raise

            log.warning("Could not serialize to a literal string",
                        exc_info=True)

    elif format == "json":
        try:
            return (json.dumps(obj, indent=1 if prettyprint else None),
                    "json")
        except (ValueError, TypeError):
            if not pickle_fallback:
                raise

            log.warning("Could not serialize to a json string",
                        exc_info=True)

    elif format == "pickle":
        return base64.encodebytes(pickle.dumps(obj)).decode('ascii'), "pickle"

    else:
        raise ValueError("Unsupported format %r" % format)

    if pickle_fallback:
        log.warning("Using pickle fallback")
        return base64.encodebytes(pickle.dumps(obj)).decode('ascii'), "pickle"
    else:
        raise Exception("Something strange happened.")


def loads(string, format):
    if format == "literal":
        return literal_eval(string)
    elif format == "json":
        return json.loads(string)
    elif format == "pickle":
        return pickle.loads(base64.decodebytes(string.encode('ascii')))
    else:
        raise ValueError("Unknown format")


# This is a subset of PyON serialization.
def literal_dumps(obj, prettyprint=False, indent=4):
    """
    Write obj into a string as a python literal.
    """
    memo = {}
    NoneType = type(None)

    def check(obj):
        if type(obj) in [int, int, float, bool, NoneType, str, str]:
            return True

        if id(obj) in memo:
            raise ValueError("{0} is a recursive structure".format(obj))

        memo[id(obj)] = obj

        if type(obj) in [list, tuple]:
            return all(map(check, obj))
        elif type(obj) is dict:
            return all(map(check, chain(iter(obj.keys()), iter(obj.values()))))
        else:
            raise TypeError("{0} can not be serialized as a python "
                             "literal".format(type(obj)))

    check(obj)

    if prettyprint:
        return pprint.pformat(obj, indent=indent)
    else:
        return repr(obj)


literal_loads = literal_eval
