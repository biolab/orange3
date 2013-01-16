"""
Scheme save/load routines.
"""
import sys
import io
import shutil

from xml.etree.ElementTree import TreeBuilder, Element, ElementTree
from xml.dom import minidom

from collections import defaultdict

import pickle

from ast import literal_eval

import logging

from . import SchemeNode
from . import SchemeLink
from .annotations import SchemeTextAnnotation, SchemeArrowAnnotation

from .. import registry

log = logging.getLogger(__name__)


def parse_scheme(scheme, stream):
    """Parse a saved scheme from `stream` and populate a `scheme`
    instance (:class:`Scheme`).

    """
    from xml.etree.ElementTree import parse
    doc = parse(stream)
    scheme_el = doc.getroot()
    version = scheme_el.attrib.get("version", None)
    if version is None:
        # Fallback: check for "widgets" tag.
        if scheme_el.find("widgets") is not None:
            version = "1.0"
        else:
            version = "2.0"

    if version == "1.0":
        parse_scheme_v_1_0(doc, scheme)
        return scheme
    else:
        parse_scheme_v_2_0(doc, scheme)
        return scheme


def scheme_node_from_element(node_el, registry):
    """Create a SchemeNode from an Element instance.
    """
    widget_desc = registry.widget(node_el.get("qualified_name"))
    title = node_el.get("title")
    pos = node_el.get("position")

    if pos is not None:
        pos = literal_eval(pos)

    return SchemeNode(widget_desc, title=title, position=pos)


def parse_scheme_v_2_0(etree, scheme, widget_registry=None):
    """Parse an ElementTree Instance.
    """
    if widget_registry is None:
        widget_registry = registry.global_registry()

    nodes = []
    links = []

    id_to_node = {}

    scheme_node = etree.getroot()
    scheme.title = scheme_node.attrib.get("title", "")
    scheme.description = scheme_node.attrib.get("description", "")

    # Load and create scheme nodes.
    for node_el in etree.findall("nodes/node"):
        # TODO: handle errors.
        try:
            node = scheme_node_from_element(node_el, widget_registry)
        except Exception:
            raise

        nodes.append(node)
        id_to_node[node_el.get("id")] = node

    # Load and create scheme links.
    for link_el in etree.findall("links/link"):
        source = id_to_node[link_el.get("source_node_id")]
        sink = id_to_node[link_el.get("sink_node_id")]
        source_channel = link_el.get("source_channel")
        sink_channel = link_el.get("sink_channel")
        enabled = link_el.get("enabled") == "true"
        link = SchemeLink(source, source_channel, sink, sink_channel,
                          enabled=enabled)
        links.append(link)

    # Load node properties
    for property_el in etree.findall("node_properties/properties"):
        node_id = property_el.attrib.get("node_id")
        node = id_to_node[node_id]

        format = property_el.attrib.get("format", "pickle")

        if "data" in property_el.attrib:
            data = literal_eval(property_el.attrib.get("data"))
        else:
            data = property_el.text

        properties = None
        try:
            if format != "pickle":
                raise ValueError("Cannot handle %r format" % format)

            properties = pickle.loads(data)
        except Exception:
            log.error("Could not load properties for %r.", node.title,
                      exc_info=True)

        if properties is not None:
            node.properties = properties

    annotations = []
    for annot_el in etree.findall("annotations/*"):
        if annot_el.tag == "text":
            rect = annot_el.attrib.get("rect", "(0, 0, 20, 20)")
            rect = literal_eval(rect)

            font_family = annot_el.attrib.get("font-family", "").strip()
            font_size = annot_el.attrib.get("font-size", "").strip()

            font = {}
            if font_family:
                font["family"] = font_family
            if font_size:
                font["size"] = literal_eval(font_size)

            annot = SchemeTextAnnotation(rect, annot_el.text or "", font=font)
        elif annot_el.tag == "arrow":
            start = annot_el.attrib.get("start", "(0, 0)")
            end = annot_el.attrib.get("end", "(0, 0)")
            start, end = list(map(literal_eval, (start, end)))

            color = annot_el.attrib.get("fill", "red")
            annot = SchemeArrowAnnotation(start, end, color=color)
        annotations.append(annot)

    for node in nodes:
        scheme.add_node(node)

    for link in links:
        scheme.add_link(link)

    for annot in annotations:
        scheme.add_annotation(annot)


def parse_scheme_v_1_0(etree, scheme, widget_registry=None):
    """ElementTree Instance of an old .ows scheme format.
    """
    if widget_registry is None:
        widget_registry = registry.global_registry()

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
        desc = widgets_by_name[name]
        node = SchemeNode(desc, title=caption,
                          position=(int(x_pos), int(y_pos)))
        nodes_by_caption[caption] = node
        nodes.append(node)
#        scheme.add_node(node)

    for channel_el in etree.findall("channels/channel"):
        in_caption = channel_el.get("inWidgetCaption")
        out_caption = channel_el.get("outWidgetCaption")
        source = nodes_by_caption[out_caption]
        sink = nodes_by_caption[in_caption]
        enabled = channel_el.get("enabled") == "1"
        signals = eval(channel_el.get("signals"))
        for source_channel, sink_channel in signals:
            link = SchemeLink(source, source_channel, sink, sink_channel,
                              enabled=enabled)
            links.append(link)
#            scheme.add_link(link)

    settings = etree.find("settings")
    properties = {}
    if settings is not None:
        data = settings.attrib.get("settingsDictionary", None)
        if data:
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


def scheme_to_etree(scheme):
    """Return an `xml.etree.ElementTree` representation of the `scheme.
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
            attrs = [(key, value) for key, value in list(attrs.items()) \
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
                data = pickle.dumps(node.properties)
            except Exception:
                log.error("Error serializing properties for node %r",
                          node.title, exc_info=True)
            if data is not None:
                builder.start("properties",
                              {"node_id": str(node_ids[node]),
                               "format": "pickle",
#                               "data": repr(data),
                               })
                builder.data(data)
                builder.end("properties")

    builder.end("node_properties")
    builder.end("scheme")
    root = builder.close()
    tree = ElementTree(root)
    return tree


def scheme_to_ows_stream(scheme, stream, pretty=False):
    """Write scheme to a a stream in Orange Scheme .ows (v 2.0) format.
    """
    tree = scheme_to_etree(scheme)
    buffer = io.StringIO()

    if sys.version_info < (2, 7):
        # in Python 2.6 the write does not have xml_declaration parameter.
        tree.write(buffer, encoding="utf-8")
    else:
        tree.write(buffer, encoding="utf-8", xml_declaration=True)

    if pretty:
        dom = minidom.parse(io.StringIO(buffer.getvalue()))
        pretty_xml = dom.toprettyxml(encoding="utf-8")
        buffer = io.StringIO(pretty_xml)
    else:
        buffer.seek(0)

    shutil.copyfileobj(buffer, stream)
