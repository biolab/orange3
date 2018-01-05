"""
======
Scheme
======

The scheme package implements and defines the underlying workflow model.

The :class:`.Scheme` class represents the workflow and is composed of a set
of :class:`.SchemeNode` connected with :class:`.SchemeLink`, defining an
directed acyclic graph (DAG). Additionally instances of
:class:`.SchemeArrowAnnotation` or :class:`.SchemeTextAnnotation` can be
inserted into the scheme.

"""

from .node import SchemeNode
from .link import SchemeLink, compatible_channels, can_connect, possible_links
from .scheme import Scheme

from .annotations import (
    BaseSchemeAnnotation, SchemeArrowAnnotation, SchemeTextAnnotation
)

# the module contains only exceptions classes; all need to be exposed
# pylint: disable=wildcard-import
from .errors import *
