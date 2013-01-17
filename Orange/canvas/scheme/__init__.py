"""
=====================
Scheme Workflow Model
=====================

"""

from .node import SchemeNode
from .link import SchemeLink, compatible_channels, can_connect, possible_links
from .scheme import Scheme

from .annotations import (
    BaseSchemeAnnotation, SchemeArrowAnnotation, SchemeTextAnnotation
)

from .errors import *

from . import utils
