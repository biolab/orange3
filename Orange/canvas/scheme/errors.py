"""
Scheme Errors
=============

"""


class SchemeTopologyError(Exception):
    """
    A general scheme topology error.
    """
    pass


class SchemeCycleError(SchemeTopologyError):
    """
    A link would create a cycle in the scheme.
    """
    pass


class SinkChannelError(SchemeTopologyError):
    """
    Sink channel already connected.
    """


class DuplicatedLinkError(SchemeTopologyError):
    """
    A link duplicates another link already present in the scheme.
    """


class IncompatibleChannelTypeError(TypeError):
    """
    Source and sink channels do not have compatible types
    """
