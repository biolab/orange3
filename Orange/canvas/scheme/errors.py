"""
Scheme Errors
"""


class SchemeTopologyError(Exception):
    pass


class SchemeCycleError(SchemeTopologyError):
    pass


class IncompatibleChannelTypeError(TypeError):
    pass
