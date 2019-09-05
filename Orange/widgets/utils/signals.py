from orangewidget.utils.signals import (
    Input, Output, Single, Multiple, Default, NonDefault, Explicit, Dynamic,
    InputSignal, OutputSignal, WidgetSignalsMixin
)

__all__ = [
    "Input", "Output", "InputSignal", "OutputSignal",
    "Single", "Multiple", "Default", "NonDefault", "Explicit", "Dynamic",
    "WidgetSignalsMixin", "AttributeList",
]


class AttributeList(list):
    """Signal type for lists of attributes (variables)"""
