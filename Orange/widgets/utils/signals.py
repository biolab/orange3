from typing import Optional

from orangewidget.utils.signals import (
    Input, Output, Single, Multiple, Default, NonDefault, Explicit, Dynamic,
    InputSignal, OutputSignal, WidgetSignalsMixin, LazyValue
)

from Orange.data import Table, Domain

__all__ = [
    "Input", "Output", "InputSignal", "OutputSignal",
    "Single", "Multiple", "Default", "NonDefault", "Explicit", "Dynamic",
    "WidgetSignalsMixin", "AttributeList",
]


class AttributeList(list):
    """Signal type for lists of attributes (variables)"""


def lazy_table_transform(domain: Domain,
                         data: Optional[Table]) -> LazyValue[Table]:
    if data is None:
        return None
    return LazyValue[Table](lambda: data.transform(domain),
                            domain=domain, length=len(data))
