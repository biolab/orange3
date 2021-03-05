"""Common QGraphicsScene components that can be composed when needed."""

from AnyQt.QtWidgets import QGraphicsScene

class UpdateItemsOnSelectGraphicsScene(QGraphicsScene):
    """Calls the selection_changed method on items.

    Whenever the scene selection changes, this view will call the
    ˙selection_changed˙ method on any item on the scene.

    Notes
    -----
    .. note:: I suspect this is completely unnecessary, but have not been able
        to find a reasonable way to keep the selection logic inside the actual
        `QGraphicsItem` objects

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selectionChanged.connect(self.__handle_selection)

    def __handle_selection(self):
        for item in self.items():
            if hasattr(item, 'selection_changed'):
                item.selection_changed()
