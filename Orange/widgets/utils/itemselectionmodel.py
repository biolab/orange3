from itertools import chain, product, groupby
from typing import List, Tuple, Iterable, Optional, Union

from AnyQt.QtCore import (
    QModelIndex, QAbstractItemModel, QItemSelectionModel, QItemSelection,
    QObject
)


class BlockSelectionModel(QItemSelectionModel):
    """
    Item selection model ensuring the selection maintains a simple block
    like structure.

    e.g.

        [a b] c [d e]
        [f g] h [i j]

    is allowed but this is not

        [a] b  c  d e
        [f  g] h [i j]

    I.e. select the Cartesian product of row and column indices.

    """
    def __init__(
            self, model: QAbstractItemModel, parent: Optional[QObject] = None,
            selectBlocks=True, **kwargs
    ) -> None:
        super().__init__(model, parent, **kwargs)
        self.__selectBlocks = selectBlocks

    def select(self, selection: Union[QItemSelection, QModelIndex],
               flags: QItemSelectionModel.SelectionFlags) -> None:
        """Reimplemented."""
        if isinstance(selection, QModelIndex):
            selection = QItemSelection(selection, selection)

        if not self.__selectBlocks:
            super().select(selection, flags)
            return

        model = self.model()

        def to_ranges(spans):
            return list(range(*r) for r in spans)

        if flags & QItemSelectionModel.Current:  # no current selection support
            flags &= ~QItemSelectionModel.Current
        if flags & QItemSelectionModel.Toggle:  # no toggle support either
            flags &= ~QItemSelectionModel.Toggle
            flags |= QItemSelectionModel.Select

        if flags == QItemSelectionModel.ClearAndSelect:
            # extend selection ranges in `selection` to span all row/columns
            sel_rows = selection_rows(selection)
            sel_cols = selection_columns(selection)
            selection = QItemSelection()
            for row_range, col_range in \
                    product(to_ranges(sel_rows), to_ranges(sel_cols)):
                selection.select(
                    model.index(row_range.start, col_range.start),
                    model.index(row_range.stop - 1, col_range.stop - 1)
                )
        elif flags & (QItemSelectionModel.Select |
                      QItemSelectionModel.Deselect):
            # extend all selection ranges in `selection` with the full current
            # row/col spans
            rows, cols = selection_blocks(self.selection())
            sel_rows = selection_rows(selection)
            sel_cols = selection_columns(selection)
            ext_selection = QItemSelection()
            for row_range, col_range in \
                    product(to_ranges(rows), to_ranges(sel_cols)):
                ext_selection.select(
                    model.index(row_range.start, col_range.start),
                    model.index(row_range.stop - 1, col_range.stop - 1)
                )
            for row_range, col_range in \
                    product(to_ranges(sel_rows), to_ranges(cols)):
                ext_selection.select(
                    model.index(row_range.start, col_range.start),
                    model.index(row_range.stop - 1, col_range.stop - 1)
                )
            selection.merge(ext_selection, QItemSelectionModel.Select)
        super().select(selection, flags)

    def selectBlocks(self):
        """Is the block selection in effect."""
        return self.__selectBlocks

    def setSelectBlocks(self, state):
        """Set the block selection state.

        If set to False, the selection model behaves as the base
        QItemSelectionModel

        """
        self.__selectBlocks = state


def selection_rows(selection):
    # type: (QItemSelection) -> List[Tuple[int, int]]
    """
    Return a list of ranges for all referenced rows contained in selection

    Parameters
    ----------
    selection : QItemSelection

    Returns
    -------
    rows : List[Tuple[int, int]]
    """
    spans = set(range(s.top(), s.bottom() + 1) for s in selection)
    indices = sorted(set(chain.from_iterable(spans)))
    return list(ranges(indices))


def selection_columns(selection):
    # type: (QItemSelection) -> List[Tuple[int, int]]
    """
    Return a list of ranges for all referenced columns contained in selection

    Parameters
    ----------
    selection : QItemSelection

    Returns
    -------
    rows : List[Tuple[int, int]]
    """
    spans = {range(s.left(), s.right() + 1) for s in selection}
    indices = sorted(set(chain.from_iterable(spans)))
    return list(ranges(indices))


def selection_blocks(selection):
    # type: (QItemSelection) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]
    if selection.count() > 0:
        rowranges = {range(span.top(), span.bottom() + 1)
                     for span in selection}
        colranges = {range(span.left(), span.right() + 1)
                     for span in selection}
    else:
        return [], []

    rows = sorted(set(chain.from_iterable(rowranges)))
    cols = sorted(set(chain.from_iterable(colranges)))
    return list(ranges(rows)), list(ranges(cols))


def ranges(indices):
    # type: (Iterable[int]) -> Iterable[Tuple[int, int]]
    """
    Group consecutive indices into `(start, stop)` tuple 'ranges'.

    >>> list(ranges([1, 2, 3, 5, 3, 4]))
    >>> [(1, 4), (5, 6), (3, 5)]

    """
    g = groupby(enumerate(indices), key=lambda t: t[1] - t[0])
    for _, range_ind in g:
        range_ind = list(range_ind)
        _, start = range_ind[0]
        _, end = range_ind[-1]
        yield start, end + 1


class SymmetricSelectionModel(QItemSelectionModel):
    def select(self, selection, flags):
        if isinstance(selection, QModelIndex):
            selection = QItemSelection(selection, selection)

        model = self.model()
        indexes = selection.indexes()
        sel_inds = {ind.row() for ind in indexes} | \
                   {ind.column() for ind in indexes}
        if flags == QItemSelectionModel.ClearAndSelect:
            selected = set()
        else:
            selected = {ind.row() for ind in self.selectedIndexes()}
        if flags & QItemSelectionModel.Select:
            selected |= sel_inds
        elif flags & QItemSelectionModel.Deselect:
            selected -= sel_inds
        new_selection = QItemSelection()
        regions = list(ranges(sorted(selected)))
        for r_start, r_end in regions:
            for c_start, c_end in regions:
                top_left = model.index(r_start, c_start)
                bottom_right = model.index(r_end - 1, c_end - 1)
                new_selection.select(top_left, bottom_right)
        QItemSelectionModel.select(self, new_selection,
                                   QItemSelectionModel.ClearAndSelect)

    def selected_items(self):
        return list({ind.row() for ind in self.selectedIndexes()})

    def set_selected_items(self, inds):
        index = self.model().index
        selection = QItemSelection()
        for i in inds:
            selection.select(index(i, i), index(i, i))
        self.select(selection, QItemSelectionModel.ClearAndSelect)
