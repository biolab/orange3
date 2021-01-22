from itertools import chain, starmap, product, groupby, islice
from functools import reduce
from operator import itemgetter
from typing import List, Tuple, Iterable, Sequence, Optional, Union

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
                qitemselection_select_range(
                    selection, model, row_range, col_range
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
                qitemselection_select_range(
                    ext_selection, model, row_range, col_range
                )
            for row_range, col_range in \
                    product(to_ranges(sel_rows), to_ranges(cols)):
                qitemselection_select_range(
                    ext_selection, model, row_range, col_range
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


def merge_ranges(
        ranges: Iterable[Tuple[int, int]]
) -> Sequence[Tuple[int, int]]:
    def merge_range_seq_accum(
            accum: List[Tuple[int, int]], r: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        last_start, last_stop = accum[-1]
        r_start, r_stop = r
        assert last_start <= r_start
        if r_start <= last_stop:
            # merge into last
            accum[-1] = last_start, max(last_stop, r_stop)
        else:
            # push a new (disconnected) range interval
            accum.append(r)
        return accum

    ranges = sorted(ranges, key=itemgetter(0))
    if ranges:
        return reduce(merge_range_seq_accum, islice(ranges, 1, None),
                      [ranges[0]])
    else:
        return []


def qitemselection_select_range(
        selection: QItemSelection,
        model: QAbstractItemModel,
        rows: range,
        columns: range
) -> None:
    assert rows.step == 1 and columns.step == 1
    selection.select(
        model.index(rows.start, columns.start),
        model.index(rows.stop - 1, columns.stop - 1)
    )


def to_ranges(spans: Iterable[Tuple[int, int]]) -> Sequence[range]:
    return list(starmap(range, spans))


class SymmetricSelectionModel(QItemSelectionModel):
    """
    Item selection model ensuring the selection is symmetric

    """
    def select(self, selection: Union[QItemSelection, QModelIndex],
               flags: QItemSelectionModel.SelectionFlags) -> None:
        if isinstance(selection, QModelIndex):
            selection = QItemSelection(selection, selection)

        if flags & QItemSelectionModel.Current:  # no current selection support
            flags &= ~QItemSelectionModel.Current
        if flags & QItemSelectionModel.Toggle:  # no toggle support either
            flags &= ~QItemSelectionModel.Toggle
            flags |= QItemSelectionModel.Select

        model = self.model()
        rows, cols = selection_blocks(selection)
        sym_ranges = to_ranges(merge_ranges(chain(rows, cols)))
        if flags == QItemSelectionModel.ClearAndSelect:
            # extend ranges in `selection` to symmetric selection
            # row/columns.
            selection = QItemSelection()
            for rows, cols in product(sym_ranges, sym_ranges):
                qitemselection_select_range(selection, model, rows, cols)
        elif flags & (QItemSelectionModel.Select |
                      QItemSelectionModel.Deselect):
            # extend ranges in sym_ranges to span all current rows/columns
            rows_current, cols_current = selection_blocks(self.selection())
            ext_selection = QItemSelection()
            for rrange, crange in product(sym_ranges, sym_ranges):
                qitemselection_select_range(selection, model, rrange, crange)
            for rrange, crange in product(sym_ranges, to_ranges(cols_current)):
                qitemselection_select_range(selection, model, rrange, crange)
            for rrange, crange in product(to_ranges(rows_current), sym_ranges):
                qitemselection_select_range(selection, model, rrange, crange)
            selection.merge(ext_selection, QItemSelectionModel.Select)
        super().select(selection, flags)

    def selectedItems(self) -> Sequence[int]:
        """Return the indices of the the symmetric selection."""
        ranges_ = starmap(range, selection_rows(self.selection()))
        return sorted(chain.from_iterable(ranges_))

    def setSelectedItems(self, inds: Iterable[int]):
        """Set and select the `inds` indices"""
        model = self.model()
        selection = QItemSelection()
        sym_ranges = to_ranges(ranges(inds))
        for rows, cols in product(sym_ranges, sym_ranges):
            qitemselection_select_range(selection, model, rows, cols)
        self.select(selection, QItemSelectionModel.ClearAndSelect)
