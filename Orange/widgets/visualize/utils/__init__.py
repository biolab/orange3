from bisect import bisect_left
from operator import attrgetter

from PyQt4.QtCore import Qt, pyqtSignal as Signal, QSize
from PyQt4.QtGui import (
    QStandardItemModel, QStandardItem, QTableView, QGraphicsTextItem,
    QGraphicsRectItem, QColor, QBrush, QPen, QGraphicsView, QDialog, QVBoxLayout
)

from Orange.data import Variable
from Orange.widgets import gui
from Orange.widgets.gui import HorizontalGridDelegate
from Orange.widgets.utils.messages import WidgetMessagesMixin
from Orange.widgets.utils.progressbar import ProgressBarMixin
from Orange.widgets.widget import Msg


class VizRankDialog(QDialog, ProgressBarMixin, WidgetMessagesMixin):
    """
    Base class for VizRank dialogs, providing a GUI with a table and a button,
    and the skeleton for managing the evaluation of visualizations.

    Derived classes need to provide generators of combinations (e.g. pairs
    of attribtutes) and the scoring function. The widget stores the current
    upon pause, and restores it upon continuation.

    The class provides a table and a button. A widget constructs a single
    instance of this dialog in its `__init__`, like (in Sieve):

        self.vizrank = SieveRank(self)
        self.vizrank_button = gui.button(
            box, self, "Score Combinations", callback=self.vizrank.reshow)

    The widget (the argument `self`) above is stored in `VizRankDialog`'s
    attribute `master` since derived classes will need to interact with is.

    When the widget receives new data, it must call the VizRankDialog's
    method :obj:`VizRankDialog.initialize()` to clear the GUI and reset the
    state.

    Clicking the Start button calls method `run` (and renames the button to
    Pause). Run sets up a progress bar by getting the number of combinations
    from :obj:`VizRankDialog.state_count()`. It restores the paused state
    (if any) and calls generator :obj:`VizRankDialog.iterate_states()`. For
    each generated state, it calls :obj:`VizRankDialog.score(state)`, which
    must return the score (lower is better) for this state. If the returned
    state is not `None`, the data returned by `row_for_state` is inserted at
    the appropriate place in the table.

    Args:
        master (Orange.widget.OWWidget): widget to which the dialog belongs

    Attributes:
        master (Orange.widget.OWWidget): widget to which the dialog belongs
        captionTitle (str): the caption for the dialog. This can be a class
          attribute. `captionTitle` is used by the `ProgressBarMixin`.
    """

    captionTitle = ""

    processingStateChanged = Signal(int)
    progressBarValueChanged = Signal(float)
    messageActivated = Signal(Msg)
    messageDeactivated = Signal(Msg)

    def __init__(self, master):
        """Initialize the attributes and set up the interface"""
        QDialog.__init__(self, windowTitle=self.captionTitle)
        WidgetMessagesMixin.__init__(self)
        self.setLayout(QVBoxLayout())

        self.insert_message_bar()
        self.layout().insertWidget(0, self.message_bar)
        self.master = master

        self.keep_running = False
        self.saved_state = None
        self.saved_progress = 0
        self.scores = []

        self.rank_model = QStandardItemModel(self)
        self.rank_table = view = QTableView(
            selectionBehavior=QTableView.SelectRows,
            selectionMode=QTableView.SingleSelection,
            showGrid=False)
        view.setItemDelegate(HorizontalGridDelegate())
        view.setModel(self.rank_model)
        view.selectionModel().selectionChanged.connect(
            self.on_selection_changed)
        view.horizontalHeader().setStretchLastSection(True)
        view.horizontalHeader().hide()
        self.layout().addWidget(view)

        self.button = gui.button(
            self, self, "Start", callback=self.toggle, default=True)

    def reshow(self):
        """Put the widget on top of all windows
        """
        self.show()
        self.raise_()
        self.activateWindow()

    def initialize(self):
        """
        Clear and initialize the dialog.

        This method must be called by the widget when the data is reset,
        e.g. from `set_data` handler.
        """
        self.keep_running = False
        self.saved_state = None
        self.saved_progress = 0
        self.scores = []
        self.rank_model.clear()
        self.button.setText("Start")
        self.button.setEnabled(self.check_preconditions())

    def check_preconditions(self):
        """Check whether there is sufficient data for ranking."""
        return True

    def on_selection_changed(self, selected, deselected):
        """
        Set the new visualization in the widget when the user select a
        row in the table.

        If derived class does not reimplement this, the table gives the
        information but the user can't click it to select the visualization.

        Args:
            selected: the index of the selected item
            deselected: the index of the previously selected item
        """
        pass

    def iterate_states(self, initial_state):
        """
        Generate all possible states (e.g. attribute combinations) for the
        given data. The content of the generated states is specific to the
        visualization.

        This method must be defined in the derived classes.

        Args:
            initial_state: initial state; None if this is the first call
        """
        raise NotImplementedError

    def state_count(self):
        """
        Return the number of states for the progress bar.

        Derived classes should implement this to ensure the proper behaviour of
        the progress bar"""
        return 0

    def compute_score(self, state):
        """
        Abstract method for computing the score for the given state. Smaller
        scores are better.

        Args:
            state: the state, e.g. the combination of attributes as generated
                by :obj:`state_count`.
        """
        raise NotImplementedError

    def row_for_state(self, state, score):
        """
        Abstract method that return the items that are inserted into the table.

        Args:
            state: the state, e.g. combination of attributes
            score: score, computed by :obj:`compute_score`
            """
        raise NotImplementedError

    def _select_first_if_none(self):
        if not self.rank_table.selectedIndexes():
            self.rank_table.selectRow(0)

    def run(self):
        """Compute and show scores"""
        with self.progressBar(self.state_count()) as progress:
            progress.advance(self.saved_progress)
            for state in self.iterate_states(self.saved_state):
                if not self.keep_running:
                    self.saved_state = state
                    self.saved_progress = progress.count
                    self._select_first_if_none()
                    return
                score = self.compute_score(state)
                if score is not None:
                    pos = bisect_left(self.scores, score)
                    self.rank_model.insertRow(
                        pos, self.row_for_state(score, state))
                    self.scores.insert(pos, score)
                progress.advance()
            self._select_first_if_none()
            self.button.setText("Finished")
            self.button.setEnabled(False)

    def toggle(self):
        """Start or pause the computation."""
        self.keep_running = not self.keep_running
        if self.keep_running:
            self.button.setText("Pause")
            self.run()
        else:
            self._select_first_if_none()
            self.button.setText("Continue")


class VizRankDialogAttrPair(VizRankDialog):
    """
    VizRank dialog for pairs of attributes. The class provides most of the
    needed methods, except for `initialize` which is expected to store a
    list of `Variable` instances to `self.attrs`, and method
    `compute_score(state)` for scoring the combinations.

    The state is a pair of indices into `self.attrs`.

    When the user selects a pair, the dialog emits signal
    `pairSelected(Variable, Variable)`.
    """

    pairSelected = Signal(Variable, Variable)

    _AttrRole = next(gui.OrangeUserRole)

    class Information(VizRankDialog.Information):
        nothing_to_rank = Msg("There is nothing to rank.")

    def __init__(self, master):
        VizRankDialog.__init__(self, master)
        self.resize(320, 512)
        self.attrs = []

    def sizeHint(self):
        """Assuming two columns in the table, return `QSize(320, 512)` as
        a reasonable default size."""
        return QSize(320, 512)

    def check_preconditions(self):
        """Refuse ranking if there are less than two feature or instances."""
        can_rank = self.master.data is not None and \
            len(self.master.data.domain.attributes) >= 2 and \
            len(self.master.data) >= 2
        self.Information.nothing_to_rank(shown=not can_rank)
        return can_rank

    def on_selection_changed(self, selected, deselected):
        attrs = [selected.indexes()[i].data(self._AttrRole) for i in (0, 1)]
        self.pairSelected.emit(*attrs)

    def state_count(self):
        n_attrs = len(self.attrs)
        return n_attrs * (n_attrs - 1) / 2

    def iterate_states(self, initial_state):
        si, sj = initial_state or (0, 0)
        for i in range(si, len(self.attrs)):
            for j in range(sj, i):
                yield i, j
            sj = 0

    def row_for_state(self, score, state):
        items = []
        attrs = sorted((self.attrs[x] for x in state), key=attrgetter("name"))
        for attr in attrs:
            item = QStandardItem(attr.name)
            item.setData(attr, self._AttrRole)
            items.append(item)
        return items


class CanvasText(QGraphicsTextItem):
    def __init__(self, canvas, text="", x=0, y=0,
                 alignment=Qt.AlignLeft | Qt.AlignTop, bold=0, font=None, z=0,
                 html_text=None, tooltip=None, show=1, vertical=False):
        QGraphicsTextItem.__init__(self, text, None)

        if font:
            self.setFont(font)
        if bold:
            font = self.font()
            font.setBold(bold)
            self.setFont(font)
        if html_text:
            self.setHtml(html_text)

        self.alignment = alignment
        self.vertical = vertical
        if vertical:
            self.setRotation(-90)

        self.setPos(x, y)
        self.x, self.y = x, y
        self.setZValue(z)
        if tooltip:
            self.setToolTip(tooltip)
        if show:
            self.show()
        else:
            self.hide()

        if canvas is not None:
            canvas.addItem(self)

    def setPos(self, x, y):
        self.x, self.y = x, y
        rect = QGraphicsTextItem.boundingRect(self)
        if self.vertical:
            h, w = rect.height(), rect.width()
            rect.setWidth(h)
            rect.setHeight(-w)
        if int(self.alignment & Qt.AlignRight):
            x -= rect.width()
        elif int(self.alignment & Qt.AlignHCenter):
            x -= rect.width() / 2.
        if int(self.alignment & Qt.AlignBottom):
            y -= rect.height()
        elif int(self.alignment & Qt.AlignVCenter):
            y -= rect.height() / 2.
        QGraphicsTextItem.setPos(self, x, y)


class CanvasRectangle(QGraphicsRectItem):
    def __init__(self, canvas, x=0, y=0, width=0, height=0,
                 pen_color=QColor(128, 128, 128), brush_color=None, pen_width=1,
                 z=0, pen_style=Qt.SolidLine, pen=None, tooltip=None, show=1,
                 onclick=None):
        super().__init__(x, y, width, height, None)
        self.onclick = onclick
        if brush_color:
            self.setBrush(QBrush(brush_color))
        if pen:
            self.setPen(pen)
        else:
            self.setPen(QPen(QBrush(pen_color), pen_width, pen_style))
        self.setZValue(z)
        if tooltip:
            self.setToolTip(tooltip)
        if show:
            self.show()
        else:
            self.hide()

        if canvas is not None:
            canvas.addItem(self)

    def mousePressEvent(self, ev):
        if self.onclick:
            self.onclick(self, ev)


class ViewWithPress(QGraphicsView):
    def __init__(self, *args, **kwargs):
        self.handler = kwargs.pop("handler")
        super().__init__(*args)

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        if not ev.isAccepted():
            self.handler()


