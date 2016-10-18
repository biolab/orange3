"""
Utility classes for visualization widgets
"""

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

    Derived classes must provide methods

    - `iterate_states` for generating combinations (e.g. pairs of attritutes),
    - `compute_score(state)` for computing the score of a combination,
    - `row_for_state(state)` that returns a list of items inserted into the
       table for the given state.

    and, optionally,

    - `state_count` that returns the number of combinations (used for progress
       bar)
    - `on_selection_changed` that handles event triggered when the user selects
      a table row. The method should emit signal
      `VizRankDialog.selectionChanged(object)`.

    The class provides a table and a button. A widget constructs a single
    instance of this dialog in its `__init__`, like (in Sieve) by using a
    convenience method :obj:`add_vizrank`::

        self.vizrank, self.vizrank_button = SieveRank.add_vizrank(
            box, self, "Score Combinations", self.set_attr)

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
    selectionChanged = Signal(object)

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

    @classmethod
    def add_vizrank(cls, widget, master, button_label, set_attr_callback):
        """
        Equip the widget with VizRank button and dialog, and monkey patch the
        widget's `closeEvent` and `hideEvent` to close/hide the vizrank, too.

        Args:
            widget (QWidget): the widget into whose layout to insert the button
            master (Orange.widgets.widget.OWWidget): the master widget
            button_label: the label for the button
            set_attr_callback: the callback for setting the projection chosen
                in the vizrank

        Returns:
            tuple with Vizrank dialog instance and push button
        """
        # Monkey patching could be avoided by mixing-in the class (not
        # necessarily a good idea since we can make a mess of multiple
        # defined/derived closeEvent and hideEvent methods). Furthermore,
        # per-class patching would be better than per-instance, but we don't
        # want to mess with meta-classes either.

        vizrank = cls(master)
        button = gui.button(
            widget, master, button_label, callback=vizrank.reshow,
            enabled=False)
        vizrank.selectionChanged.connect(lambda args: set_attr_callback(*args))

        master_close_event = master.closeEvent
        master_hide_event = master.hideEvent

        def closeEvent(event):
            vizrank.close()
            master_close_event(event)

        def hideEvent(event):
            vizrank.hide()
            master_hide_event(event)

        master.closeEvent = closeEvent
        master.hideEvent = hideEvent
        return vizrank, button

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

    When the user selects a pair, the dialog emits signal `selectionChanged`
    with a tuple of variables as parameter.
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
        self.selectionChanged.emit(attrs)

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
    """QGraphicsTextItem with more convenient constructor

       Args:
           scene (QGraphicsScene): scene into which the text is placed
           text (str): text; see also argument `html_text` (default: `""`)
           x (int): x-coordinate (default: 0)
           y (int): y-coordinate (default: 0)
           alignment (Qt.Alignment): text alignment
               (default: Qt.AlignLeft | Qt.AlignTop)
           bold (bool): if `True`, font is set to bold (default: `False`)
           font (QFont): text font
           z (int): text layer
           html_text (str): text as html; if present (default is `None`),
               it overrides the `text` argument
           tooltip (str): text tooltip
           show (bool): if `False`, the text is hidden (default: `True`)
           vertical (bool): if `True`, the text is rotated by 90 degrees
               (default: `False`)
    """
    def __init__(self, scene, text="", x=0, y=0,
                 alignment=Qt.AlignLeft | Qt.AlignTop, bold=False, font=None,
                 z=0, html_text=None, tooltip=None, show=True, vertical=False):
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

        if scene is not None:
            scene.addItem(self)

    def setPos(self, x, y):
        """setPos with adjustment for alignment"""
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
    """QGraphicsRectItem with more convenient constructor

    Args:
        scene (QGraphicsScene): scene into which the rectangle is placed
        x (int): x-coordinate (default: 0)
        y (int): y-coordinate (default: 0)
        width (int): rectangle's width (default: 0)
        height (int): rectangle's height (default: 0)
        z (int): z-layer
        pen (QPen): pen for the border; if present, it overrides the separate
            arguments for color, width and style
        pen_color (QColor or QPen): the (color of) the pen
            (default: `QColor(128, 128, 128)`)
        pen_width (int): pen width
        pen_style (PenStyle): pen style (default: `Qt.SolidLine`)
        brush_color (QColor): the color for the interior (default: same as pen)
        tooltip (str): tooltip
        show (bool): if `False`, the text is hidden (default: `True`)
        onclick (callable): callback for mouse click event
    """

    def __init__(self, scene, x=0, y=0, width=0, height=0,
                 pen_color=QColor(128, 128, 128), brush_color=None, pen_width=1,
                 z=0, pen_style=Qt.SolidLine, pen=None, tooltip=None, show=True,
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

        if scene is not None:
            scene.addItem(self)

    def mousePressEvent(self, event):
        if self.onclick:
            self.onclick(self, event)


class ViewWithPress(QGraphicsView):
    """QGraphicsView with a callback for mouse press event. The callback
    is given as keyword argument `handler`.
    """
    def __init__(self, *args, **kwargs):
        self.handler = kwargs.pop("handler")
        super().__init__(*args)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if not event.isAccepted():
            self.handler()


