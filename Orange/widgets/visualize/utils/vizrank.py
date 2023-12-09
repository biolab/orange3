from bisect import bisect_left
from queue import Queue, Empty
from types import SimpleNamespace as namespace
from typing import Optional, Iterable, List, Callable, Iterator, Any, Type
from threading import Timer

from AnyQt.QtCore import (
    Qt, QSize, QSortFilterProxyModel, QObject, pyqtSignal as Signal)
from AnyQt.QtGui import (
    QStandardItemModel, QStandardItem, QShowEvent, QCloseEvent, QHideEvent)
from AnyQt.QtWidgets import (
    QTableView, QDialog, QVBoxLayout, QLineEdit, QPushButton)

from orangewidget.widget import OWBaseWidget
from Orange.widgets import gui
from Orange.widgets.gui import HorizontalGridDelegate, TableBarItem
from Orange.widgets.utils.concurrent import ConcurrentMixin, TaskState
from Orange.widgets.utils.progressbar import ProgressBarMixin


class Result(namespace):
    queue = None  # type: Queue[QueuedScore, ...]
    scores = None  # type: Optional[List[float, ...]]


class QueuedScore(namespace):
    position = None  # type: int
    score = None  # type: float
    state = None  # type: Iterable


class RunState(namespace):
    Invalid = 0  # Used only as default, changed to Initialized when instantiated
    Initialized = 1  # Has data; prepare_run must be called before starting computation
    Ready = 2  # Has data, iterator is initialized, but never used
    Running = 3  # Scoring thread is running
    Paused = 4  # Scoring thread is inactive, but can continue (without prepare_run)
    Hidden = 5  # Dialog is hidden and ranking paused; will resume on reopen
    Done = 6  # Scoring is done

    # The difference between Paused and Hidden is that for the former the
    # ranking is not automatically restarted when the dialog is reopened.

    state: int = Invalid
    iterator: Iterable = None
    completed: int = 0

    def can_run(self):
        return self.state in (self.Ready, self.Paused, self.Hidden)


class VizRankDialog(QDialog, ProgressBarMixin, ConcurrentMixin):
    """
    Base class for VizRank dialogs, providing a GUI with a table and a button,
    and the skeleton for managing the evaluation of visualizations.

    A new instance of this class is used for new data or any change made in
    the widget (e.g. color attribute in the scatter plot).

    The attribute run_state (and the related signal runStateChanged) can be
    used for tracking the work flow of the widget. run_state.state can be

    - RunState.Initialized (after __init__): the widget has the data, but
       the state generator is not constructed, total statecount is not known
       yet. The dialog may reenter this state after, for instance, changing
       the number of attributes per combination in radviz.
    - RunState.Ready (after prepare_run): the iterator is ready, state count
       is known: the ranking can commence.
    - RunState.Running (after start_computation): ranking is in progress.
       This state is entered by start_computation. If start_computation is
       called when the state is Initialized, start computation will call
       prepare_run.
    - RunState.Paused (after pause_computation): ranking is paused. This may
       continue (with start_computation) or be reset when parameters of VizRank
       are changed (by calling `set_run_state(RunState.Initialized)` and then
       start_computation, which will first call prepare_run to go to Ready).
    - RunState.Hidden (after closing the dialog): ranking is paused. This is
       similar to `RunState.Paused`, except that reopening dialog resumes it.
    - RunState.Done: ranking is done. The only allowed next state is
       Initialized (with the procedure described in the previous point).

    State should be changed through set_run_state, which also changes the button
    label and disables the button when Done.

    Derived classes must provide methods

    - `__init__(parent, *args, **kwargs)`: stores the data to run vizrank,
    - `state_generator()`: generates combinations (e.g. of attributes),
    - `compute_score(state)`: computes the score for the combination,
    - `row_for_state(state)`: returns a list of items inserted into the
       table for the given state.

    and, probably,

    - emit `selectionChanged` when the user changes the selection in the table,

    and, optionally,

    - `state_count`: return the number of combinations (used for progress bar)
    - `bar_length`: return the length of the bar corresponding to the score,
    - `auto_select`: selects the row corresponding to the given data.

    Derived classes are also responsible for connecting to
    rank_table.selectionModel().selectionChanged and emitting something useful
    via selectionChanged

    The constructor shouldn't do anything but store the necessary data (as is)
    because instances of this object are created on any new data. The actual
    work can only start in `prepare_run`. `prepare_run` usually won't be
    overriden, so the first computation in derived classes will usually happen
    in `compute_score` or, in classes derived from`VizRankAttributes`, in
    `score_attributes`.

    Args:
        parent (Orange.widget.OWWidget): widget to which the dialog belongs

    Attributes:
        captionTitle (str): the caption for the dialog. This can be a class
          attribute. `captionTitle` is used by the `ProgressBarMixin`.
        show_bars (True): if True (default), table with scores contains bars
          of length -score. For a different length, override `bar_lenght`.
          To hide bars (e.g. because scores can't be normalized) set this to
          `False`.

    Signal:
        selectionChanged(object): emitted when selection in the table is
            changed. The data type depends on the derived class (e.g. a list
            of attributes)
        runStateChanged(int, dict): emitted when the run state changes
            (e.g. start, pause...). Derived classes can fill the dictionary
            with additional data, e.g. the state of the user interface
    """
    captionTitle = "Score Plots"
    show_bars = True

    selectionChanged = Signal(object)
    runStateChanged = Signal(int, dict)

    button_labels = {RunState.Initialized: "Start",
                     RunState.Ready: "Start",
                     RunState.Running: "Pause",
                     RunState.Paused: "Continue",
                     RunState.Hidden: "Continue",
                     RunState.Done: "Finished",
                     RunState.Invalid: "Start"}

    def __init__(self, parent):
        QDialog.__init__(self, parent, windowTitle=self.captionTitle)
        ConcurrentMixin.__init__(self)
        ProgressBarMixin.__init__(self)
        self.setLayout(QVBoxLayout())

        self.scores = []
        self.add_to_model = Queue()
        self.run_state = RunState()
        self.total_states = 1

        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter ...")
        self.layout().addWidget(self.filter)

        self.rank_model = QStandardItemModel(self)
        self.model_proxy = QSortFilterProxyModel(
            self, filterCaseSensitivity=Qt.CaseInsensitive)
        self.model_proxy.setSourceModel(self.rank_model)
        self.filter.textChanged.connect(self.model_proxy.setFilterFixedString)

        self.rank_table = view = QTableView(
            selectionBehavior=QTableView.SelectRows,
            selectionMode=QTableView.SingleSelection,
            showGrid=False,
            editTriggers=gui.TableView.NoEditTriggers)
        view.setItemDelegate(TableBarItem() if self.show_bars
                             else HorizontalGridDelegate())
        view.setModel(self.model_proxy)
        view.horizontalHeader().setStretchLastSection(True)
        view.horizontalHeader().hide()
        self.layout().addWidget(view)

        self.button = gui.button(self, self, "Start", default=True)

        @self.button.pressed.connect
        def on_button_pressed():
            if self.run_state.state == RunState.Running:
                self.pause_computation()
            else:
                self.start_computation()

        self.set_run_state(RunState.Initialized)

    def prepare_run(self) -> None:
        """
        Called by start_computation before running for the first time or with
        new parameters within the vizrank gui, e.g. a different number of
        attributes in combinations.

        Derived classes may override this method to add other preparation steps,
        but shouldn't need to call it.
        """
        self.progressBarInit()
        self.scores = []
        self._update_model()  # empty queue
        self.rank_model.clear()
        self.run_state.iterator = self.state_generator()
        self.total_states = self.state_count() or 1
        self.set_run_state(RunState.Ready)

    def start_computation(self) -> None:
        if self.run_state.state == RunState.Initialized:
            self.prepare_run()
        if not self.run_state.can_run():
            return
        self.set_run_state(RunState.Running)
        self.start(
            self.run_vizrank,
            self.compute_score, self.scores,
            self.run_state.iterator, self.run_state.completed)

    def pause_computation(self, new_state=RunState.Paused) -> None:
        if not self.run_state.state == RunState.Running:
            return
        self.set_run_state(new_state)
        self.cancel()
        self._update_model()

    def dialog_reopened(self):
        if self.run_state.state != RunState.Paused:
            self.start_computation()

    @staticmethod
    def run_vizrank(compute_score: Callable, scores: List,
                    state_iterator: Iterator, completed: int, task: TaskState):
        res = Result(queue=Queue(), scores=None, completed_states=completed)
        scores = scores.copy()
        can_set_partial_result = True

        def do_work(st: Any):
            score = compute_score(st)
            if score is not None:
                pos = bisect_left(scores, score)
                res.queue.put_nowait(QueuedScore(position=pos, score=score,
                                                 state=st))
                scores.insert(pos, score)
            res.scores = scores.copy()
            res.completed_states += 1

        def reset_flag():
            nonlocal can_set_partial_result
            can_set_partial_result = True

        for state in state_iterator:
            do_work(state)
            # Prevent simple scores from invoking 'task.set_partial_result')
            # too frequently and making the widget unresponsive
            if can_set_partial_result:
                task.set_partial_result(res)
                can_set_partial_result = False
                Timer(0.01, reset_flag).start()
            if task.is_interruption_requested():
                return res
        task.set_partial_result(res)
        return res

    def on_partial_result(self, result: Result) -> None:
        try:
            while True:
                queued = result.queue.get_nowait()
                self.add_to_model.put_nowait(queued)
        except Empty:
            pass
        self.scores = result.scores
        self._update_model()
        self.run_state.completed = result.completed_states
        self.progressBarSet(self._progress)

    @property
    def _progress(self) -> int:
        return int(round(self.run_state.completed * 100 / self.total_states))

    def on_done(self, result: Result) -> None:
        self.progressBarFinished()
        self.set_run_state(RunState.Done)
        self._update_model()
        if not self.rank_table.selectedIndexes() \
                and self.rank_table.model().rowCount():
            self.rank_table.selectRow(0)

    def set_run_state(self, state: Any) -> None:
        if state != self.run_state.state:
            self.run_state.state = state
            if state <= RunState.Ready:
                self.run_state.completed = 0
            self.emit_run_state_changed()
        if state == RunState.Paused:
            self.setWindowTitle(
                f"{self.captionTitle} (paused at {self._progress}%)")
        self.set_button_state()

    def emit_run_state_changed(self):
        self.runStateChanged.emit(self.run_state.state, {})

    def set_button_state(self,
                         label: Optional[str] = None,
                         enabled: Optional[bool]=None) -> None:
        state = self.run_state.state
        self.button.setText(
            label if label is not None
            else self.button_labels[state])
        self.button.setEnabled(
            enabled if enabled is not None
            else state not in [RunState.Done, RunState.Invalid])

    def _update_model(self) -> None:
        try:
            while True:
                queued = self.add_to_model.get_nowait()
                row_items = self.row_for_state(queued.score, queued.state)
                bar_length = self.bar_length(queued.score)
                if bar_length is not None:
                    row_items[0].setData(bar_length,
                                         gui.TableBarItem.BarRole)
                self.rank_model.insertRow(queued.position, row_items)
        except Empty:
            pass

    def showEvent(self, event: QShowEvent) -> None:
        # pylint: disable=protected-access
        self.parent()._restore_vizrank_geometry()
        super().showEvent(event)

    def hideEvent(self, event: QHideEvent) -> None:
        # pylint: disable=protected-access
        self.pause_computation(RunState.Hidden)
        self.parent()._save_vizrank_geometry()
        super().hideEvent(event)

    def state_generator(self) -> Iterable:
        """
        Generate all possible states (e.g. attribute combinations) for the
        given data. The content of the generated states is specific to the
        visualization.
        """
        raise NotImplementedError

    def compute_score(self, state: Any) -> Optional[float]:
        """
        Abstract method for computing the score for the given state. Smaller
        scores are better.

        Args:
            state: the state, e.g. the combination of attributes as generated
                by :obj:`state_count`.
        """
        raise NotImplementedError

    def row_for_state(self, score: float, state: Any) -> List[QStandardItem]:
        """
        Return a list of items that are inserted into the table.

        Args:
            score: score, computed by :obj:`compute_score`
            state: the state, e.g. combination of attributes
            """
        raise NotImplementedError

    def auto_select(self, arg: Any) -> None:
        """
        Select the row corresponding to the give data.
        """

    def state_count(self) -> int:
        """
        Return the total number of states, needed for the progress bar.
        """
        return 1

    def bar_length(self, score: float) -> float:
        """
        Return the bar length (between 0 and 1) corresponding to the score.
        Return `None` if the score cannot be normalized.
        """
        return max(0., -score)


# According to PEP-8, names should follow usage, not implementation
# pylint: disable=invalid-name
def VizRankMixin(vizrank_class: Type[VizRankDialog]) -> Type[type]:
    """
    A mixin that serves as an interface between the vizrank dialog and the widget.

    Widget should avoid directly access the vizrank dialog.

    This mixin takes care of constructing the vizrank dialog, raising it,
    and for closing and shutting the vizrank down when necessary. Data for
    vizrank is passed to the vizrank through the mixin, and the mixin forwards
    the signals from vizrank dialog (e.g. when the user selects rows in vizrank)
    to the widget.

    The mixin is parametrized: it must be given the VizRank class to open,
    as in

    ```
    class OWMosaicDisplay(OWWidget, VizRankMixin(MosaicVizRank)):
    ```

    There should therefore be no need to subclass this class.

    Method `vizrank_button` returns a button to be placed into the widget.

    Signals:
     - `vizrankSelectionChanged` is connected to VizRank's selectionChanged.
       E.g. MosaicVizRank.selectionChanged is forwarded to selectionChanged,
       and contains the data sent by the former.
     - `virankRunStateChanged` is connected to VizRank's runStateChanged.
       This can be used to retrieve the settings from the dialog at appropriate
       times.
    - If the widget emits a signal `vizrankAutoSelect` when
      the user manually changes the variables shown in the plot (e.g. x or y
      attribute in the scatter plot), the vizrank will also select this
      combination in the list, if found.
"""

    # __VizRankMixin must be derived from OWBaseWidget so that it is
    # properly included in MRO and calls to super(),
    # e.g. super().onDeleteWidget() are properly called.
    #
    # Yet PyQt cannot handle diamond-shaped inheritance: signals exist,
    # but can't be connected. While PyQt states that "It is not possible to
    # define a new Python class that sub-classes from more than one Qt class
    # (https://www.riverbankcomputing.com/static/Docs/PyQt5/gotchas.html#multiple-inheritance),
    # this is not the case here, because we subclass from QDialog in the
    # tip of the diamond, hence it is a single inheritance with multiple
    # paths. Hence we define signals in a separate object, instantiate it,
    # and forward the signals through it.
    class VizrankSignalDelegator(QObject):
        vizrankSelectionChanged = Signal(object)
        vizrankRunStateChanged = Signal(int, dict)
        vizrankAutoSelect = Signal(object)

    class __VizRankMixin(OWBaseWidget, openclass=True):  # pylint: disable=invalid_name
        __button: Optional[QPushButton] = None
        __vizrank: Optional[VizRankDialog] = None
        __vizrank_geometry: Optional[bytes] = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__signal_delegator = VizrankSignalDelegator()

        @property
        def vizrankSelectionChanged(self):
            return self.__signal_delegator.vizrankSelectionChanged

        @property
        def vizrankAutoSelect(self):
            return self.__signal_delegator.vizrankAutoSelect

        @property
        def vizrankRunStateChanged(self):
            return self.__signal_delegator.vizrankRunStateChanged

        @property
        def vizrank_dialog(self) -> Optional[VizRankDialog]:
            """
            The vizrank dialog. This should be used only in tests.
            """
            return self.__vizrank

        def vizrank_button(self, button_label: Optional[str] = None) -> QPushButton:
            """
            A button that opens/starts the vizrank.

            The label is optional because this function is used for
            constructing the button as well as for retrieving it later.
            """
            if self.__button is None:
                self.__button = QPushButton()
                self.__button.pressed.connect(self.start_vizrank)
                self.__button.pressed.connect(self.raise_vizrank)
                # It's implausible that vizrank_button will be called after
                # init_vizrank, we could just disable the button. But let's
                # play it safe.
                self.__button.setDisabled(self.__vizrank is None)
            if button_label is not None:
                self.__button.setText(button_label)
            return self.__button

        def init_vizrank(self, *args, **kwargs) -> None:
            """
            Construct the vizrank dialog

            Any data is given to the constructor. This also enables the button
            if it exists.
            """
            self.shutdown_vizrank()
            self.__vizrank = vizrank_class(self, *args, **kwargs)
            self.__vizrank.selectionChanged.connect(self.vizrankSelectionChanged)
            self.__vizrank.runStateChanged.connect(self.vizrankRunStateChanged)
            self.vizrankAutoSelect.connect(self.__vizrank.auto_select)
            # There may be Vizrank without a button ... perhaps.
            if self.__button is not None:
                self.__button.setEnabled(True)
                self.__button.setToolTip("")

        def disable_vizrank(self, reason: str = "") -> None:
            """
            Shut down the vizrank thread, closes it, disables the button.

            The method should be called when the widget has no data or cannot
            run vizrank on it. The optional `reason` is set as tool tip.
            """
            self.shutdown_vizrank()
            if self.__button is not None:
                self.__button.setEnabled(False)
                self.__button.setToolTip(reason)

        def start_vizrank(self) -> None:
            """
            Start the ranking.

            There should be no reason to call this directly,
            unless the widget has no vizrank button.
            """
            self.__vizrank.dialog_reopened()

        def raise_vizrank(self) -> None:
            """
            Start the ranking.

            There should be no reason to call this directly,
            unless the widget has no vizrank button.
            """
            if self.__vizrank is None:
                return
            self.__vizrank.show()
            self.__vizrank.activateWindow()
            self.__vizrank.raise_()

        def shutdown_vizrank(self) -> None:
            """
            Start the ranking.

            There should be no reason to call this directly:
            the method is called
            - from init_vizrank (the widget received new data),
            - from disable_vizrank (the widget lost data, or data is unsuitable),
            - when the widget is deleted.
            """
            if self.__vizrank is None:
                return
            self.__vizrank.shutdown()
            self.__vizrank.close()
            self.__vizrank.deleteLater()
            self.__vizrank = None

        # The following methods ensure that the vizrank dialog is
        # closed/hidden/destroyed together with its parent widget.
        def closeEvent(self, event: QCloseEvent) -> None:
            if self.__vizrank:
                self.__vizrank.close()
            super().closeEvent(event)

        def hideEvent(self, event: QHideEvent) -> None:
            if self.__vizrank:
                self.__vizrank.hide()
            super().hideEvent(event)

        def onDeleteWidget(self) -> None:
            self.shutdown_vizrank()
            super().onDeleteWidget()

        def _save_vizrank_geometry(self) -> None:
            assert self.__vizrank
            self.__vizrank_geometry = self.__vizrank.saveGeometry()

        def _restore_vizrank_geometry(self) -> None:
            assert self.__vizrank
            if self.__vizrank_geometry is not None:
                self.__vizrank.restoreGeometry(self.__vizrank_geometry)

    # Give the returned class a proper name, like MosaicVizRankMixin
    return type(f"{vizrank_class.__name__}Mixin", (__VizRankMixin, ), {})


# This is an abstract class, pylint: disable=abstract-method
class VizRankDialogAttrs(VizRankDialog):
    """
    Base class for VizRank classes that work over combinations of attributes.

    Constructor accepts
     - data (Table),
     - attributes (list[Variable]; if omitted, data.domain.variables is used),
     - attr_color (Optional[Variable]): the "color" attribute, if applicable.

    The class assumes that `state` is a sequence of indices into a list of
    attributes. On this basis, it provides

    - `row_for_state`, that constructs a list containing a single QStandardItem
      with names of attributes and with `_AttrRole` data set to a list of
      attributes for the given state.
    - `on_selection_changed` that emits a `selectionChanged` signal with the
      above list.

    Derived classes must still provide

    - `state_generator()`: generates combinations of attribute indices
    - `compute_score(state)`: computes the score for the combination

    Derived classes will usually provide
    - `score_attribute` that will returned a list of attributes, such as found
      in self.attrs, but sorted by importance according to some heuristic.

    Attributes:
        - data (Table): data used in ranking
        - attrs (list[Variable]): applicable variables
        - attr_color (Variable or None): the target attribute

    Class attributes:
        - sort_names_in_row (bool): if set to True (default is False),
          variables in the view will be sorted alphabetically.
    """
    _AttrRole = next(gui.OrangeUserRole)
    sort_names_in_row = False

    # Ideally, the only argument would be `data`, with attributes used for
    # ranking and class_var would be the "color". This would however require
    # that widgets prepare such data when initializing vizrank even though
    # vizrank wouldn't necessarily be called at all. We will be able to afford
    # that after we migrate Table to pandas.
    def __init__(self, parent,
                 data: "Orange.data.Table",
                 attributes: Optional[List["Orange.data.Variable"]] = None,
                 attr_color: Optional["Orange.data.Variable"] = None):
        super().__init__(parent)
        self.data = data or None
        self.attrs = attributes or (
            self.data and self.data.domain.variables)
        self.attr_color = attr_color
        self._attr_order = None

        self.rank_table.selectionModel().selectionChanged.connect(
            self.on_selection_changed)

    @property
    def attr_order(self) -> List["Orange.data.Variable"]:
        """
        Attributes, sorted according to some heuristic.

        The property is computed by score_attributes when neceessary, and
        cached.
        """
        if self._attr_order is None:
            self._attr_order = self.score_attributes()
        return self._attr_order

    def score_attributes(self) -> None:
        """
        Return a list of attributes ordered according by some heuristic.

        Default implementation returns the original list `self.attrs`.
        """
        return self.attrs

    def on_selection_changed(self, selected, deselected) -> None:
        """
        Emit the currently selected combination of variables.
        """
        selection = selected.indexes()
        if not selection:
            return
        attrs = selected.indexes()[0].data(self._AttrRole)
        self.selectionChanged.emit(attrs)

    def row_for_state(self, score: float, state: List[int]
                      ) -> List[QStandardItem]:
        """
        Return the QStandardItem for the given combination of attributes.
        """
        attrs = [self.attr_order[s] for s in state]
        if self.sort_names_in_row:
            attrs.sort(key=lambda attr: attr.name.lower())
        attr_names = (a.name for a in attrs)
        item = QStandardItem(', '.join(attr_names))
        item.setData(attrs, self._AttrRole)
        return [item]

    # Renamed from the more general 'arg', pylint: disable=arguments-renamed
    def auto_select(self, attrs: List["Orange.data.Variable"]) -> None:
        """
        Find the given combination of variables and select it (if it exists)
        """
        model = self.rank_model
        self.rank_table.selectionModel().clear()
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            row_attrs = model.data(index, self._AttrRole)
            if all(x is y for x, y in zip(row_attrs, attrs)):
                self.rank_table.selectRow(row)
                self.rank_table.scrollTo(index)
                return


# This is an abstract class, pylint: disable=abstract-method
class VizRankDialogAttrPair(VizRankDialogAttrs):
    """
    Base class for VizRanks with combinations of two variables.

    Provides state_generator and state_count; derived classes must provide
    compute_score and, possibly, score_attributes.
    """
    def __init__(self, parent, data, attributes=None, attr_color=None):
        super().__init__(parent, data, attributes, attr_color)
        self.resize(320, 512)

    def sizeHint(self) -> QSize:
        return QSize(320, 512)

    def state_count(self) -> int:
        n_attrs = len(self.attrs)
        return n_attrs * (n_attrs - 1) // 2

    def state_generator(self) -> Iterable:
        return ((j, i) for i in range(len(self.attr_order)) for j in range(i))


# This is an abstract class, pylint: disable=abstract-method
class VizRankDialogNAttrs(VizRankDialogAttrs):
    """
    Base class for VizRanks with a spin for selecting the number of attributes.

    Constructor requires data, attributes, attr_color and, also, the initial
    number of attributes in the spin box.

    - Ranking is stopped if the user interacts with the spin.
    - The button label is changed to "Restart with {...} variables" if the
      number selected in the spin doesn't match the number of varialbes in the
      paused ranking, and reset back to "Continue" when it matches.
    - start_computation is overriden to lower the state to Initialized before
      calling super, if the number of selected in the spin doesn't match the
      previous run.
    - The dictionary passed by the signal runStateChanged contains n_attrs
      with the number of attributes used in the current/last ranking.
    - When closing the dialog, the spin is reset to the number of attributes
      used in the last run.
    """
    attrsSelected = Signal([])

    def __init__(self, parent,
                 data: "Orange.data.Table",
                 attributes: List["Orange.data.Variable"],
                 color: "Orange.data.Variable",
                 n_attrs: int,
                 *, spin_label: str = "Number of variables: "):
        # Add the spin box for a number of attributes to take into account.
        self.n_attrs = n_attrs
        super().__init__(parent, data, attributes, color)
        self._attr_order = None

        box = gui.hBox(self)
        self.n_attrs_spin = gui.spin(
            box, self, None, 3, 8, label=spin_label,
            controlWidth=50, alignment=Qt.AlignRight,
            callback=self.on_n_attrs_changed)

        n_cont = self.max_attrs()
        self.n_attrs_spin.setValue(min(self.n_attrs, n_cont))
        self.n_attrs_spin.setMaximum(n_cont)
        self.n_attrs_spin.parent().setDisabled(not n_cont)

    def max_attrs(self) -> int:
        return sum(v is not self.attr_color for v in self.attrs)

    def start_computation(self) -> None:
        if self.n_attrs != self.n_attrs_spin.value():
            self.n_attrs = self.n_attrs_spin.value()
            self.set_run_state(RunState.Initialized)
        self.n_attrs_spin.lineEdit().deselect()
        self.rank_table.setFocus(Qt.FocusReason.OtherFocusReason)
        super().start_computation()

    def on_n_attrs_changed(self) -> None:
        if self.run_state.state == RunState.Running:
            self.pause_computation()

        new_attrs = self.n_attrs_spin.value()
        if new_attrs == self.n_attrs:
            self.set_button_state()
        else:
            self.set_button_state(label=f"Restart with {new_attrs} variables",
                                  enabled=True)

    def emit_run_state_changed(self) -> None:
        self.runStateChanged.emit(self.run_state.state,
                                  {"n_attrs":self.n_attrs})

    def closeEvent(self, event) -> None:
        self.n_attrs_spin.setValue(self.n_attrs)
        super().closeEvent(event)
