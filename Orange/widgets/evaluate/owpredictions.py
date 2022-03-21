from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from itertools import chain
from typing import Set, Sequence, Union, Optional, List, NamedTuple

import numpy
from AnyQt.QtWidgets import (
    QTableView, QSplitter, QToolTip, QStyle, QApplication, QSizePolicy,
    QPushButton)
from AnyQt.QtGui import QPainter, QStandardItem, QPen, QColor
from AnyQt.QtCore import (
    Qt, QSize, QRect, QRectF, QPoint, QLocale,
    QModelIndex, pyqtSignal, QTimer,
    QItemSelectionModel, QItemSelection)

from orangewidget.report import plural
from orangewidget.utils.itemmodels import AbstractSortTableModel

import Orange
from Orange.evaluation import Results
from Orange.base import Model
from Orange.data import ContinuousVariable, DiscreteVariable, Domain
from Orange.data.table import DomainTransformationError
from Orange.data.util import get_unique_names
from Orange.widgets import gui, settings
from Orange.widgets.evaluate.utils import (
    ScoreTable, usable_scorers, learner_name, scorer_caller)
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Msg, Input, Output, MultiInput
from Orange.widgets.utils.itemmodels import TableModel
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.utils.colorpalettes import LimitedDiscretePalette
from Orange.widgets.utils.itemdelegates import TableDataDelegate

# Input slot for the Predictors channel
PredictorSlot = NamedTuple(
    "PredictorSlot", [
        ("predictor", Model),  # The `Model` instance
        ("name", str),       # Predictor name
        ("results", Optional[Results]),    # Computed prediction results or None.
])


class OWPredictions(OWWidget):
    name = "Predictions"
    icon = "icons/Predictions.svg"
    priority = 200
    description = "Display predictions of models for an input dataset."
    keywords = []

    want_control_area = False

    class Inputs:
        data = Input("Data", Orange.data.Table)
        predictors = MultiInput("Predictors", Model, filter_none=True)

    class Outputs:
        predictions = Output("Predictions", Orange.data.Table)
        evaluation_results = Output("Evaluation Results", Results)

    class Warning(OWWidget.Warning):
        empty_data = Msg("Empty dataset")
        wrong_targets = Msg(
            "Some model(s) predict a different target (see more ...)\n{}")
        missing_targets = Msg("Instances with missing targets "
                              "are ignored while scoring.")

    class Error(OWWidget.Error):
        predictor_failed = Msg("Some predictor(s) failed (see more ...)\n{}")
        scorer_failed = Msg("Some scorer(s) failed (see more ...)\n{}")

    settingsHandler = settings.ClassValuesContextHandler()
    score_table = settings.SettingProvider(ScoreTable)

    #: List of selected class value indices in the `class_values` list
    PROB_OPTS = ["(None)",
                 "Classes in data", "Classes known to the model", "Classes in data and model"]
    PROB_TOOLTIPS = ["Don't show probabilities",
                     "Show probabilities for classes in the data",
                     "Show probabilities for classes known to the model,\n"
                     "including those that don't appear in this data",
                     "Show probabilities for classes in data that are also\n"
                     "known to the model"
                     ]
    NO_PROBS, DATA_PROBS, MODEL_PROBS, BOTH_PROBS = range(4)
    shown_probs = settings.ContextSetting(NO_PROBS)
    selection = settings.Setting([], schema_only=True)
    show_scores = settings.Setting(True)
    TARGET_AVERAGE = "(Average over classes)"
    target_class = settings.ContextSetting(TARGET_AVERAGE)

    def __init__(self):
        super().__init__()

        self.data = None  # type: Optional[Orange.data.Table]
        self.predictors = []  # type: List[PredictorSlot]
        self.class_values = []  # type: List[str]
        self._delegates = []
        self.scorer_errors = []
        self.left_width = 10
        self.selection_store = None
        self.__pending_selection = self.selection

        self._prob_controls = []

        predopts = gui.hBox(
            None, sizePolicy=(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))
        self._prob_controls = [
            gui.widgetLabel(predopts, "Show probabilities for"),
            gui.comboBox(
                predopts, self, "shown_probs", contentsLength=30,
                callback=self._update_prediction_delegate)
        ]
        gui.rubber(predopts)
        self.reset_button = button = QPushButton("Restore Original Order")
        button.clicked.connect(self._reset_order)
        button.setToolTip("Show rows in the original order")
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        predopts.layout().addWidget(self.reset_button)

        self.score_opt_box = scoreopts = gui.hBox(
            None, sizePolicy=(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))
        gui.checkBox(
            scoreopts, self, "show_scores", "Show perfomance scores",
            callback=self._update_score_table_visibility
        )
        gui.separator(scoreopts, 32)
        self._target_controls = [
            gui.widgetLabel(scoreopts, "Target class:"),
            gui.comboBox(
            scoreopts, self, "target_class", items=[], contentsLength=30,
            sendSelectedValue=True, callback=self._on_target_changed)
        ]
        gui.rubber(scoreopts)

        table_opts = dict(horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
                          horizontalScrollMode=QTableView.ScrollPerPixel,
                          selectionMode=QTableView.ExtendedSelection,
                          focusPolicy=Qt.StrongFocus)
        self.dataview = TableView(
            sortingEnabled=True,
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
            **table_opts)
        self.predictionsview = TableView(
            sortingEnabled=True,
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOff,
            **table_opts)
        self.dataview.verticalHeader().hide()
        dsbar = self.dataview.verticalScrollBar()
        psbar = self.predictionsview.verticalScrollBar()
        psbar.valueChanged.connect(dsbar.setValue)
        dsbar.valueChanged.connect(psbar.setValue)

        self.dataview.verticalHeader().setDefaultSectionSize(22)
        self.predictionsview.verticalHeader().setDefaultSectionSize(22)
        self.dataview.verticalHeader().sectionResized.connect(
            lambda index, _, size:
            self.predictionsview.verticalHeader().resizeSection(index, size))

        self.dataview.setItemDelegate(DataItemDelegate(self.dataview))

        self.splitter = QSplitter(
            orientation=Qt.Horizontal, childrenCollapsible=False, handleWidth=2)
        self.splitter.splitterMoved.connect(self.splitter_resized)
        self.splitter.addWidget(self.predictionsview)
        self.splitter.addWidget(self.dataview)

        self.score_table = ScoreTable(self)
        self.mainArea.layout().setSpacing(0)
        self.mainArea.layout().setContentsMargins(4, 0, 4, 4)
        self.mainArea.layout().addWidget(predopts)
        self.mainArea.layout().addWidget(self.splitter)
        self.mainArea.layout().addWidget(scoreopts)
        self.mainArea.layout().addWidget(self.score_table.view)

    def get_selection_store(self, model):
        # Both models map the same, so it doesn't matter which one is used
        # to initialize SharedSelectionStore
        if self.selection_store is None:
            self.selection_store = SharedSelectionStore(model)
        return self.selection_store

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        self.Warning.empty_data(shown=data is not None and not data)
        self.closeContext()
        self.data = data
        self.selection_store = None
        if not data:
            self.dataview.setModel(None)
            self.predictionsview.setModel(None)
        else:
            # force full reset of the view's HeaderView state
            self.dataview.setModel(None)
            model = DataModel(data, parent=None)
            self.dataview.setModel(model)
            sel_model = SharedSelectionModel(
                self.get_selection_store(model), model, self.dataview)
            self.dataview.setSelectionModel(sel_model)
            if self.__pending_selection is not None:
                self.selection = self.__pending_selection
                self.__pending_selection = None
                self.selection_store.select_rows(
                    set(self.selection), QItemSelectionModel.ClearAndSelect)
            sel_model.selectionChanged.connect(self.commit)
            sel_model.selectionChanged.connect(self._store_selection)

            self.dataview.model().list_sorted.connect(
                partial(
                    self._update_data_sort_order, self.dataview,
                    self.predictionsview))

        self._set_target_combos()
        if self.is_discrete_class:
            self.openContext(self.class_var.values)
        self._invalidate_predictions()

    def _store_selection(self):
        self.selection = list(self.selection_store.rows)

    @property
    def class_var(self):
        return self.data is not None and self.data.domain.class_var

    @property
    def is_discrete_class(self):
        return bool(self.class_var) and self.class_var.is_discrete

    @Inputs.predictors
    def set_predictor(self, index, predictor: Model):
        item = self.predictors[index]
        self.predictors[index] = item._replace(
            predictor=predictor, name=predictor.name, results=None
        )

    @Inputs.predictors.insert
    def insert_predictor(self, index, predictor: Model):
        item = PredictorSlot(predictor, predictor.name, None)
        self.predictors.insert(index, item)

    @Inputs.predictors.remove
    def remove_predictor(self, index):
        self.predictors.pop(index)

    def _set_target_combos(self):
        prob_combo = self.controls.shown_probs
        target_combo = self.controls.target_class
        prob_combo.clear()
        target_combo.clear()

        self._update_control_visibility()

        # Set these to prevent warnings when setting self.shown_probs
        target_combo.addItem(self.TARGET_AVERAGE)
        prob_combo.addItems(self.PROB_OPTS)

        if self.is_discrete_class:
            target_combo.addItems(self.class_var.values)
            prob_combo.addItems(self.class_var.values)
            for i, tip in enumerate(self.PROB_TOOLTIPS):
                prob_combo.setItemData(i, tip, Qt.ToolTipRole)
            self.shown_probs = self.DATA_PROBS
            self.target_class = self.TARGET_AVERAGE
        else:
            self.shown_probs = self.NO_PROBS

    def _update_control_visibility(self):
        for widget in self._prob_controls:
            widget.setVisible(self.is_discrete_class)

        for widget in self._target_controls:
            widget.setVisible(self.is_discrete_class and self.show_scores)

        self.score_opt_box.setVisible(bool(self.class_var))

    def _set_class_values(self):
        self.class_values = []
        if self.is_discrete_class:
            self.class_values += self.data.domain.class_var.values
        for slot in self.predictors:
            class_var = slot.predictor.domain.class_var
            if class_var and class_var.is_discrete:
                for value in class_var.values:
                    if value not in self.class_values:
                        self.class_values.append(value)

    def handleNewSignals(self):
        # Disconnect the model: the model and the delegate will be inconsistent
        # between _set_class_values and update_predictions_model.
        self.predictionsview.setModel(None)
        self._set_class_values()
        self._call_predictors()
        self._update_scores()
        self._update_predictions_model()
        self._update_prediction_delegate()
        self._set_errors()
        self.commit()

    def _on_target_changed(self):
        self._update_scores()

    def _call_predictors(self):
        if not self.data:
            return
        if self.class_var:
            domain = self.data.domain
            classless_data = self.data.transform(
                Domain(domain.attributes, None, domain.metas))
        else:
            classless_data = self.data

        for index, slot in enumerate(self.predictors):
            if isinstance(slot.results, Results):
                continue

            predictor = slot.predictor
            try:
                class_var = predictor.domain.class_var
                if class_var and predictor.domain.class_var.is_discrete:
                    pred, prob = predictor(classless_data, Model.ValueProbs)
                else:
                    pred = predictor(classless_data, Model.Value)
                    prob = numpy.zeros((len(pred), 0))
            except (ValueError, DomainTransformationError) as err:
                self.predictors[index] = \
                    slot._replace(results=f"{predictor.name}: {err}")
                continue

            results = Results()
            results.data = self.data
            results.domain = self.data.domain
            results.row_indices = numpy.arange(len(self.data))
            results.folds = (Ellipsis, )
            results.actual = self.data.Y
            results.unmapped_probabilities = prob
            results.unmapped_predicted = pred
            results.probabilities = results.predicted = None
            self.predictors[index] = slot._replace(results=results)

            target = predictor.domain.class_var
            if target != self.class_var:
                continue

            if target is not self.class_var and target.is_discrete:
                backmappers, n_values = predictor.get_backmappers(self.data)
                prob = predictor.backmap_probs(prob, n_values, backmappers)
                pred = predictor.backmap_value(pred, prob, n_values, backmappers)
            results.predicted = pred.reshape((1, len(self.data)))
            results.probabilities = prob.reshape((1,) + prob.shape)

    def _update_scores(self):
        model = self.score_table.model
        if self.is_discrete_class and self.target_class != self.TARGET_AVERAGE:
            target = self.class_var.values.index(self.target_class)
        else:
            target = None
        model.clear()
        scorers = usable_scorers(self.data.domain) if self.data else []
        self.score_table.update_header(scorers)
        self.scorer_errors = errors = []
        for pred in self.predictors:
            results = pred.results
            if not isinstance(results, Results) or results.predicted is None:
                continue
            row = [QStandardItem(learner_name(pred.predictor)),
                   QStandardItem("N/A"), QStandardItem("N/A")]
            try:
                actual = results.actual
                predicted = results.predicted
                probabilities = results.probabilities

                if self.class_var:
                    mask = numpy.isnan(results.actual)
                else:
                    mask = numpy.any(numpy.isnan(results.actual), axis=1)
                no_targets = mask.sum() == len(results.actual)
                results.actual = results.actual[~mask]
                results.predicted = results.predicted[:, ~mask]
                results.probabilities = results.probabilities[:, ~mask]

                for scorer in scorers:
                    item = QStandardItem()
                    if no_targets:
                        item.setText("NA")
                    else:
                        try:
                            score = scorer_caller(scorer, results,
                                                  target=target)()[0]
                            item.setText(f"{score:.3f}")
                        except Exception as exc:  # pylint: disable=broad-except
                            item.setToolTip(str(exc))
                            # false pos.; pylint: disable=unsupported-membership-test
                            if scorer.name in self.score_table.shown_scores:
                                errors.append(str(exc))
                    row.append(item)
                self.score_table.model.appendRow(row)

            finally:
                results.actual = actual
                results.predicted = predicted
                results.probabilities = probabilities

        self._update_score_table_visibility()

    def _update_score_table_visibility(self):
        self._update_control_visibility()
        view = self.score_table.view
        nmodels = self.score_table.model.rowCount()
        if nmodels and self.show_scores:
            view.setVisible(True)
            view.ensurePolished()
            view.resizeColumnsToContents()
            view.resizeRowsToContents()
            view.setFixedHeight(
                5 + view.horizontalHeader().height() +
                view.verticalHeader().sectionSize(0) * nmodels)

            errors = "\n".join(self.scorer_errors)
            self.Error.scorer_failed(errors, shown=bool(errors))
        else:
            view.setVisible(False)
            self.Error.scorer_failed.clear()

    def _set_errors(self):
        # Not all predictors are run every time, so errors can't be collected
        # in _call_predictors
        errors = "\n".join(
            f"- {p.predictor.name}: {p.results}"
            for p in self.predictors
            if isinstance(p.results, str) and p.results)
        self.Error.predictor_failed(errors, shown=bool(errors))

        if self.class_var:
            inv_targets = "\n".join(
                f"- {pred.name} predicts '{pred.domain.class_var.name}'"
                for pred in (p.predictor for p in self.predictors
                             if isinstance(p.results, Results)
                             and p.results.probabilities is None))
            self.Warning.wrong_targets(inv_targets, shown=bool(inv_targets))

            show_warning = numpy.isnan(self.data.Y).any() and self.predictors
            self.Warning.missing_targets(shown=show_warning)
        else:
            self.Warning.wrong_targets.clear()
            self.Warning.missing_targets.clear()

    def _get_details(self):
        details = "Data:<br>"
        details += format_summary_details(self.data, format=Qt.RichText)
        details += "<hr>"
        pred_names = [v.name for v in self.predictors]
        n_predictors = len(self.predictors)
        if n_predictors:
            n_valid = len(self._non_errored_predictors())
            details += plural("Model: {number} model{s}", n_predictors)
            if n_valid != n_predictors:
                details += f" ({n_predictors - n_valid} failed)"
            details += "<ul>"
            for name in pred_names:
                details += f"<li>{name}</li>"
            details += "</ul>"
        else:
            details += "Model:<br>No model on input."
        return details

    def _invalidate_predictions(self):
        for i, pred in enumerate(self.predictors):
            self.predictors[i] = pred._replace(results=None)

    def _non_errored_predictors(self):
        return [p for p in self.predictors
                if isinstance(p.results, Results)]

    def _reordered_probabilities(self, prediction):
        cur_values = prediction.predictor.domain.class_var.values
        new_ind = [self.class_values.index(x) for x in cur_values]
        probs = prediction.results.unmapped_probabilities
        new_probs = numpy.full(
            (probs.shape[0], len(self.class_values)), numpy.nan)
        new_probs[:, new_ind] = probs
        return new_probs

    def _update_predictions_model(self):
        headers = []
        all_values = []
        all_probs = []
        for p in self._non_errored_predictors():
            values = p.results.unmapped_predicted
            target = p.predictor.domain.class_var
            if target and target.is_discrete:
                # order probabilities in order from Show prob. for
                prob = self._reordered_probabilities(p)
                values = numpy.array(target.values)[values.astype(int)]
            else:
                prob = numpy.zeros((len(values), 0))
            all_values.append(values)
            all_probs.append(prob)
            headers.append(p.predictor.name)

        if all_values:
            model = PredictionsModel(all_values, all_probs, headers)
            model.list_sorted.connect(
                partial(
                    self._update_data_sort_order, self.predictionsview,
                    self.dataview))
        else:
            model = None

        if self.selection_store is not None:
            self.selection_store.unregister(
                self.predictionsview.selectionModel())

        self.predictionsview.setModel(model)
        if model is not None:
            self.predictionsview.setSelectionModel(
                SharedSelectionModel(self.get_selection_store(model),
                                     model, self.predictionsview))

        hheader = self.predictionsview.horizontalHeader()
        hheader.setSortIndicatorShown(False)
        hheader.setSectionsClickable(True)

    def _update_data_sort_order(self, sort_source_view, sort_dest_view):
        sort_source = sort_source_view.model()
        sort_dest = sort_dest_view.model()

        sort_source_view.horizontalHeader().setSortIndicatorShown(
            sort_source.sortColumn() != -1)
        sort_dest_view.horizontalHeader().setSortIndicatorShown(False)

        if sort_dest is not None:
            if sort_source is not None and sort_source.sortColumn() >= 0:
                sort_dest.setSortIndices(sort_source.mapToSourceRows(...))
            else:
                sort_dest.setSortIndices(None)
        self.commit()

    def _reset_order(self):
        datamodel = self.dataview.model()
        predmodel = self.predictionsview.model()
        if datamodel is not None:
            datamodel.setSortIndices(None)
            datamodel.sort(-1)
        if predmodel is not None:
            predmodel.setSortIndices(None)
            predmodel.sort(-1)
        self.predictionsview.horizontalHeader().setSortIndicatorShown(False)
        self.dataview.horizontalHeader().setSortIndicatorShown(False)

    def _all_color_values(self):
        """
        Return list of colors together with their values from all predictors
        classes. Colors and values are sorted according to the values order
        for simpler comparison.
        """
        predictors = self._non_errored_predictors()
        color_values = [
            list(zip(*sorted(zip(
                p.predictor.domain.class_var.colors,
                p.predictor.domain.class_var.values
            ), key=itemgetter(1))))
            for p in predictors
            if p.predictor.domain.class_var and
            p.predictor.domain.class_var.is_discrete
        ]
        return color_values if color_values else [([], [])]

    @staticmethod
    def _colors_match(colors1, values1, color2, values2):
        """
        Test whether colors for values match. Colors matches when all
        values match for shorter list and colors match for shorter list.
        It is assumed that values will be sorted together with their colors.
        """
        shorter_length = min(len(colors1), len(color2))
        return (values1[:shorter_length] == values2[:shorter_length]
                and (numpy.array(colors1[:shorter_length]) ==
                     numpy.array(color2[:shorter_length])).all())

    def _get_colors(self):
        """
        Defines colors for values. If colors match in all models use the union
        otherwise use standard colors.
        """
        all_colors_values = self._all_color_values()
        base_color, base_values = all_colors_values[0]
        for c, v in all_colors_values[1:]:
            if not self._colors_match(base_color, base_values, c, v):
                base_color = []
                break
            # replace base_color if longer
            if len(v) > len(base_color):
                base_color = c
                base_values = v

        if len(base_color) != len(self.class_values):
            return LimitedDiscretePalette(len(self.class_values)).palette
        # reorder colors to widgets order
        colors = [None] * len(self.class_values)
        for c, v in zip(base_color, base_values):
            colors[self.class_values.index(v)] = c
        return colors

    def _update_prediction_delegate(self):
        self._delegates.clear()
        colors = self._get_colors()
        shown_class = ""  # just to silence warnings about undefined var
        if self.shown_probs == self.NO_PROBS:
            tooltip_probs = ()
        elif self.shown_probs == self.DATA_PROBS:
            tooltip_probs = self.class_var.values
        elif self.shown_probs >= len(self.PROB_OPTS):
            shown_class = self.class_var.values[self.shown_probs
                                                - len(self.PROB_OPTS)]
            tooltip_probs = (shown_class, )
        sort_col_indices = []
        for col, slot in enumerate(self.predictors):
            target = slot.predictor.domain.class_var
            if target is not None and target.is_discrete:
                shown_probs = self._shown_prob_indices(target, in_target=True)
                if self.shown_probs in (self.MODEL_PROBS, self.BOTH_PROBS):
                    tooltip_probs = [self.class_values[i]
                                     for i in shown_probs if i is not None]
                delegate = PredictionsItemDelegate(
                    self.class_values, colors, shown_probs, tooltip_probs,
                    parent=self.predictionsview)
                sort_col_indices.append([col for col in shown_probs
                                         if col is not None])

            else:
                delegate = PredictionsItemDelegate(
                    None, colors, (), (), target.format_str if target is not None else None,
                    parent=self.predictionsview)
                sort_col_indices.append(None)

            # QAbstractItemView does not take ownership of delegates, so we must
            self._delegates.append(delegate)
            self.predictionsview.setItemDelegateForColumn(col, delegate)
            self.predictionsview.setColumnHidden(col, False)

        self.predictionsview.resizeColumnsToContents()
        self._recompute_splitter_sizes()
        if self.predictionsview.model() is not None:
            self.predictionsview.model().setProbInd(sort_col_indices)

    def _shown_prob_indices(self, target: DiscreteVariable, in_target):
        if self.shown_probs == self.NO_PROBS:
            values = []
        elif self.shown_probs == self.DATA_PROBS:
            values = self.class_var.values
        elif self.shown_probs == self.MODEL_PROBS:
            values = target.values
        elif self.shown_probs == self.BOTH_PROBS:
            # Don't use set intersection because it's unordered!
            values = (value for value in self.class_var.values
                      if value in target.values)
        else:
            shown_cls_idx = self.shown_probs - len(self.PROB_OPTS)
            values = [self.class_var.values[shown_cls_idx]]

        return [self.class_values.index(value)
                if not in_target or value in target.values
                else None
                for value in values]

    def _recompute_splitter_sizes(self):
        if not self.data:
            return
        view = self.predictionsview
        self.left_width = \
            view.horizontalHeader().length() + view.verticalHeader().width()
        self._update_splitter()

    def _update_splitter(self):
        w1, w2 = self.splitter.sizes()
        self.splitter.setSizes([self.left_width, w1 + w2 - self.left_width])

    def splitter_resized(self):
        self.left_width = self.splitter.sizes()[0]

    def commit(self):
        self._commit_predictions()
        self._commit_evaluation_results()

    def _commit_evaluation_results(self):
        slots = [p for p in self._non_errored_predictors()
                 if p.results.predicted is not None]
        if not slots or not self.class_var:
            self.Outputs.evaluation_results.send(None)
            return

        nanmask = numpy.isnan(self.data.get_column_view(self.class_var)[0])
        data = self.data[~nanmask]
        results = Results(data, store_data=True)
        results.folds = [...]
        results.models = numpy.array([[p.predictor for p in self.predictors]])
        results.row_indices = numpy.arange(len(data))
        results.actual = data.Y.ravel()
        results.predicted = numpy.vstack(
            tuple(p.results.predicted[0][~nanmask] for p in slots))
        if self.is_discrete_class:
            results.probabilities = numpy.array(
                [p.results.probabilities[0][~nanmask] for p in slots])
        results.learner_names = [p.name for p in slots]
        self.Outputs.evaluation_results.send(results)

    def _commit_predictions(self):
        if not self.data:
            self.Outputs.predictions.send(None)
            return

        newmetas = []
        newcolumns = []
        for slot in self._non_errored_predictors():
            target = slot.predictor.domain.class_var
            if target and target.is_discrete:
                self._add_classification_out_columns(slot, newmetas, newcolumns)
            else:
                self._add_regression_out_columns(slot, newmetas, newcolumns)

        attrs = list(self.data.domain.attributes)
        metas = list(self.data.domain.metas)
        names = [var.name for var in chain(attrs, self.data.domain.class_vars, metas) if var]
        uniq_newmetas = []
        for new_ in newmetas:
            uniq = get_unique_names(names, new_.name)
            if uniq != new_.name:
                new_ = new_.copy(name=uniq)
            uniq_newmetas.append(new_)
            names.append(uniq)

        metas += uniq_newmetas
        domain = Orange.data.Domain(attrs, self.data.domain.class_vars, metas=metas)
        predictions = self.data.transform(domain)
        if newcolumns:
            newcolumns = numpy.hstack(
                [col.reshape((-1, 1)) for col in newcolumns])
            with predictions.unlocked(predictions.metas):
                predictions.metas[:, -newcolumns.shape[1]:] = newcolumns

        datamodel = self.dataview.model()
        predmodel = self.predictionsview.model()
        assert datamodel is not None  # because we have data
        assert self.selection_store is not None
        rows = numpy.array(list(self.selection_store.rows))
        if rows.size:
            # Reorder rows as they are ordered in view
            shown_rows = datamodel.mapFromSourceRows(rows)
            rows = rows[numpy.argsort(shown_rows)]
            predictions = predictions[rows]
        elif datamodel.sortColumn() >= 0 \
                or predmodel is not None and predmodel.sortColumn() > 0:
            # No selection: output all, but in the shown order
            predictions = predictions[datamodel.mapToSourceRows(...)]
        self.Outputs.predictions.send(predictions)

    def _add_classification_out_columns(self, slot, newmetas, newcolumns):
        pred = slot.predictor
        name = pred.name
        values = pred.domain.class_var.values
        probs = slot.results.unmapped_probabilities

        # Column with class prediction
        newmetas.append(DiscreteVariable(name=name, values=values))
        newcolumns.append(slot.results.unmapped_predicted)

        # Columns with probability predictions (same as shown in the view)
        for cls_idx in self._shown_prob_indices(pred.domain.class_var,
                                                in_target=False):
            value = self.class_values[cls_idx]
            newmetas.append(ContinuousVariable(f"{name} ({value})"))
            if value in values:
                newcolumns.append(probs[:, values.index(value)])
            else:
                newcolumns.append(numpy.zeros(probs.shape[0]))

    @staticmethod
    def _add_regression_out_columns(slot, newmetas, newcolumns):
        newmetas.append(ContinuousVariable(name=slot.predictor.name))
        newcolumns.append(slot.results.unmapped_predicted)

    def send_report(self):
        def merge_data_with_predictions():
            data_model = self.dataview.model()
            predictions_view = self.predictionsview
            predictions_model = predictions_view.model()

            # use ItemDelegate to style prediction values
            delegates = [predictions_view.itemDelegateForColumn(i)
                         for i in range(predictions_model.columnCount())]

            # iterate only over visible columns of data's QTableView
            iter_data_cols = list(filter(lambda x: not self.dataview.isColumnHidden(x),
                                         range(data_model.columnCount())))

            # print header
            yield [''] + \
                  [predictions_model.headerData(col, Qt.Horizontal, Qt.DisplayRole)
                   for col in range(predictions_model.columnCount())] + \
                  [data_model.headerData(col, Qt.Horizontal, Qt.DisplayRole)
                   for col in iter_data_cols]

            # print data & predictions
            for i in range(data_model.rowCount()):
                yield [data_model.headerData(i, Qt.Vertical, Qt.DisplayRole)] + \
                      [delegate.displayText(
                          predictions_model.data(predictions_model.index(i, j)),
                          QLocale())
                       for j, delegate in enumerate(delegates)] + \
                      [data_model.data(data_model.index(i, j),
                                       role=Qt.DisplayRole)
                       for j in iter_data_cols]

        if self.data:
            text = self._get_details().replace('\n', '<br>')
            if self.is_discrete_class and self.shown_probs != self.NO_PROBS:
                text += '<br>Showing probabilities for '
                if self.shown_probs == self.MODEL_PROBS:
                    text += "all classes known to the model"
                elif self.shown_probs == self.DATA_PROBS:
                    text += "all classes that appear in the data"
                elif self.shown_probs == self.BOTH_PROBS:
                    text += "all classes that appear in the data " \
                            "and are known to the model"
                else:
                    class_idx = self.shown_probs - len(self.PROB_OPTS)
                    text += f"'{self.class_var.values[class_idx]}'"
            self.report_paragraph('Info', text)
            self.report_table("Data & Predictions", merge_data_with_predictions(),
                              header_rows=1, header_columns=1)

            self.report_name("Scores")
            if self.is_discrete_class:
                self.report_items([("Target class", self.target_class)])
            self.report_table(self.score_table.view)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_splitter()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._update_splitter)


class ItemDelegate(TableDataDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if self.parent().selectionModel().isSelected(index):
            option.state |= QStyle.State_Selected
        if self.parent().window().isActiveWindow():
            option.state |= QStyle.State_Active | QStyle.State_HasFocus


class DataItemDelegate(ItemDelegate):
    pass


class PredictionsItemDelegate(ItemDelegate):
    """
    A Item Delegate for custom formatting of predictions/probabilities
    """
    #: Roles supplied by the `PredictionsModel`
    DefaultRoles = (Qt.DisplayRole, )

    def __init__(
            self, class_values, colors, shown_probabilities=(),
            tooltip_probabilities=(),
            target_format=None, parent=None,
    ):
        super().__init__(parent)
        self.class_values = class_values  # will be None for continuous
        self.colors = [QColor(*c) for c in colors]
        # target format for cont. vars
        self.target_format = target_format if target_format else '%.2f'
        self.shown_probabilities = self.fmt = self.tooltip = None  # set below
        self.setFormat(shown_probabilities, tooltip_probabilities)

    def setFormat(self, shown_probabilities=(), tooltip_probabilities=()):
        self.shown_probabilities = shown_probabilities
        if self.class_values is None:
            # is continuous class
            self.fmt = f"{{value:{self.target_format[1:]}}}"
        else:
            self.fmt = " \N{RIGHTWARDS ARROW} ".join(
                [" : ".join(f"{{dist[{i}]:.2f}}" if i is not None else "-"
                            for i in shown_probabilities)]
                * bool(shown_probabilities)
                + ["{value!s}"])
        if tooltip_probabilities:
            self.tooltip = f"p({', '.join(tooltip_probabilities)})"
        else:
            self.tooltip = ""

    def displayText(self, value, _):
        try:
            value, dist = value
        except ValueError:
            return ""
        else:
            return self.fmt.format(value=value, dist=dist)

    def helpEvent(self, event, view, option, index):
        if self.tooltip is not None:
            # ... but can be an empty string, so the current tooltip is removed
            QToolTip.showText(event.globalPos(), self.tooltip, view)
            return True
        else:
            return super().helpEvent(event, view, option, index)

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if self.class_values is None:
            option.displayAlignment = \
                (option.displayAlignment & Qt.AlignVertical_Mask) | \
                Qt.AlignRight

    def sizeHint(self, option, index):
        # reimplemented
        sh = super().sizeHint(option, index)
        if option.widget is not None:
            style = option.widget.style()
        else:
            style = QApplication.style()
        margin = style.pixelMetric(
            QStyle.PM_FocusFrameHMargin, option, option.widget) + 1
        metrics = option.fontMetrics
        height = sh.height() + metrics.leading() + 2 * margin
        return QSize(sh.width(), height)

    def distribution(self, index):
        value = self.cachedData(index, Qt.DisplayRole)
        if isinstance(value, tuple) and len(value) == 2:
            _, dist = value
            return dist
        else:
            return None

    def paint(self, painter, option, index):
        dist = self.distribution(index)
        if dist is None or self.colors is None:
            super().paint(painter, option, index)
            return

        if option.widget is not None:
            style = option.widget.style()
        else:
            style = QApplication.style()

        self.initStyleOption(option, index)

        text = option.text
        metrics = option.fontMetrics

        margin = style.pixelMetric(
            QStyle.PM_FocusFrameHMargin, option, option.widget) + 1
        bottommargin = min(margin, 1)
        rect = option.rect.adjusted(margin, margin, -margin, -bottommargin)
        option.text = ""
        textrect = style.subElementRect(
            QStyle.SE_ItemViewItemText, option, option.widget)
        # Are the margins included in the subElementRect?? -> No!
        textrect = textrect.adjusted(margin, margin, -margin, -bottommargin)
        spacing = max(metrics.leading(), 1)

        distheight = rect.height() - metrics.height() - spacing
        distheight = min(max(distheight, 2), metrics.height())
        painter.save()
        painter.setClipRect(option.rect)
        painter.setFont(option.font)
        painter.setRenderHint(QPainter.Antialiasing)

        style.drawPrimitive(
            QStyle.PE_PanelItemViewRow, option, painter, option.widget)
        style.drawPrimitive(
            QStyle.PE_PanelItemViewItem, option, painter, option.widget)

        textrect = textrect.adjusted(0, 0, 0, -distheight - spacing)
        distrect = QRect(
            textrect.bottomLeft() + QPoint(0, spacing),
            QSize(rect.width(), distheight))
        painter.setPen(QPen(Qt.lightGray, 0.3))
        self.drawDistBar(painter, distrect, dist)
        painter.restore()
        if text:
            option.text = text
            self.drawViewItemText(style, painter, option, textrect)

    def drawDistBar(self, painter, rect, distribution):
        painter.save()
        painter.translate(rect.topLeft())
        for i in self.shown_probabilities:
            if i is None:
                continue
            dvalue = distribution[i]
            if not dvalue > 0:  # This also skips nans
                continue
            painter.setBrush(self.colors[i])
            width = rect.width() * dvalue
            painter.drawRoundedRect(QRectF(0, 0, width, rect.height()), 1, 2)
            painter.translate(width, 0.0)
        painter.restore()


class PredictionsModel(AbstractSortTableModel):
    list_sorted = pyqtSignal()

    def __init__(self, values=None, probs=None, headers=None, parent=None):
        super().__init__(parent)
        self._values = values
        self._probs = probs
        self.__probInd = None
        if values is not None:
            assert len(self._values) == len(self._probs) != 0
            sizes = {len(x) for c in (values, probs) for x in c}
            assert len(sizes) == 1
            self.__columnCount = len(values)
            self.__rowCount = sizes.pop()
            if headers is None:
                headers = [None] * self.__columnCount
        else:
            assert probs is None
            assert headers is None
            self.__columnCount = self.__rowCount = 0
        self._header = headers

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.__rowCount

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.__columnCount

    def data(self, index, role=Qt.DisplayRole):
        if role in (Qt.DisplayRole, Qt.EditRole):
            column = index.column()
            row = self.mapToSourceRows(index.row())
            return self._values[column][row], self._probs[column][row]
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Vertical:
                return str(section + 1)
            elif self._header is not None and section < len(self._header):
                return self._header[section]
        return None

    def setProbInd(self, indicess):
        self.__probInd = indicess
        self.sort(self.sortColumn(), self.sortOrder())

    def sortColumnData(self, column):
        values = self._values[column]
        probs = self._probs[column]
        # Let us assume that probs can be None, numpy array or list of arrays
        # self.__probInd[column] can be None (numeric) or empty (no probs
        # shown for particular model)
        if probs is not None and len(probs) and len(probs[0]) \
                and self.__probInd is not None \
                and self.__probInd[column]:
            return probs[:, self.__probInd[column]]
        else:
            return values

    def sort(self, column, order=Qt.AscendingOrder):
        super().sort(column, order)
        self.list_sorted.emit()


# PredictionsModel and DataModel have the same signal and sort method, but
# extracting them into a mixin (because they're derived from different classes)
# would be more complicated and longer than some code repetition.
class DataModel(TableModel):
    list_sorted = pyqtSignal()

    def sort(self, column, order=Qt.AscendingOrder):
        super().sort(column, order)
        self.list_sorted.emit()


class SharedSelectionStore:
    """
    An object shared between multiple selection models

    The object assumes that the underlying models are AbstractSortTableModel.
    Internally, the object stores indices of unmapped, source rows (as int).

    The class implements method `select` with the same signature as
    QItemSelectionModel.select. Selection models that share this object
    must call this method. After changing the selection, this method will
    call `emit_selection_rows_changed` of all selection models, so they
    can emit the signal selectionChanged.
    """
    def __init__(self, model):
        # unmapped indices of selected rows
        self._rows: Set[int] = set()
        self.model: AbstractSortTableModel = model
        self._selection_models: List[SharedSelectionModel] = []

    @property
    def rows(self) -> Set[int]:
        """Indices of selected rows in the source model"""
        return self._rows

    def register(self, selection_model):
        """
        Add a selection mode to the list of models

        Args:
            selection_model (SharedSelectionModel): a new model
        """
        self._selection_models.append(selection_model)

    def unregister(self, selection_model):
        """
        Remove a selection mode to the list of models

        Args:
            selection_model (SharedSelectionModel): a model to remove
        """
        if selection_model in self._selection_models:
            self._selection_models.remove(selection_model)

    def select(self, selection: Union[QModelIndex, QItemSelection], flags: int):
        """
        (De)Select given rows

        Args:
            selection (QModelIndex or QItemSelection):
                rows to select; indices are mapped to rows in the view
            flags (QItemSelectionModel.SelectionFlags):
                flags that tell whether to Clear, Select, Deselect or Toggle
        """
        rows = set()
        if isinstance(selection, QModelIndex):
            if selection.model() is not None:
                rows = {selection.model().mapToSourceRows(selection.row())}
        else:
            indices = selection.indexes()
            if indices:
                map_to = indices[0].model().mapToSourceRows
                rows = set(map_to([index.row() for index in indices]))
        self.select_rows(rows, flags)

    def select_rows(self, rows: Set[int], flags):
        """
        (De)Select given rows

        Args:
            selection (set of int):
                rows to select; indices refer to unmapped rows in model, not view
            flags (QItemSelectionModel.SelectionFlags):
                flags that tell whether to Clear, Select, Deselect or Toggle
        """
        with self._emit_changed():
            if flags & QItemSelectionModel.Clear:
                self._rows.clear()
            if flags & QItemSelectionModel.Select:
                self._rows |= rows
            if flags & QItemSelectionModel.Deselect:
                self._rows -= rows
            if flags & QItemSelectionModel.Toggle:
                self._rows ^= rows

    def clear_selection(self):
        """Clear selection and emit changeSelection signal to all models"""
        with self._emit_changed():
            self._rows.clear()

    def reset(self):
        """Clear selection without emiting a signal,"""
        self._rows.clear()

    @contextmanager
    def _emit_changed(self):
        """
        A context manager that calls `emit_selection_rows_changed after
        changing a selection.
        """
        def map_from_source(rows):
            return self.model.mapFromSourceRows(list(rows))

        old_rows = self._rows.copy()
        try:
            yield
        finally:
            if self.model.rowCount() != 0:
                deselected = map_from_source(old_rows - self._rows)
                selected = map_from_source(self._rows - old_rows)
                if len(selected) != 0 or len(deselected) != 0:
                    for model in self._selection_models:
                        model.emit_selection_rows_changed(selected, deselected)


class SharedSelectionModel(QItemSelectionModel):
    """
    A selection model that shares the selection with its peers.

    It assumes that the underlying model is a AbstractTableModel.
    """
    def __init__(self, shared_store, model, parent):
        super().__init__(model, parent)
        self.store: SharedSelectionStore = shared_store
        self.store.register(self)

    def select(self, selection, flags):
        self.store.select(selection, flags)

    def selection_from_rows(self, rows: Sequence[int]) -> QItemSelection:
        """
        Return selection across all columns for given row indices (as ints)

        Args:
            rows (sequence of int): row indices, as shown in the view, not model

        Returns: QItemSelection
        """
        model = self.model()
        index = model.index
        last_col = model.columnCount() - 1
        sel = QItemSelection()
        for row in rows:
            sel.select(index(row, 0), index(row, last_col))
        return sel

    def emit_selection_rows_changed(
            self, selected: Sequence[int], deselected: Sequence[int]):
        """
        Given a sequence of indices of selected and deselected rows,
        emit a selectionChanged signal.

        Args:
            selected (Sequence[int]): indices of selected rows
            deselected (Sequence[int]): indices of deselected rows
        """
        self.selectionChanged.emit(
            self.selection_from_rows(selected),
            self.selection_from_rows(deselected))

    def selection(self):
        rows = self.model().mapFromSourceRows(list(self.store.rows))
        return self.selection_from_rows(rows)

    def hasSelection(self) -> bool:
        return bool(self.store.rows)

    def isColumnSelected(self, *_) -> bool:
        return len(self.store.rows) == self.model().rowCount()

    def isRowSelected(self, row, _parent=None) -> bool:
        mapped_row = self.model().mapToSourceRows(row)
        return mapped_row in self.store.rows

    rowIntersectsSelection = isRowSelected

    def isSelected(self, index) -> bool:
        return self.model().mapToSourceRows(index.row()) in self.store.rows

    def selectedColumns(self, row: int):
        if self.isColumnSelected():
            index = self.model().index
            return [index(row, col)
                    for col in range(self.model().columnCount())]
        else:
            return []

    def _selected_rows_arr(self):
        return numpy.fromiter(self.store.rows, int, len(self.store.rows))

    def selectedRows(self, col: int):
        index = self.model().index
        rows = self.model().mapFromSourceRows(self._selected_rows_arr())
        return [index(row, col) for row in rows]

    def selectedIndexes(self):
        index = self.model().index
        rows = self.model().mapFromSourceRows(self._selected_rows_arr())
        return [index(row, col)
                for col in range(self.model().columnCount())
                for row in rows]

    def clearSelection(self):
        self.store.clear_selection()

    def reset(self):
        self.store.reset()
        self.clearCurrentIndex()


class TableView(gui.HScrollStepMixin, QTableView):
    MaxSizeHintSamples = 1000

    def sizeHintForColumn(self, column):
        """
        Reimplemented from `QTableView.sizeHintForColumn`

        Note: This does not match the QTableView's implementation,
        in particular size hints from editor/index widgets are not taken
        into account.

        Parameters
        ----------
        column : int
        """
        # This is probably not needed in Qt5?
        if self.model() is None:
            return -1

        self.ensurePolished()
        model = self.model()
        vheader = self.verticalHeader()
        top = vheader.visualIndexAt(0)
        bottom = vheader.visualIndexAt(self.viewport().height())
        if bottom < 0:
            bottom = self.model().rowCount()

        options = self.viewOptions()
        options.widget = self

        width = 0
        sample_count = 0

        for row in range(top, bottom):
            if not vheader.isSectionHidden(vheader.logicalIndex(row)):
                index = model.index(row, column)
                delegate = self.itemDelegate(index)
                if not delegate:
                    continue
                size = delegate.sizeHint(options, index)
                width = max(size.width(), width)
                sample_count += 1

            if sample_count >= TableView.MaxSizeHintSamples:
                break

        return width + 1 if self.showGrid() else width


def tool_tip(value):
    value, dist = value
    if dist is not None:
        return f"{value:!s} {dist:!s}"
    else:
        return str(value)


if __name__ == "__main__":  # pragma: no cover
    filename = "iris.tab"
    iris = Orange.data.Table(filename)
    iris2 = iris[:100]

    def pred_error(data, *args, **kwargs):
        raise ValueError

    pred_error.domain = iris.domain
    pred_error.name = "To err is human"

    if iris.domain.has_discrete_class:
        idom = iris.domain
        dom = Domain(
            idom.attributes,
            DiscreteVariable(idom.class_var.name, idom.class_var.values[1::-1])
        )
        iris2 = iris2.transform(dom)
        predictors_ = [
            Orange.classification.SVMLearner(probability=True)(iris2),
            Orange.classification.LogisticRegressionLearner()(iris),
            pred_error
        ]
    elif iris.domain.has_continuous_class:
        predictors_ = [
            Orange.regression.RidgeRegressionLearner(alpha=1.0)(iris),
            Orange.regression.LinearRegressionLearner()(iris),
            pred_error
        ]
    else:
        predictors_ = [pred_error]

    WidgetPreview(OWPredictions).run(
        set_data=iris,
        insert_predictor=list(enumerate(predictors_)))
