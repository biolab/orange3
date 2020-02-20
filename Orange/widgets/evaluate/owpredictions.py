from collections import namedtuple
from functools import partial
from operator import itemgetter

import numpy
from AnyQt.QtWidgets import (
    QTableView, QListWidget, QSplitter, QStyledItemDelegate,
    QToolTip, QStyle, QApplication, QSizePolicy)
from AnyQt.QtGui import QPainter, QStandardItem, QPen, QColor
from AnyQt.QtCore import (
    Qt, QSize, QRect, QRectF, QPoint, QLocale,
    QModelIndex, QAbstractTableModel, QSortFilterProxyModel, pyqtSignal, QTimer)

import Orange
from Orange.evaluation import Results
from Orange.base import Model
from Orange.data import ContinuousVariable, DiscreteVariable, Value, Domain
from Orange.data.table import DomainTransformationError
from Orange.widgets import gui, settings
from Orange.widgets.evaluate.utils import (
    ScoreTable, usable_scorers, learner_name, scorer_caller)
from Orange.widgets.utils.colorpalette import ColorPaletteGenerator
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.widgets.utils.itemmodels import TableModel
from Orange.widgets.utils.sql import check_sql_input


# Input slot for the Predictors channel
PredictorSlot = namedtuple(
    "PredictorSlot",
    ["predictor",  # The `Model` instance
     "name",       # Predictor name
     "results"]    # Computed prediction results or None.
)


class OWPredictions(OWWidget):
    name = "Predictions"
    icon = "icons/Predictions.svg"
    priority = 200
    description = "Display predictions of models for an input dataset."
    keywords = []

    class Inputs:
        data = Input("Data", Orange.data.Table)
        predictors = Input("Predictors", Model, multiple=True)

    class Outputs:
        predictions = Output("Predictions", Orange.data.Table)
        evaluation_results = Output("Evaluation Results", Results)

    class Warning(OWWidget.Warning):
        empty_data = Msg("Empty dataset")
        wrong_targets = Msg(
            "Some model(s) predict a different target (see more ...)\n{}")

    class Error(OWWidget.Error):
        predictor_failed = Msg("Some predictor(s) failed (see more ...)\n{}")
        scorer_failed = Msg("Some scorer(s) failed (see more ...)\n{}")

    settingsHandler = settings.ClassValuesContextHandler()
    score_table = settings.SettingProvider(ScoreTable)

    #: List of selected class value indices in the `class_values` list
    selected_classes = settings.ContextSetting([])

    def __init__(self):
        super().__init__()

        self.data = None  # type: Optional[Orange.data.Table]
        self.predictors = {}  # type: Dict[object, PredictorSlot]
        self.class_values = []  # type: List[str]
        self._delegates = []
        self.left_width = 10

        gui.listBox(self.controlArea, self, "selected_classes", "class_values",
                    box="Show probabibilities for",
                    callback=self._update_prediction_delegate,
                    selectionMode=QListWidget.ExtendedSelection,
                    addSpace=False,
                    sizePolicy=(QSizePolicy.Preferred, QSizePolicy.Preferred))
        gui.rubber(self.controlArea)
        self.reset_button = gui.button(
            self.controlArea, self, "Restore Original Order",
            callback=self._reset_order,
            tooltip="Show rows in the original order")

        table_opts = dict(horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
                          horizontalScrollMode=QTableView.ScrollPerPixel,
                          selectionMode=QTableView.NoSelection,
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

        self.splitter = QSplitter(
            orientation=Qt.Horizontal, childrenCollapsible=False, handleWidth=2)
        self.splitter.splitterMoved.connect(self.splitter_resized)
        self.splitter.addWidget(self.predictionsview)
        self.splitter.addWidget(self.dataview)

        self.score_table = ScoreTable(self)
        self.vsplitter = gui.vBox(self.mainArea)
        self.vsplitter.layout().addWidget(self.splitter)
        self.vsplitter.layout().addWidget(self.score_table.view)

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        self.Warning.empty_data(shown=data is not None and not data)
        self.data = data
        if not data:
            self.dataview.setModel(None)
            self.predictionsview.setModel(None)
        else:
            # force full reset of the view's HeaderView state
            self.dataview.setModel(None)
            model = TableModel(data, parent=None)
            modelproxy = SortProxyModel()
            modelproxy.setSourceModel(model)
            self.dataview.setModel(modelproxy)

            self.dataview.model().list_sorted.connect(
                partial(
                    self._update_data_sort_order, self.dataview,
                    self.predictionsview))

        self._invalidate_predictions()

    @property
    def class_var(self):
        return self.data and self.data.domain.class_var

    # pylint: disable=redefined-builtin
    @Inputs.predictors
    def set_predictor(self, predictor=None, id=None):
        if id in self.predictors:
            if predictor is not None:
                self.predictors[id] = self.predictors[id]._replace(
                    predictor=predictor, name=predictor.name, results=None)
            else:
                del self.predictors[id]
        elif predictor is not None:
            self.predictors[id] = PredictorSlot(predictor, predictor.name, None)

    def _set_class_values(self):
        class_values = []
        for slot in self.predictors.values():
            class_var = slot.predictor.domain.class_var
            if class_var and class_var.is_discrete:
                for value in class_var.values:
                    if value not in class_values:
                        class_values.append(value)

        if self.class_var and self.class_var.is_discrete:
            values = self.class_var.values
            self.class_values = sorted(
                class_values, key=lambda val: val not in values)
            self.selected_classes = [
                i for i, name in enumerate(class_values) if name in values]
        else:
            self.class_values = class_values  # This assignment updates listview
            self.selected_classes = []

    def handleNewSignals(self):
        self._set_class_values()
        self._call_predictors()
        self._update_scores()
        self._update_predictions_model()
        self._update_prediction_delegate()
        self._set_errors()
        self._update_info()
        self.commit()

    def _call_predictors(self):
        if not self.data:
            return
        if self.class_var:
            domain = self.data.domain
            classless_data = self.data.transform(
                Domain(domain.attributes, None, domain.metas))
        else:
            classless_data = self.data

        for inputid, slot in self.predictors.items():
            if isinstance(slot.results, Results):
                continue

            predictor = slot.predictor
            try:
                if predictor.domain.class_var.is_discrete:
                    pred, prob = predictor(classless_data, Model.ValueProbs)
                else:
                    pred = predictor(classless_data, Model.Value)
                    prob = numpy.zeros((len(pred), 0))
            except (ValueError, DomainTransformationError) as err:
                self.predictors[inputid] = \
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
            self.predictors[inputid] = slot._replace(results=results)

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
        model.clear()
        scorers = usable_scorers(self.class_var) if self.class_var else []
        self.score_table.update_header(scorers)
        errors = []
        for inputid, pred in self.predictors.items():
            results = self.predictors[inputid].results
            if not isinstance(results, Results) or results.predicted is None:
                continue
            row = [QStandardItem(learner_name(pred.predictor)),
                   QStandardItem("N/A"), QStandardItem("N/A")]
            for scorer in scorers:
                item = QStandardItem()
                try:
                    score = scorer_caller(scorer, results)()[0]
                    item.setText(f"{score:.3f}")
                except Exception as exc:  # pylint: disable=broad-except
                    item.setToolTip(str(exc))
                    if scorer.name in self.score_table.shown_scores:
                        errors.append(str(exc))
                row.append(item)
            self.score_table.model.appendRow(row)

        view = self.score_table.view
        if model.rowCount():
            view.setVisible(True)
            view.ensurePolished()
            view.setFixedHeight(
                5 + view.horizontalHeader().height() +
                view.verticalHeader().sectionSize(0) * model.rowCount())
        else:
            view.setVisible(False)

        self.Error.scorer_failed("\n".join(errors), shown=bool(errors))

    def _set_errors(self):
        # Not all predictors are run every time, so errors can't be collected
        # in _call_predictors
        errors = "\n".join(
            f"- {p.predictor.name}: {p.results}"
            for p in self.predictors.values()
            if isinstance(p.results, str) and p.results)
        self.Error.predictor_failed(errors, shown=bool(errors))

        if self.class_var:
            inv_targets = "\n".join(
                f"- {pred.name} predicts '{pred.domain.class_var.name}'"
                for pred in (p.predictor for p in self.predictors.values()
                             if isinstance(p.results, Results)
                             and p.results.probabilities is None))
            self.Warning.wrong_targets(inv_targets, shown=bool(inv_targets))
        else:
            self.Warning.wrong_targets.clear()

    def _update_info(self):
        n_predictors = len(self.predictors)
        if not self.data and not n_predictors:
            self.info.set_input_summary(self.info.NoInput)
            return

        n_valid = len(self._non_errored_predictors())
        summary = str(len(self.data)) if self.data else "0"
        details = f"{len(self.data)} instances" if self.data else "No data"
        details += f"\n{n_predictors} models" if n_predictors else "No models"
        if n_valid != n_predictors:
            details += f" ({n_predictors - n_valid} failed)"
        self.info.set_input_summary(summary, details)

    def _invalidate_predictions(self):
        for inputid, pred in list(self.predictors.items()):
            self.predictors[inputid] = pred._replace(results=None)

    def _non_errored_predictors(self):
        return [p for p in self.predictors.values()
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
        results = []
        headers = []
        for p in self._non_errored_predictors():
            values = p.results.unmapped_predicted
            target = p.predictor.domain.class_var
            if target.is_discrete:
                # order probabilities in order from Show prob. for
                prob = self._reordered_probabilities(p)
                values = [Value(target, v) for v in values]
            else:
                prob = numpy.zeros((len(values), 0))
            results.append((values, prob))
            headers.append(p.predictor.name)

        if results:
            results = list(zip(*(zip(*res) for res in results)))
            model = PredictionsModel(results, headers)
        else:
            model = None

        predmodel = PredictionsSortProxyModel()
        predmodel.setSourceModel(model)
        predmodel.setDynamicSortFilter(True)
        self.predictionsview.setModel(predmodel)
        hheader = self.predictionsview.horizontalHeader()
        hheader.setSortIndicatorShown(False)
        # SortFilterProxyModel is slow due to large abstraction overhead
        # (every comparison triggers multiple `model.index(...)`,
        # model.rowCount(...), `model.parent`, ... calls)
        hheader.setSectionsClickable(predmodel.rowCount() < 20000)

        self.predictionsview.model().list_sorted.connect(
            partial(
                self._update_data_sort_order, self.predictionsview,
                self.dataview))

        self.predictionsview.resizeColumnsToContents()

    @staticmethod
    def _update_data_sort_order(sort_source_view, sort_dest_view):
        sort_dest = sort_dest_view.model()
        sort_source = sort_source_view.model()
        sortindicatorshown = False
        if sort_dest is not None:
            assert isinstance(sort_dest, QSortFilterProxyModel)
            n = sort_dest.rowCount()
            if sort_source is not None and sort_source.sortColumn() >= 0:
                sortind = numpy.argsort(
                    [sort_source.mapToSource(sort_source.index(i, 0)).row()
                     for i in range(n)])
                sortind = numpy.array(sortind, numpy.int)
                sortindicatorshown = True
            else:
                sortind = None

            sort_dest.setSortIndices(sortind)

        sort_dest_view.horizontalHeader().setSortIndicatorShown(
            False)
        sort_source_view.horizontalHeader().setSortIndicatorShown(
            sortindicatorshown)

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
            for p in predictors if p.predictor.domain.class_var.is_discrete
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
            return ColorPaletteGenerator.palette(len(self.class_values))
        # reorder colors to widgets order
        colors = [None] * len(self.class_values)
        for c, v in zip(base_color, base_values):
            colors[self.class_values.index(v)] = c
        return colors

    def _update_prediction_delegate(self):
        self._delegates.clear()
        colors = self._get_colors()
        for col, slot in enumerate(self.predictors.values()):
            target = slot.predictor.domain.class_var
            shown_probs = (
                () if target.is_continuous else
                [val if self.class_values[val] in target.values else None
                 for val in self.selected_classes]
            )
            delegate = PredictionsItemDelegate(
                None if target.is_continuous else self.class_values,
                colors,
                shown_probs,
                target.format_str if target.is_continuous else None
            )
            # QAbstractItemView does not take ownership of delegates, so we must
            self._delegates.append(delegate)
            self.predictionsview.setItemDelegateForColumn(col, delegate)
            self.predictionsview.setColumnHidden(col, False)

        self.predictionsview.resizeColumnsToContents()
        self._recompute_splitter_sizes()
        if self.predictionsview.model() is not None:
            self.predictionsview.model().setProbInd(self.selected_classes)

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
        if not slots:
            self.Outputs.evaluation_results.send(None)
            return

        nanmask = numpy.isnan(self.data.get_column_view(self.class_var)[0])
        data = self.data[~nanmask]
        results = Results(data, store_data=True)
        results.folds = None
        results.row_indices = numpy.arange(len(data))
        results.actual = data.Y.ravel()
        results.predicted = numpy.vstack(
            tuple(p.results.predicted[0][~nanmask] for p in slots))
        if self.class_var and self.class_var.is_discrete:
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
            if slot.predictor.domain.class_var.is_discrete:
                self._add_classification_out_columns(slot, newmetas, newcolumns)
            else:
                self._add_regression_out_columns(slot, newmetas, newcolumns)

        attrs = list(self.data.domain.attributes)
        metas = list(self.data.domain.metas) + newmetas
        domain = Orange.data.Domain(attrs, self.class_var, metas=metas)
        predictions = self.data.transform(domain)
        if newcolumns:
            newcolumns = numpy.hstack(
                [numpy.atleast_2d(cols) for cols in newcolumns])
            predictions.metas[:, -newcolumns.shape[1]:] = newcolumns
        self.Outputs.predictions.send(predictions)

    @staticmethod
    def _add_classification_out_columns(slot, newmetas, newcolumns):
        # Mapped or unmapped predictions?!
        # Or provide a checkbox so the user decides?
        pred = slot.predictor
        name = pred.name
        values = pred.domain.class_var.values
        newmetas.append(DiscreteVariable(name=name, values=values))
        newcolumns.append(slot.results.unmapped_predicted.reshape(-1, 1))
        newmetas += [ContinuousVariable(name=f"{name} ({value})")
                     for value in values]
        newcolumns.append(slot.results.unmapped_probabilities)

    @staticmethod
    def _add_regression_out_columns(slot, newmetas, newcolumns):
        newmetas.append(ContinuousVariable(name=slot.predictor.name))
        newcolumns.append(slot.results.unmapped_predicted.reshape((-1, 1)))

    def send_report(self):
        def merge_data_with_predictions():
            data_model = self.dataview.model()
            predictions_model = self.predictionsview.model()

            # use ItemDelegate to style prediction values
            style = lambda x: self.predictionsview.itemDelegate().displayText(x, QLocale())

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
                      [style(predictions_model.data(predictions_model.index(i, j)))
                       for j in range(predictions_model.columnCount())] + \
                      [data_model.data(data_model.index(i, j))
                       for j in iter_data_cols]

        if self.data:
            text = self.infolabel.text().replace('\n', '<br>')
            if self.selected_classes:
                text += '<br>Showing probabilities for: '
                text += ', '. join([self.class_values[i]
                                    for i in self.selected_classes])
            self.report_paragraph('Info', text)
            self.report_table("Data & Predictions", merge_data_with_predictions(),
                              header_rows=1, header_columns=1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_splitter()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._update_splitter)


class PredictionsItemDelegate(QStyledItemDelegate):
    """
    A Item Delegate for custom formatting of predictions/probabilities
    """
    def __init__(
            self, class_values, colors, shown_probabilities=(),
            target_format=None, parent=None,
    ):
        super().__init__(parent)
        self.class_values = class_values  # will be None for continuous
        self.colors = [QColor(*c) for c in colors]
        self.target_format = target_format  # target format for cont. vars
        self.shown_probabilities = self.fmt = self.tooltip = None  # set below
        self.setFormat(shown_probabilities)

    def setFormat(self, shown_probabilities=()):
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
        self.tooltip = ""
        if shown_probabilities:
            val = ', '.join(
                self.class_values[i] if i is not None else "-"
                for i in shown_probabilities if i is not None
            )
            self.tooltip = f"p({val})"

    def displayText(self, value, _locale):
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

    @staticmethod
    def distribution(index):
        value = index.data(Qt.DisplayRole)
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

        textrect = style.subElementRect(
            QStyle.SE_ItemViewItemText, option, option.widget)
        # Are the margins included in the subElementRect?? -> No!
        textrect = textrect.adjusted(margin, margin, -margin, -bottommargin)

        text = option.fontMetrics.elidedText(
            text, option.textElideMode, textrect.width())

        spacing = max(metrics.leading(), 1)

        distheight = rect.height() - metrics.height() - spacing
        distheight = numpy.clip(distheight, 2, metrics.height())

        painter.save()
        painter.setClipRect(option.rect)
        painter.setFont(option.font)
        painter.setRenderHint(QPainter.Antialiasing)

        style.drawPrimitive(
            QStyle.PE_PanelItemViewRow, option, painter, option.widget)
        style.drawPrimitive(
            QStyle.PE_PanelItemViewItem, option, painter, option.widget)

        if option.state & QStyle.State_Selected:
            color = option.palette.highlightedText().color()
        else:
            color = option.palette.text().color()
        painter.setPen(QPen(color))

        textrect = textrect.adjusted(0, 0, 0, -distheight - spacing)
        distrect = QRect(
            textrect.bottomLeft() + QPoint(0, spacing),
            QSize(rect.width(), distheight))
        painter.setPen(QPen(Qt.lightGray, 0.3))
        self.drawDistBar(painter, distrect, dist)
        painter.restore()
        if text:
            style.drawItemText(
                painter, textrect, option.displayAlignment, option.palette,
                option.state & QStyle.State_Enabled, text)

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


class SortProxyModel(QSortFilterProxyModel):
    """
    QSortFilter model used in both TableView and PredictionsView
    """
    list_sorted = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.__sortInd = None

    def setSortIndices(self, indices):
        if indices is not None:
            indices = numpy.array(indices, dtype=numpy.int)
            if indices.shape != (self.rowCount(),):
                raise ValueError("indices.shape != (self.rowCount(),)")
            indices.flags.writeable = False

        self.__sortInd = indices

        if self.__sortInd is not None:
            self.custom_sort(0)  # need valid order to call lessThan

    def lessThan(self, left, right):
        if self.__sortInd is None:
            return super().lessThan(left, right)

        assert not (left.parent().isValid() or right.parent().isValid()), \
            "Not a table model"

        rleft, rright = left.row(), right.row()
        try:
            ileft, iright = self.__sortInd[rleft], self.__sortInd[rright]
        except IndexError:
            return False
        else:
            return ileft < iright

    def isSorted(self):
        return self.__sortInd is not None

    def sort(self, n, order=Qt.AscendingOrder):
        """
        This sort is called only on click, in other cases we manually call
        custom_sort
        """
        # reset sort - otherwise when same parameters set by lessThan function
        # clicking on header would not trigger resort
        self.__sortInd = None
        self.custom_sort(n, order=order)
        self.list_sorted.emit()

    def custom_sort(self, n, order=Qt.AscendingOrder):
        """
        When sorting is dmanded because of sort change in the other view
        this sorting is called. It will not reset __sortInd to None.
        """
        if self.sortColumn() == n and self.sortOrder() == order:
            self.invalidate()
        else:
            super().sort(n, order)  # need some valid sort column


class PredictionsSortProxyModel(SortProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.__probInd = None

    def setProbInd(self, indices):
        self.__probInd = indices
        self.invalidate()
        self.list_sorted.emit()

    def lessThan(self, left, right):
        if self.isSorted():
            return super().lessThan(left, right)

        role = self.sortRole()
        left_data = self.sourceModel().data(left, role)
        right_data = self.sourceModel().data(right, role)

        return self._key(left_data) < self._key(right_data)

    def _key(self, prediction):
        value, probs = prediction
        if probs is not None:
            if self.__probInd is not None:
                probs = probs[self.__probInd]
            probs = tuple(probs)

        return probs, value


class PredictionsModel(QAbstractTableModel):
    def __init__(self, table=None, headers=None, parent=None):
        super().__init__(parent)
        self._table = [[]] if table is None else table
        if headers is None:
            headers = [None] * len(self._table)
        self._header = headers
        self.__columnCount = max([len(row) for row in self._table] or [0])

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._table)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.__columnCount

    def _value(self, index):
        return self._table[index.row()][index.column()]

    def data(self, index, role=Qt.DisplayRole):
        if role in (Qt.DisplayRole, Qt.EditRole):
            return self._value(index)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return str(section + 1)
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return (self._header[section] if section < len(self._header)
                    else str(section))
        return None


class TableView(QTableView):
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
        return "{!s} {!s}".format(value, dist)
    else:
        return str(value)


if __name__ == "__main__":  # pragma: no cover
    filename = "iris.tab"
    iris = Orange.data.Table(filename)
    idom = iris.domain
    dom = Domain(
        idom.attributes,
        DiscreteVariable(idom.class_var.name, idom.class_var.values[1::-1])
    )
    iris2 = iris[:100].transform(dom)

    def pred_error(data, *args, **kwargs):
        raise ValueError

    pred_error.domain = iris.domain
    pred_error.name = "To err is human"

    if iris.domain.has_discrete_class:
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
        set_data=iris2,
        set_predictor=[(pred, i) for i, pred in enumerate(predictors_)])
