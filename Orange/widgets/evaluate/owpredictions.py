"""
Predictions widget

"""
from collections import OrderedDict, namedtuple

import numpy
from AnyQt.QtWidgets import (
    QTableView, QListWidget, QSplitter, QStyledItemDelegate,
    QToolTip, QAbstractItemView, QStyleOptionViewItem, QStyle,
    QApplication,
)
from AnyQt.QtGui import QPainter, QHelpEvent, QColor, QPen
from AnyQt.QtCore import (
    Qt, QRect, QRectF, QPoint, QSize,
    QModelIndex, QAbstractTableModel, QSortFilterProxyModel,
    QLocale, pyqtSlot as Slot
)

import Orange
import Orange.evaluation

from Orange.base import Model
from Orange.data import ContinuousVariable, DiscreteVariable, Value
from Orange.data.table import DomainTransformationError
from Orange.widgets import gui, settings
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.widgets.utils.itemmodels import TableModel
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils import colorpalette

QWIDGETSIZE_MAX = 16777215


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
    description = "Display the predictions of models for an input dataset."
    keywords = []

    class Inputs:
        data = Input("Data", Orange.data.Table)
        predictors = Input("Predictors", Model, multiple=True)

    class Outputs:
        predictions = Output("Predictions", Orange.data.Table)
        evaluation_results = Output("Evaluation Results",
                                    Orange.evaluation.Results,
                                    dynamic=False)

    class Warning(OWWidget.Warning):
        empty_data = Msg("Empty dataset")

    class Error(OWWidget.Error):
        predictor_failed = \
            Msg("One or more predictors failed (see more...)\n{}")
        predictors_target_mismatch = \
            Msg("Predictors do not have the same target.")
        data_target_mismatch = \
            Msg("Data does not have the same target as predictors.")

    settingsHandler = settings.ClassValuesContextHandler()
    #: Display the full input dataset or only the target variable columns (if
    #: available)
    show_attrs = settings.Setting(True)
    #: Show predicted values (for discrete target variable)
    show_predictions = settings.Setting(True)
    #: Show predictions probabilities (for discrete target variable)
    show_probabilities = settings.Setting(True)
    #: List of selected class value indices in the "Show probabilities" list
    selected_classes = settings.ContextSetting([])
    #: Draw colored distribution bars
    draw_dist = settings.Setting(True)

    output_attrs = settings.Setting(True)
    output_predictions = settings.Setting(True)
    output_probabilities = settings.Setting(True)

    def __init__(self):
        super().__init__()

        #: Input data table
        self.data = None  # type: Optional[Orange.data.Table]
        #: A dict mapping input ids to PredictorSlot
        self.predictors = OrderedDict()  # type: Dict[object, PredictorSlot]
        #: A class variable (prediction target)
        self.class_var = None  # type: Optional[Orange.data.Variable]
        #: List of (discrete) class variable's values
        self.class_values = []  # type: List[str]

        box = gui.vBox(self.controlArea, "Info")
        self.infolabel = gui.widgetLabel(
            box, "No data on input.\nPredictors: 0\nTask: N/A")
        self.infolabel.setMinimumWidth(150)
        gui.button(box, self, "Restore Original Order",
                   callback=self._reset_order,
                   tooltip="Show rows in the original order")

        self.classification_options = box = gui.vBox(
            self.controlArea, "Show", spacing=-1, addSpace=False)

        gui.checkBox(box, self, "show_predictions", "Predicted class",
                     callback=self._update_prediction_delegate)
        b = gui.checkBox(box, self, "show_probabilities",
                         "Predicted probabilities for:",
                         callback=self._update_prediction_delegate)
        ibox = gui.indentedBox(box, sep=gui.checkButtonOffsetHint(b),
                               addSpace=False)
        gui.listBox(ibox, self, "selected_classes", "class_values",
                    callback=self._update_prediction_delegate,
                    selectionMode=QListWidget.MultiSelection,
                    addSpace=False)
        gui.checkBox(box, self, "draw_dist", "Draw distribution bars",
                     callback=self._update_prediction_delegate)

        box = gui.vBox(self.controlArea, "Data View")
        gui.checkBox(box, self, "show_attrs", "Show full dataset",
                     callback=self._update_column_visibility)

        box = gui.vBox(self.controlArea, "Output", spacing=-1)
        self.checkbox_class = gui.checkBox(
            box, self, "output_attrs", "Original data",
            callback=self.commit)
        self.checkbox_class = gui.checkBox(
            box, self, "output_predictions", "Predictions",
            callback=self.commit)
        self.checkbox_prob = gui.checkBox(
            box, self, "output_probabilities", "Probabilities",
            callback=self.commit)

        gui.rubber(self.controlArea)

        self.splitter = QSplitter(
            orientation=Qt.Horizontal,
            childrenCollapsible=False,
            handleWidth=2,
        )
        self.dataview = TableView(
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
            horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
            horizontalScrollMode=QTableView.ScrollPerPixel,
            selectionMode=QTableView.NoSelection,
            focusPolicy=Qt.StrongFocus
        )
        self.predictionsview = TableView(
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOff,
            horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOn,
            horizontalScrollMode=QTableView.ScrollPerPixel,
            selectionMode=QTableView.NoSelection,
            focusPolicy=Qt.StrongFocus,
            sortingEnabled=True,
        )

        self.predictionsview.setItemDelegate(PredictionsItemDelegate())
        self.dataview.verticalHeader().hide()

        dsbar = self.dataview.verticalScrollBar()
        psbar = self.predictionsview.verticalScrollBar()

        psbar.valueChanged.connect(dsbar.setValue)
        dsbar.valueChanged.connect(psbar.setValue)

        self.dataview.verticalHeader().setDefaultSectionSize(22)
        self.predictionsview.verticalHeader().setDefaultSectionSize(22)
        self.dataview.verticalHeader().sectionResized.connect(
            lambda index, _, size:
            self.predictionsview.verticalHeader().resizeSection(index, size)
        )

        self.splitter.addWidget(self.predictionsview)
        self.splitter.addWidget(self.dataview)

        self.mainArea.layout().addWidget(self.splitter)

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        """Set the input dataset"""
        if data is not None and not data:
            data = None
            self.Warning.empty_data()
        else:
            self.Warning.empty_data.clear()

        self.data = data
        if data is None:
            self.dataview.setModel(None)
            self.predictionsview.setModel(None)
            self.predictionsview.setItemDelegate(PredictionsItemDelegate())
        else:
            # force full reset of the view's HeaderView state
            self.dataview.setModel(None)
            model = TableModel(data, parent=None)
            modelproxy = TableSortProxyModel()
            modelproxy.setSourceModel(model)
            self.dataview.setModel(modelproxy)
            self._update_column_visibility()

        self._invalidate_predictions()

    @Inputs.predictors
    # pylint: disable=redefined-builtin
    def set_predictor(self, predictor=None, id=None):
        if id in self.predictors:
            if predictor is not None:
                self.predictors[id] = self.predictors[id]._replace(
                    predictor=predictor, name=predictor.name, results=None)
            else:
                del self.predictors[id]
        elif predictor is not None:
            self.predictors[id] = \
                PredictorSlot(predictor, predictor.name, None)

    def set_class_var(self):
        pred_classes = set(pred.predictor.domain.class_var
                           for pred in self.predictors.values())
        self.Error.predictors_target_mismatch.clear()
        self.Error.data_target_mismatch.clear()
        self.class_var = None
        if len(pred_classes) > 1:
            self.Error.predictors_target_mismatch()
        if len(pred_classes) == 1:
            self.class_var = pred_classes.pop()
            if self.data is not None and \
                    self.data.domain.class_var is not None and \
                    self.class_var != self.data.domain.class_var:
                self.Error.data_target_mismatch()
                self.class_var = None

        discrete_class = self.class_var is not None \
                         and self.class_var.is_discrete
        self.classification_options.setVisible(discrete_class)
        self.closeContext()
        if discrete_class:
            self.class_values = list(self.class_var.values)
            self.selected_classes = list(range(len(self.class_values)))
            self.openContext(self.class_var)
        else:
            self.class_values = []
            self.selected_classes = []

    def handleNewSignals(self):
        self.set_class_var()
        if self.data is not None:
            self._call_predictors()
        self._update_predictions_model()
        self._update_prediction_delegate()
        self._set_errors()
        self._update_info()
        self.commit()

    def _call_predictors(self):
        for inputid, pred in self.predictors.items():
            if pred.results is None \
                    or isinstance(pred.results, str) \
                    or numpy.isnan(pred.results[0]).all():
                try:
                    results = self.predict(pred.predictor, self.data)
                except (ValueError, DomainTransformationError) as err:
                    results = "{}: {}".format(pred.predictor.name, err)
                self.predictors[inputid] = pred._replace(results=results)

    def _set_errors(self):
        errors = "\n".join(p.results for p in self.predictors.values()
                           if isinstance(p.results, str))
        if errors:
            self.Error.predictor_failed(errors)
        else:
            self.Error.predictor_failed.clear()

    def _update_info(self):
        info = []
        if self.data is not None:
            info.append("Data: {} instances.".format(len(self.data)))
        else:
            info.append("Data: N/A")

        n_predictors = len(self.predictors)
        n_valid = len(self._valid_predictors())
        if n_valid != n_predictors:
            info.append("Predictors: {} (+ {} failed)".format(
                n_valid, n_predictors - n_valid))
        else:
            info.append("Predictors: {}".format(n_predictors or "N/A"))

        if self.class_var is None:
            info.append("Task: N/A")
        elif self.class_var.is_discrete:
            info.append("Task: Classification")
            self.checkbox_class.setEnabled(True)
            self.checkbox_prob.setEnabled(True)
        else:
            info.append("Task: Regression")
            self.checkbox_class.setEnabled(False)
            self.checkbox_prob.setEnabled(False)

        self.infolabel.setText("\n".join(info))

    def _invalidate_predictions(self):
        for inputid, pred in list(self.predictors.items()):
            self.predictors[inputid] = pred._replace(results=None)

    def _valid_predictors(self):
        if self.class_var is not None and self.data is not None:
            return [p for p in self.predictors.values()
                    if p.results is not None and not isinstance(p.results, str)]
        else:
            return []

    def _update_predictions_model(self):
        """Update the prediction view model."""
        if self.data is not None and self.class_var is not None:
            slots = self._valid_predictors()
            results = []
            class_var = self.class_var
            for p in slots:
                values, prob = p.results
                if self.class_var.is_discrete:
                    # if values were added to class_var between building the
                    # model and predicting, add zeros for new class values,
                    # which are always at the end
                    prob = numpy.c_[
                        prob,
                        numpy.zeros((prob.shape[0], len(class_var.values) - prob.shape[1]))]
                    values = [Value(class_var, v) for v in values]
                results.append((values, prob))
            results = list(zip(*(zip(*res) for res in results)))
            headers = [p.name for p in slots]
            model = PredictionsModel(results, headers)
        else:
            model = None

        predmodel = PredictionsSortProxyModel()
        predmodel.setSourceModel(model)
        predmodel.setDynamicSortFilter(True)
        self.predictionsview.setItemDelegate(PredictionsItemDelegate())
        self.predictionsview.setModel(predmodel)
        hheader = self.predictionsview.horizontalHeader()
        hheader.setSortIndicatorShown(False)
        # SortFilterProxyModel is slow due to large abstraction overhead
        # (every comparison triggers multiple `model.index(...)`,
        # model.rowCount(...), `model.parent`, ... calls)
        hheader.setSectionsClickable(predmodel.rowCount() < 20000)

        predmodel.layoutChanged.connect(self._update_data_sort_order)
        self._update_data_sort_order()
        self.predictionsview.resizeColumnsToContents()

    def _update_column_visibility(self):
        """Update data column visibility."""
        if self.data is not None and self.class_var is not None:
            domain = self.data.domain
            first_attr = len(domain.class_vars) + len(domain.metas)

            for i in range(first_attr, first_attr + len(domain.attributes)):
                self.dataview.setColumnHidden(i, not self.show_attrs)
            if domain.class_var:
                self.dataview.setColumnHidden(0, False)

    def _update_data_sort_order(self):
        """Update data row order to match the current predictions view order"""
        datamodel = self.dataview.model()  # data model proxy
        predmodel = self.predictionsview.model()  # predictions model proxy
        sortindicatorshown = False
        if datamodel is not None:
            assert isinstance(datamodel, TableSortProxyModel)
            n = datamodel.rowCount()
            if predmodel is not None and predmodel.sortColumn() >= 0:
                sortind = numpy.argsort(
                    [predmodel.mapToSource(predmodel.index(i, 0)).row()
                     for i in range(n)])
                sortind = numpy.array(sortind, numpy.int)
                sortindicatorshown = True
            else:
                sortind = None

            datamodel.setSortIndices(sortind)

        self.predictionsview.horizontalHeader() \
            .setSortIndicatorShown(sortindicatorshown)

    def _reset_order(self):
        """Reset the row sorting to original input order."""
        datamodel = self.dataview.model()
        predmodel = self.predictionsview.model()
        if datamodel is not None:
            datamodel.sort(-1)
        if predmodel is not None:
            predmodel.sort(-1)
        self.predictionsview.horizontalHeader().setSortIndicatorShown(False)

    def _update_prediction_delegate(self):
        """Update the predicted probability visibility state"""
        if self.class_var is not None:
            delegate = PredictionsItemDelegate()
            if self.class_var.is_continuous:
                self._setup_delegate_continuous(delegate)
            else:
                self._setup_delegate_discrete(delegate)
                proxy = self.predictionsview.model()
                if proxy is not None:
                    proxy.setProbInd(
                        numpy.array(self.selected_classes, dtype=int))
            self.predictionsview.setItemDelegate(delegate)
            self.predictionsview.resizeColumnsToContents()
        self._update_spliter()

    def _setup_delegate_discrete(self, delegate):
        colors = [QColor(*rgb) for rgb in self.class_var.colors]
        fmt = []
        if self.show_probabilities:
            fmt.append(" : ".join("{{dist[{}]:.2f}}".format(i)
                                  for i in sorted(self.selected_classes)))
        if self.show_predictions:
            fmt.append("{value!s}")
        delegate.setFormat(" \N{RIGHTWARDS ARROW} ".join(fmt))
        if self.draw_dist and colors is not None:
            delegate.setColors(colors)
        return delegate

    def _setup_delegate_continuous(self, delegate):
        delegate.setFormat("{{value:{}}}".format(self.class_var.format_str[1:]))

    def _update_spliter(self):
        if self.data is None:
            return

        def width(view):
            h_header = view.horizontalHeader()
            v_header = view.verticalHeader()
            return h_header.length() + v_header.width()

        w = width(self.predictionsview) + 4
        w1, w2 = self.splitter.sizes()
        self.splitter.setSizes([w, w1 + w2 - w])

    def commit(self):
        self._commit_predictions()
        self._commit_evaluation_results()

    def _commit_evaluation_results(self):
        slots = self._valid_predictors()
        if not slots or self.data.domain.class_var is None:
            self.Outputs.evaluation_results.send(None)
            return

        class_var = self.class_var
        nanmask = numpy.isnan(self.data.get_column_view(class_var)[0])
        data = self.data[~nanmask]
        N = len(data)
        results = Orange.evaluation.Results(data, store_data=True)
        results.folds = None
        results.row_indices = numpy.arange(N)
        results.actual = data.Y.ravel()
        results.predicted = numpy.vstack(
            tuple(p.results[0][~nanmask] for p in slots))
        if class_var and class_var.is_discrete:
            results.probabilities = numpy.array(
                [p.results[1][~nanmask] for p in slots])
        results.learner_names = [p.name for p in slots]
        self.Outputs.evaluation_results.send(results)

    def _commit_predictions(self):
        slots = self._valid_predictors()
        if not slots:
            self.Outputs.predictions.send(None)
            return

        if self.class_var and self.class_var.is_discrete:
            newmetas, newcolumns = self._classification_output_columns()
        else:
            newmetas, newcolumns = self._regression_output_columns()

        attrs = list(self.data.domain.attributes) if self.output_attrs else []
        metas = list(self.data.domain.metas) + newmetas
        domain = \
            Orange.data.Domain(attrs, self.data.domain.class_var, metas=metas)
        predictions = self.data.transform(domain)
        if newcolumns:
            newcolumns = numpy.hstack(
                [numpy.atleast_2d(cols) for cols in newcolumns])
            predictions.metas[:, -newcolumns.shape[1]:] = newcolumns
        self.Outputs.predictions.send(predictions)

    def _classification_output_columns(self):
        newmetas = []
        newcolumns = []
        slots = self._valid_predictors()
        if self.output_predictions:
            newmetas += [DiscreteVariable(name=p.name, values=self.class_values)
                         for p in slots]
            newcolumns += [p.results[0].reshape((-1, 1)) for p in slots]

        if self.output_probabilities:
            newmetas += [ContinuousVariable(name="%s (%s)" % (p.name, value))
                         for p in slots for value in self.class_values]
            newcolumns += [p.results[1] for p in slots]
        return newmetas, newcolumns

    def _regression_output_columns(self):
        slots = self._valid_predictors()
        newmetas = [ContinuousVariable(name=p.name) for p in slots]
        newcolumns = [p.results[0].reshape((-1, 1)) for p in slots]
        return newmetas, newcolumns

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

        if self.data is not None and self.class_var is not None:
            text = self.infolabel.text().replace('\n', '<br>')
            if self.show_probabilities and self.selected_classes:
                text += '<br>Showing probabilities for: '
                text += ', '. join([self.class_values[i]
                                    for i in self.selected_classes])
            self.report_paragraph('Info', text)
            self.report_table("Data & Predictions", merge_data_with_predictions(),
                              header_rows=1, header_columns=1)

    @classmethod
    def predict(cls, predictor, data):
        class_var = predictor.domain.class_var
        if class_var:
            if class_var.is_discrete:
                return cls.predict_discrete(predictor, data)
            else:
                return cls.predict_continuous(predictor, data)
        return None

    @staticmethod
    def predict_discrete(predictor, data):
        return predictor(data, Model.ValueProbs)

    @staticmethod
    def predict_continuous(predictor, data):
        values = predictor(data, Model.Value)
        return values, [None] * len(data)


class PredictionsItemDelegate(QStyledItemDelegate):
    """
    A Item Delegate for custom formatting of predictions/probabilities
    """
    def __init__(self, parent=None, **kwargs):
        self.__fmt = "{value!s}"
        self.__colors = None
        super().__init__(parent, **kwargs)

    def setFormat(self, fmt):
        if fmt != self.__fmt:
            self.__fmt = fmt
            self.sizeHintChanged.emit(QModelIndex())

    def setColors(self, colortable):
        if colortable is not None:
            colortable = list(colortable)
            if not all(isinstance(c, QColor) for c in colortable):
                raise TypeError

        self.__colors = colortable

    def displayText(self, value, _locale):
        try:
            value, dist = value
        except ValueError:
            return ""
        else:
            fmt = self.__fmt
            if dist is not None:
                text = fmt.format(dist=dist, value=value)
            else:
                text = fmt.format(value=value)
            return text

    @Slot(QHelpEvent, QAbstractItemView, QStyleOptionViewItem, QModelIndex,
          result=bool)
    def helpEvent(self, event, view, option, index):
        # reimplemented
        # NOTE: This is a slot in Qt4, but is a virtual func in Qt5
        value = index.data(Qt.EditRole)
        if isinstance(value, tuple) and len(value) == 2:
            try:
                tooltip = tool_tip(value)
            except ValueError:
                return False
            QToolTip.showText(event.globalPos(), tooltip, view)
            return True
        else:
            return super().helpEvent(event, view, option, index)

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        dist = self.distribution(index)
        if dist is None:
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
        if dist is None or self.__colors is None:
            super().paint(painter, option, index)
            return
        if not numpy.isfinite(numpy.sum(dist)):
            super().paint(painter, option, index)
            return

        nvalues = len(dist)
        if len(self.__colors) < nvalues:
            colors = colorpalette.ColorPaletteGenerator(nvalues)
            colors = [colors[i] for i in range(nvalues)]
        else:
            colors = self.__colors

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
            QSize(rect.width(), distheight)
        )
        painter.setPen(QPen(Qt.lightGray, 0.3))
        drawDistBar(painter, distrect, dist, colors)
        painter.restore()
        if text:
            style.drawItemText(
                painter, textrect, option.displayAlignment, option.palette,
                option.state & QStyle.State_Enabled, text)


def drawDistBar(painter, rect, distribution, colortable):
    """
    Parameters
    ----------
    painter : QPainter
    rect : QRect
    distribution : numpy.ndarray
    colortable : List[QColor]
    """
    # assert numpy.isclose(numpy.sum(distribution), 1.0)
    # assert numpy.all(distribution >= 0)
    painter.save()
    painter.translate(rect.topLeft())
    for dvalue, color in zip(distribution, colortable):
        if dvalue and numpy.isfinite(dvalue):
            painter.setBrush(color)
            width = rect.width() * dvalue
            painter.drawRoundedRect(QRectF(0, 0, width, rect.height()), 1, 2)
            painter.translate(width, 0.0)
    painter.restore()


class PredictionsSortProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.__probInd = None

    def setProbInd(self, indices):
        self.__probInd = indices
        self.invalidate()

    def lessThan(self, left, right):
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


class _TableModel(QAbstractTableModel):
    def __init__(self, table=None, headers=None, parent=None):
        super().__init__(parent)
        self._table = [[]] if table is None else table
        if headers is None:
            headers = [None] * len(self._table)
        self._header = headers
        self.__columnCount = max([len(row) for row in self._table] or [0])

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        else:
            return len(self._table)

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        else:
            return self.__columnCount

    def _value(self, index):
        row, column = index.row(), index.column()
        return self._table[row][column]

    def data(self, index, role=Qt.DisplayRole):
        if role in (Qt.DisplayRole, Qt.EditRole):
            return self._value(index)
        else:
            return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return str(section + 1)
        elif orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return (self._header[section] if section < len(self._header)
                    else str(section))
        else:
            return None

PredictionsModel = _TableModel


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
                size = self.itemDelegate(index).sizeHint(options, index)
                width = max(size.width(), width)
                sample_count += 1

            if sample_count >= TableView.MaxSizeHintSamples:
                break

        return width + 1 if self.showGrid() else width


class TableSortProxyModel(QSortFilterProxyModel):
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

        if self.sortColumn() < 0 and self.__sortInd is not None:
            self.sort(0)  # need some valid sort column
        elif self.__sortInd is None:
            self.sort(-1)  # explicit sort reset
        else:
            self.invalidate()

    def sortIndices(self):
        return self.__sortInd

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


def tool_tip(value):
    value, dist = value
    if dist is not None:
        return "{!s} {!s}".format(value, dist)
    else:
        return str(value)


if __name__ == "__main__":  # pragma: no cover
    filename = "iris.tab"
    test_data = Orange.data.Table(filename)

    def pred_error(data, *args, **kwargs):
        raise ValueError

    pred_error.domain = test_data.domain
    pred_error.name = "To err is human"

    if test_data.domain.has_discrete_class:
        predictors = [
            Orange.classification.SVMLearner(probability=True)(test_data),
            Orange.classification.LogisticRegressionLearner()(test_data),
            pred_error
        ]
    elif test_data.domain.has_continuous_class:
        predictors = [
            Orange.regression.RidgeRegressionLearner(alpha=1.0)(test_data),
            Orange.regression.LinearRegressionLearner()(test_data),
            pred_error
        ]
    else:
        predictors = [pred_error]

    WidgetPreview(OWPredictions).run(
        set_data=test_data,
        set_predictor=[(pred, i) for i, pred in enumerate(predictors)])
