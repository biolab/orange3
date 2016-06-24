from itertools import chain
from bisect import bisect_left

import numpy as np
from scipy import stats

from PyQt4.QtCore import Qt, QSize
from PyQt4.QtGui import (QGraphicsScene, QColor, QPen, QBrush, QTableView,
                         QStandardItemModel, QStandardItem,
                         QDialog, QApplication, QSizePolicy, QGraphicsLineItem)

from Orange.data import Table, filter
from Orange.data.sql.table import SqlTable, LARGE_TABLE, DEFAULT_SAMPLE_TIME
from Orange.preprocess import Discretize
from Orange.preprocess.discretize import EqualFreq
from Orange.statistics.contingency import get_contingency
from Orange.widgets import gui
from Orange.widgets.settings import DomainContextHandler, ContextSetting
from Orange.widgets.utils import getHtmlCompatibleString as to_html
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.visualize.owmosaic import (
    CanvasText, CanvasRectangle, ViewWithPress)
from Orange.widgets.widget import OWWidget, Default, AttributeList


class OWSieveDiagram(OWWidget):
    name = "Sieve Diagram"
    description = "A two-way contingency table providing information on the " \
                  "relation between the observed and expected frequencies " \
                  "of a combination of feature values under the assumption of independence."
    icon = "icons/SieveDiagram.svg"
    priority = 4200

    inputs = [("Data", Table, "set_data", Default),
              ("Features", AttributeList, "set_input_features")]
    outputs = [("Selection", Table)]

    graph_name = "canvas"

    want_control_area = False

    settingsHandler = DomainContextHandler()
    attrX = ContextSetting("")
    attrY = ContextSetting("")
    selection = ContextSetting(set())

    def __init__(self):
        super().__init__()

        self.data = self.discrete_data = None
        self.areas = None
        self.input_features = None
        self.attrs = []

        self.attr_box = gui.hBox(self.mainArea)
        model = VariableListModel()
        model.wrap(self.attrs)
        self.attrXCombo = gui.comboBox(
            self.attr_box, self, value="attrX", contentsLength=12,
            callback=self.change_attr, sendSelectedValue=True, valueType=str)
        self.attrXCombo.setModel(model)
        gui.widgetLabel(self.attr_box, "\u2715").\
            setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.attrYCombo = gui.comboBox(
            self.attr_box, self, value="attrY", contentsLength=12,
            callback=self.change_attr, sendSelectedValue=True, valueType=str)
        self.attrYCombo.setModel(model)
        self.vizrank = self.VizRank(self)
        self.vizrank_button = gui.button(
            self.attr_box, self, "Score Combinations",
            callback=self.vizrank.reshow,
            tooltip="Find projections with good class separation",
            sizePolicy=QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self.vizrank_button.setEnabled(False)

        self.canvas = QGraphicsScene()
        self.canvasView = ViewWithPress(self.canvas, self.mainArea,
                                        handler=self.reset_selection)
        self.mainArea.layout().addWidget(self.canvasView)
        self.canvasView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.canvasView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        box = gui.hBox(self.mainArea)
        box.layout().addWidget(self.graphButton)
        box.layout().addWidget(self.report_button)

    def sizeHint(self):
        return QSize(450, 550)

    def set_data(self, data):
        if type(data) == SqlTable and data.approx_len() > LARGE_TABLE:
            data = data.sample_time(DEFAULT_SAMPLE_TIME)

        self.closeContext()
        self.data = data
        self.areas = []
        if self.data is None:
            self.attrs[:] = []
        else:
            if any(attr.is_continuous for attr in data.domain):
                self.discrete_data = Discretize(method=EqualFreq(n=4))(data)
            else:
                self.discrete_data = self.data
            self.attrs[:] = [
                var for var in chain(
                    self.discrete_data.domain,
                    (var for var in self.data.domain.metas if var.is_discrete))
            ]
        if self.attrs:
            self.attrX = self.attrs[0].name
            self.attrY = self.attrs[len(self.attrs) > 1].name
        else:
            self.attrX = self.attrY = None
            self.areas = self.selection = None
        self.openContext(self.data)
        self.resolve_shown_attributes()
        self.update_selection()
        self.vizrank._initialize()
        self.vizrank_button.setEnabled(
            self.data is not None and
            len(self.data) > 1 and
            len(self.data.domain.attributes) > 1)

    def change_attr(self, attributes=None):
        if attributes is not None:
            self.attrX, self.attrY = attributes
        self.selection = set()
        self.updateGraph()
        self.update_selection()

    def set_input_features(self, attrList):
        self.input_features = attrList
        self.resolve_shown_attributes()
        self.update_selection()

    def resolve_shown_attributes(self):
        self.warning(1)
        self.attr_box.setEnabled(True)
        if self.input_features:  # non-None and non-empty!
            features = [f for f in self.input_features if f in self.attrs]
            if not features:
                self.warning(1, "Features from the input signal "
                                "are not present in the data")
            else:
                old_attrs = self.attrX, self.attrY
                self.attrX, self.attrY = [f.name for f in (features * 2)[:2]]
                self.attr_box.setEnabled(False)
                if (self.attrX, self.attrY) != old_attrs:
                    self.selection = set()
        # else: do nothing; keep current features, even if input with the
        # features just changed to None
        self.updateGraph()

    def resizeEvent(self, e):
        OWWidget.resizeEvent(self,e)
        self.updateGraph()

    def showEvent(self, ev):
        OWWidget.showEvent(self, ev)
        self.updateGraph()

    def reset_selection(self):
        self.selection = set()
        self.update_selection()

    def select_area(self, area, ev):
        if ev.button() != Qt.LeftButton:
            return
        index = self.areas.index(area)
        if ev.modifiers() & Qt.ControlModifier:
            self.selection ^= {index}
        else:
            self.selection = {index}
        self.update_selection()

    def update_selection(self):
        if self.areas is None or not self.selection:
            self.send("Selection", None)
            return

        filters = []
        for i, area in enumerate(self.areas):
            if i in self.selection:
                width = 4
                val_x, val_y = area.value_pair
                filters.append(
                    filter.Values([
                        filter.FilterDiscrete(self.attrX, [val_x]),
                        filter.FilterDiscrete(self.attrY, [val_y])
                    ]))
            else:
                width = 1
            pen = area.pen()
            pen.setWidth(width)
            area.setPen(pen)
        if len(filters) == 1:
            filters = filters[0]
        else:
            filters = filter.Values(filters, conjunction=False)
        selection = filters(self.discrete_data)
        if self.discrete_data is not self.data:
            idset = set(selection.ids)
            sel_idx = [i for i, id in enumerate(self.data.ids) if id in idset]
            selection = self.data[sel_idx]
        self.send("Selection", selection)

    class ChiSqStats:
        def __init__(self, data, attr1, attr2):
            self.observed = get_contingency(data, attr1, attr2)
            self.n = np.sum(self.observed)
            self.probs_x = self.observed.sum(axis=0) / self.n
            self.probs_y = self.observed.sum(axis=1) / self.n
            self.expected = np.outer(self.probs_y, self.probs_x) * self.n
            self.residuals = \
                (self.observed - self.expected) / np.sqrt(self.expected)
            self.chisqs = self.residuals ** 2
            self.chisq = np.sum(self.chisqs)
            self.p = stats.distributions.chi2.sf(
                self.chisq, (len(self.probs_x) - 1) * (len(self.probs_y) - 1))

    def updateGraph(self, *args):
        def text(txt, *args, **kwargs):
            return CanvasText(self.canvas, "", html_text=to_html(txt),
                              *args, **kwargs)

        def width(txt):
            return text(txt, 0, 0, show=False).boundingRect().width()

        for item in self.canvas.items():
            self.canvas.removeItem(item)
        if self.data is None or len(self.data) == 0 or \
                self.attrX is None or self.attrY is None:
            return

        ddomain = self.discrete_data.domain
        attr_x, attr_y = self.attrX, self.attrY
        disc_x, disc_y = ddomain[attr_x], ddomain[attr_y]
        view = self.canvasView

        chi = self.ChiSqStats(self.discrete_data, attr_x, attr_y)
        n = chi.n
        max_ylabel_w = max((width(val) for val in disc_y.values), default=0)
        max_ylabel_w = min(max_ylabel_w, 200)
        x_off = width(attr_x) + max_ylabel_w
        y_off = 15
        square_size = min(view.width() - x_off - 35, view.height() - y_off - 50)
        square_size = max(square_size, 10)
        self.canvasView.setSceneRect(0, 0, view.width(), view.height())

        curr_x = x_off
        max_xlabel_h = 0
        self.areas = []
        for x, (px, xval_name) in enumerate(zip(chi.probs_x, disc_x.values)):
            if px == 0:
                continue
            width = square_size * px
            curr_y = y_off
            for y in range(len(chi.probs_y) - 1, -1, -1):  # bottom-up order
                py = chi.probs_y[y]
                yval_name = disc_y.values[y]
                if py == 0:
                    continue
                height = square_size * py

                selected = len(self.areas) in self.selection
                rect = CanvasRectangle(
                    self.canvas, curr_x + 2, curr_y + 2, width - 4, height - 4,
                    z=-10, onclick=self.select_area)
                rect.value_pair = x, y
                self.areas.append(rect)
                self.show_pearson(rect, chi.residuals[y, x], 3 * selected)

                def _addeq(attr_name, txt):
                    if self.data.domain[attr_name] is ddomain[attr_name]:
                        return "="
                    return " " if txt[0] in "<≥" else " in "

                tooltip_text = """
                    <b>{attrX}{xeq}{xval_name}</b>: {obs_x}/{n} ({prob_x:.0f} %)
                    <br/>
                    <b>{attrY}{yeq}{yval_name}</b>: {obs_y}/{n} ({prob_y:.0f} %)
                    <hr/>
                    <b>combination of values: </b><br/>
                       &nbsp;&nbsp;&nbsp;expected {exp:.2f} ({p_exp:.0f} %)<br/>
                       &nbsp;&nbsp;&nbsp;observed {obs:.2f} ({p_obs:.0f} %)
                    """.format(
                    n=int(n),
                    attrX=to_html(attr_x),
                    xeq=_addeq(attr_x, xval_name),
                    xval_name=to_html(xval_name),
                    obs_x=int(chi.probs_x[x] * n),
                    prob_x=100 * chi.probs_x[x],
                    attrY=to_html(attr_y),
                    yeq=_addeq(attr_y, yval_name),
                    yval_name=to_html(yval_name),
                    obs_y=int(chi.probs_y[y] * n),
                    prob_y=100 * chi.probs_y[y],
                    exp=chi.expected[y, x],
                    p_exp=100 * chi.expected[y, x] / n,
                    obs=int(chi.observed[y, x]),
                    p_obs=100 * chi.observed[y, x] / n)
                rect.setToolTip(tooltip_text)

                if not x:
                    text(yval_name, x_off, curr_y + height / 2,
                         Qt.AlignRight | Qt.AlignVCenter)
                curr_y += height

            xl = text(xval_name, curr_x + width / 2, y_off + square_size,
                      Qt.AlignHCenter | Qt.AlignTop)
            max_xlabel_h = max(int(xl.boundingRect().height()), max_xlabel_h)
            curr_x += width

        text(attr_y, 0, y_off + square_size / 2, Qt.AlignLeft | Qt.AlignVCenter,
             bold=True, vertical=True)
        text(attr_x, x_off + square_size / 2,
             y_off + square_size + max_xlabel_h, Qt.AlignHCenter | Qt.AlignTop,
             bold=True)

    def show_pearson(self, rect, pearson, pen_width):
        r = rect.rect()
        x, y, w, h = r.x(), r.y(), r.width(), r.height()
        if w == 0 or h == 0:
            return

        r = g = b = 255
        if pearson > 0:
            r = g = max(255 - 20 * pearson, 55)
        elif pearson < 0:
            b = g = max(255 + 20 * pearson, 55)
        rect.setBrush(QBrush(QColor(r, g, b)))
        pen = QPen(QColor(255 * (r == 255), 255 * (g == 255), 255 * (b == 255)),
                   pen_width)
        rect.setPen(pen)
        if pearson > 0:
            pearson = min(pearson, 10)
            dist = 20 - 1.6 * pearson
        else:
            pearson = max(pearson, -10)
            dist = 20 - 8 * pearson
        pen.setWidth(1)

        def _offseted_line(ax, ay):
            r = QGraphicsLineItem(x + ax, y + ay, x + (ax or w), y + (ay or h))
            self.canvas.addItem(r)
            r.setPen(pen)

        ax = dist
        while ax < w:
            _offseted_line(ax, 0)
            ax += dist

        ay = dist
        while ay < h:
            _offseted_line(0, ay)
            ay += dist

    def closeEvent(self, ce):
        QDialog.closeEvent(self, ce)

    def get_widget_name_extension(self):
        if self.data is not None:
            return "{} vs {}".format(self.attrX, self.attrY)

    def send_report(self):
        self.report_plot()

    class VizRank(OWWidget):
        name = "Rank projections (Sieve)"
        want_control_area = False

        def __init__(self, parent_widget):
            super().__init__()
            self.parent_widget = parent_widget
            self.running = False
            self.progress = None
            self.i = self.j = 0

            self.projectionTable = QTableView()
            self.mainArea.layout().addWidget(self.projectionTable)
            self.projectionTable.setSelectionBehavior(QTableView.SelectRows)
            self.projectionTable.setSelectionMode(QTableView.SingleSelection)
            self.projectionTable.setSortingEnabled(True)
            self.projectionTableModel = QStandardItemModel(self)
            self.projectionTable.setModel(self.projectionTableModel)
            self.projectionTable.selectionModel().selectionChanged.connect(
                self.on_selection_changed)
            self.projectionTable.horizontalHeader().hide()

            self.button = gui.button(self.mainArea, self, "Start evaluation",
                                     callback=self.toggle, default=True)
            self.resize(320, 512)
            self._initialize()

        def _initialize(self):
            self.running = False
            self.projectionTableModel.clear()
            self.projectionTable.setColumnWidth(0, 120)
            self.projectionTable.setColumnWidth(1, 120)
            self.button.setText("Start evaluation")
            self.button.setEnabled(False)
            self.pause = False
            self.scores = []
            self.i = self.j = 0
            if self.progress:
                self.progress.finish()
            self.progress = None

            self.information(0)
            if self.parent_widget.data:
                if not self.parent_widget.data.domain.class_var:
                    self.information(
                        0, "Data with a class variable is required.")
                    return
                if len(self.parent_widget.data.domain.attributes) < 2:
                    self.information(
                        0, 'At least 2 features are needed.')
                    return
                if len(self.parent_widget.data) < 2:
                    self.information(
                        0, 'At least 2 instances are needed.')
                    return
                self.button.setEnabled(True)

        def on_selection_changed(self, selected, deselected):
            """Called when the ranks view selection changes."""
            a1 = selected.indexes()[0].data()
            a2 = selected.indexes()[1].data()
            self.parent_widget.change_attr(attributes=(a1, a2))

        def toggle(self):
            self.running ^= 1
            if self.running:
                self.button.setText("Pause")
                self.run()
            else:
                self.button.setText("Continue")
                self.button.setEnabled(False)

        def stop(self, i, j):
            self.i, self.j = i, j
            if not self.projectionTable.selectedIndexes():
                self.projectionTable.selectRow(0)
            self.button.setEnabled(True)

        def run(self):
            widget = self.parent_widget
            attrs = widget.attrs
            if not self.progress:
                self.progress = gui.ProgressBar(self, len(attrs))
            for i in range(self.i, len(attrs)):
                for j in range(self.j, i):
                    if not self.running:
                        self.stop(i, j)
                        return
                    score = widget.ChiSqStats(widget.discrete_data, i, j).p
                    pos = bisect_left(self.scores, score)
                    self.projectionTableModel.insertRow(
                        len(self.scores) - pos,
                        [QStandardItem(widget.attrs[i].name),
                         QStandardItem(widget.attrs[j].name)])
                    self.scores.insert(pos, score)
                self.progress.advance()
            self.progress.finish()
            if not self.projectionTable.selectedIndexes():
                self.projectionTable.selectRow(0)
            self.button.setText("Finished")
            self.button.setEnabled(False)


# test widget appearance
if __name__ == "__main__":
    import sys
    a = QApplication(sys.argv)
    ow = OWSieveDiagram()
    ow.show()
    data = Table(r"zoo.tab")
    ow.set_data(data)
    a.exec_()
    ow.saveSettings()
