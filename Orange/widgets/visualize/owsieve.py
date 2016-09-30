from itertools import chain

import numpy as np
from scipy.stats.distributions import chi2

from PyQt4.QtCore import Qt, QSize
from PyQt4.QtGui import (
    QGraphicsScene, QColor, QPen, QBrush, QSizePolicy, QGraphicsLineItem)

from Orange.data import Table, filter
from Orange.data.sql.table import SqlTable, LARGE_TABLE, DEFAULT_SAMPLE_TIME
from Orange.preprocess import Discretize
from Orange.preprocess.discretize import EqualFreq
from Orange.statistics.contingency import get_contingency
from Orange.widgets import gui
from Orange.widgets.settings import DomainContextHandler, ContextSetting
from Orange.widgets.utils import to_html as to_html
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.visualize.utils import (
    CanvasText, CanvasRectangle, ViewWithPress, VizRankDialogAttrPair)
from Orange.widgets.widget import OWWidget, Default, AttributeList


class ChiSqStats:
    """
    Compute and store statistics needed to show a plot for the given
    pair of attributes. The class is also used for ranking.
    """
    def __init__(self, data, attr1, attr2):
        self.observed = get_contingency(data, attr1, attr2)
        self.n = np.sum(self.observed)
        self.probs_x = self.observed.sum(axis=0) / self.n
        self.probs_y = self.observed.sum(axis=1) / self.n
        self.expected = np.outer(self.probs_y, self.probs_x) * self.n
        self.residuals = \
            (self.observed - self.expected) / np.sqrt(self.expected)
        self.chisqs = self.residuals ** 2
        self.chisq = float(np.sum(self.chisqs))
        self.p = chi2.sf(
            self.chisq, (len(self.probs_x) - 1) * (len(self.probs_y) - 1))


class SieveRank(VizRankDialogAttrPair):
    captionTitle = "Sieve Rank"

    def initialize(self):
        super().initialize()
        self.attrs = self.master.attrs

    def compute_score(self, state):
        return ChiSqStats(self.master.discrete_data, *state).p


class OWSieveDiagram(OWWidget):
    name = "Sieve Diagram"
    description = "Visualize the observed and expected frequencies " \
                  "for a combination of values."
    icon = "icons/SieveDiagram.svg"
    priority = 200

    inputs = [("Data", Table, "set_data", Default),
              ("Features", AttributeList, "set_input_features")]
    outputs = [("Selection", Table)]

    graph_name = "canvas"

    want_control_area = False

    settingsHandler = DomainContextHandler()
    attrX = ContextSetting("", exclude_metas=False)
    attrY = ContextSetting("", exclude_metas=False)
    selection = ContextSetting(set())

    def __init__(self):
        # pylint: disable=missing-docstring
        super().__init__()

        self.data = self.discrete_data = None
        self.attrs = []
        self.input_features = None
        self.areas = []
        self.selection = set()

        self.attr_box = gui.hBox(self.mainArea)
        model = VariableListModel()
        model.wrap(self.attrs)
        combo_args = dict(
            widget=self.attr_box, master=self, contentsLength=12,
            callback=self.update_attr, sendSelectedValue=True, valueType=str,
            model=model)
        fixed_size = (QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.attrXCombo = gui.comboBox(value="attrX", **combo_args)
        gui.widgetLabel(self.attr_box, "\u2715", sizePolicy=fixed_size)
        self.attrYCombo = gui.comboBox(value="attrY", **combo_args)
        self.vizrank, self.vizrank_button = SieveRank.add_vizrank(
            self.attr_box, self, "Score Combinations", self.set_attr)
        self.vizrank_button.setSizePolicy(*fixed_size)

        self.canvas = QGraphicsScene()
        self.canvasView = ViewWithPress(
            self.canvas, self.mainArea, handler=self.reset_selection)
        self.mainArea.layout().addWidget(self.canvasView)
        self.canvasView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.canvasView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        box = gui.hBox(self.mainArea)
        box.layout().addWidget(self.graphButton)
        box.layout().addWidget(self.report_button)

    def sizeHint(self):
        return QSize(450, 550)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_graph()

    def showEvent(self, event):
        super().showEvent(event)
        self.update_graph()

    def set_data(self, data):
        """
        Discretize continuous attributes, and put all attributes and discrete
        metas into self.attrs, which is used as a model for combos.

        Select the first two attributes unless context overrides this.
        Method `resolve_shown_attributes` is called to use the attributes from
        the input, if it exists and matches the attributes in the data.

        Remove selection; again let the context override this.
        Initialize the vizrank dialog, but don't show it.

        Args:
            data (Table): input data
        """
        if isinstance(data, SqlTable) and data.approx_len() > LARGE_TABLE:
            data = data.sample_time(DEFAULT_SAMPLE_TIME)

        self.closeContext()
        self.data = data
        self.areas = []
        self.selection = set()
        if self.data is None:
            self.attrs[:] = []
        else:
            if any(attr.is_continuous for attr in data.domain):
                discretizer = Discretize(
                    method=EqualFreq(n=4),
                    discretize_classes=True, discretize_metas=True)
                self.discrete_data = discretizer(data)
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
            self.areas = []
            self.selection = set()
        self.openContext(self.data)
        self.resolve_shown_attributes()
        self.update_graph()
        self.update_selection()

        self.vizrank.initialize()
        self.vizrank_button.setEnabled(
            self.data is not None and len(self.data) > 1 and
            len(self.data.domain.attributes) > 1)

    def set_attr(self, attr_x, attr_y):
        self.attrX, self.attrY = attr_x.name, attr_y.name
        self.update_attr()

    def update_attr(self):
        """Update the graph and selection."""
        self.selection = set()
        self.update_graph()
        self.update_selection()

    def set_input_features(self, attr_list):
        """
        Handler for the Features signal.

        The method stores the attributes and calls `resolve_shown_attributes`

        Args:
            attr_list (AttributeList): data from the signal
        """
        self.input_features = attr_list
        self.resolve_shown_attributes()
        self.update_selection()

    def resolve_shown_attributes(self):
        """
        Use the attributes from the input signal if the signal is present
        and at least two attributes appear in the domain. If there are
        multiple, use the first two. Combos are disabled if inputs are used.
        """
        self.warning()
        self.attr_box.setEnabled(True)
        if not self.input_features:  # None or empty
            return
        features = [f for f in self.input_features if f in self.attrs]
        if not features:
            self.warning(
                "Features from the input signal are not present in the data")
            return
        old_attrs = self.attrX, self.attrY
        self.attrX, self.attrY = [f.name for f in (features * 2)[:2]]
        self.attr_box.setEnabled(False)
        if (self.attrX, self.attrY) != old_attrs:
            self.selection = set()
            self.update_graph()

    def reset_selection(self):
        self.selection = set()
        self.update_selection()

    def select_area(self, area, event):
        """
        Add or remove the clicked area from the selection

        Args:
            area (QRect): the area that is clicked
            event (QEvent): event description
        """
        if event.button() != Qt.LeftButton:
            return
        index = self.areas.index(area)
        if event.modifiers() & Qt.ControlModifier:
            self.selection ^= {index}
        else:
            self.selection = {index}
        self.update_selection()

    def update_selection(self):
        """
        Update the graph (pen width) to show the current selection.
        Filter and output the data.
        """
        if self.areas is None or not self.selection:
            self.send("Selection", None)
            return

        filts = []
        for i, area in enumerate(self.areas):
            if i in self.selection:
                width = 4
                val_x, val_y = area.value_pair
                filts.append(
                    filter.Values([
                        filter.FilterDiscrete(self.attrX, [val_x]),
                        filter.FilterDiscrete(self.attrY, [val_y])
                    ]))
            else:
                width = 1
            pen = area.pen()
            pen.setWidth(width)
            area.setPen(pen)
        if len(filts) == 1:
            filts = filts[0]
        else:
            filts = filter.Values(filts, conjunction=False)
        selection = filts(self.discrete_data)
        if self.discrete_data is not self.data:
            idset = set(selection.ids)
            sel_idx = [i for i, id in enumerate(self.data.ids) if id in idset]
            selection = self.data[sel_idx]
        self.send("Selection", selection)

    def update_graph(self):
        # Function uses weird names like r, g, b, but it does it with utmost
        # caution, hence
        # pylint: disable=invalid-name
        """Update the graph."""

        def text(txt, *args, **kwargs):
            return CanvasText(self.canvas, "", html_text=to_html(txt),
                              *args, **kwargs)

        def width(txt):
            return text(txt, 0, 0, show=False).boundingRect().width()

        def fmt(val):
            return str(int(val)) if val % 1 == 0 else "{:.2f}".format(val)

        def show_pearson(rect, pearson, pen_width):
            """
            Color the given rectangle according to its corresponding
            standardized Pearson residual.

            Args:
                rect (QRect): the rectangle being drawn
                pearson (float): signed standardized pearson residual
                pen_width (int): pen width (bolder pen is used for selection)
            """
            r = rect.rect()
            x, y, w, h = r.x(), r.y(), r.width(), r.height()
            if w == 0 or h == 0:
                return

            r = b = 255
            if pearson > 0:
                r = g = max(255 - 20 * pearson, 55)
            elif pearson < 0:
                b = g = max(255 + 20 * pearson, 55)
            else:
                r = g = b = 224
            rect.setBrush(QBrush(QColor(r, g, b)))
            pen_color = QColor(255 * (r == 255), 255 * (g == 255),
                               255 * (b == 255))
            pen = QPen(pen_color, pen_width)
            rect.setPen(pen)
            if pearson > 0:
                pearson = min(pearson, 10)
                dist = 20 - 1.6 * pearson
            else:
                pearson = max(pearson, -10)
                dist = 20 - 8 * pearson
            pen.setWidth(1)

            def _offseted_line(ax, ay):
                r = QGraphicsLineItem(x + ax, y + ay, x + (ax or w),
                                      y + (ay or h))
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

        def make_tooltip():
            """Create the tooltip. The function uses local variables from
            the enclosing scope."""
            # pylint: disable=undefined-loop-variable
            def _oper(attr_name, txt):
                if self.data.domain[attr_name] is ddomain[attr_name]:
                    return "="
                return " " if txt[0] in "<≥" else " in "

            return (
                "<b>{attrX}{xeq}{xval_name}</b>: {obs_x}/{n} ({p_x:.0f} %)".
                format(attrX=to_html(attr_x),
                       xeq=_oper(attr_x, xval_name),
                       xval_name=to_html(xval_name),
                       obs_x=fmt(chi.probs_x[x] * n),
                       n=int(n),
                       p_x=100 * chi.probs_x[x]) +
                "<br/>" +
                "<b>{attrY}{yeq}{yval_name}</b>: {obs_y}/{n} ({p_y:.0f} %)".
                format(attrY=to_html(attr_y),
                       yeq=_oper(attr_y, yval_name),
                       yval_name=to_html(yval_name),
                       obs_y=fmt(chi.probs_y[y] * n),
                       n=int(n),
                       p_y=100 * chi.probs_y[y]) +
                "<hr/>" +
                """<b>combination of values: </b><br/>
                   &nbsp;&nbsp;&nbsp;expected {exp} ({p_exp:.0f} %)<br/>
                   &nbsp;&nbsp;&nbsp;observed {obs} ({p_obs:.0f} %)""".
                format(exp=fmt(chi.expected[y, x]),
                       p_exp=100 * chi.expected[y, x] / n,
                       obs=fmt(chi.observed[y, x]),
                       p_obs=100 * chi.observed[y, x] / n))

        for item in self.canvas.items():
            self.canvas.removeItem(item)
        if self.data is None or len(self.data) == 0 or \
                self.attrX is None or self.attrY is None:
            return

        ddomain = self.discrete_data.domain
        attr_x, attr_y = self.attrX, self.attrY
        disc_x, disc_y = ddomain[attr_x], ddomain[attr_y]
        view = self.canvasView

        chi = ChiSqStats(self.discrete_data, attr_x, attr_y)
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
                show_pearson(rect, chi.residuals[y, x], 3 * selected)
                rect.setToolTip(make_tooltip())

                if x == 0:
                    text(yval_name, x_off, curr_y + height / 2,
                         Qt.AlignRight | Qt.AlignVCenter)
                curr_y += height

            xl = text(xval_name, curr_x + width / 2, y_off + square_size,
                      Qt.AlignHCenter | Qt.AlignTop)
            max_xlabel_h = max(int(xl.boundingRect().height()), max_xlabel_h)
            curr_x += width

        bottom = y_off + square_size + max_xlabel_h
        text(attr_y, 0, y_off + square_size / 2,
             Qt.AlignLeft | Qt.AlignVCenter, bold=True, vertical=True)
        text(attr_x, x_off + square_size / 2, bottom,
             Qt.AlignHCenter | Qt.AlignTop, bold=True)
        xl = text("χ²={:.2f}, p={:.3f}".format(chi.chisq, chi.p),
                  0, bottom)
        # Assume similar height for both lines
        text("N = " + fmt(chi.n), 0, bottom - xl.boundingRect().height())

    def get_widget_name_extension(self):
        if self.data is not None:
            return "{} vs {}".format(self.attrX, self.attrY)

    def send_report(self):
        self.report_plot()


def main():
    # pylint: disable=missing-docstring
    import sys
    from PyQt4.QtGui import QApplication
    a = QApplication(sys.argv)
    ow = OWSieveDiagram()
    ow.show()
    data = Table(r"zoo.tab")
    ow.set_data(data)
    a.exec_()
    ow.saveSettings()

if __name__ == "__main__":
    main()
