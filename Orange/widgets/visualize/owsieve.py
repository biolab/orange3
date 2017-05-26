import math
from itertools import chain

import numpy as np
from scipy.stats.distributions import chi2

from AnyQt.QtCore import Qt, QSize
from AnyQt.QtGui import QColor, QPen, QBrush
from AnyQt.QtWidgets import QGraphicsScene, QGraphicsLineItem, QSizePolicy

from Orange.data import Table, filter, Variable
from Orange.data.sql.table import SqlTable, LARGE_TABLE, DEFAULT_SAMPLE_TIME
from Orange.preprocess import Discretize
from Orange.preprocess.discretize import EqualFreq
from Orange.statistics.contingency import get_contingency
from Orange.widgets import gui, settings
from Orange.widgets.settings import DomainContextHandler, ContextSetting
from Orange.widgets.utils import to_html as to_html
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.visualize.utils import (
    CanvasText, CanvasRectangle, ViewWithPress, VizRankDialogAttrPair)
from Orange.widgets.widget import OWWidget, AttributeList, Input, Output


class ChiSqStats:
    """
    Compute and store statistics needed to show a plot for the given
    pair of attributes. The class is also used for ranking.
    """
    def __init__(self, data, attr1, attr2):
        attr1 = data.domain[attr1]
        attr2 = data.domain[attr2]
        if attr1.is_discrete and not attr1.values or \
                attr2.is_discrete and not attr2.values:
            self.p = np.nan
            return
        self.observed = get_contingency(data, attr1, attr2)
        self.n = np.sum(self.observed)
        self.probs_x = self.observed.sum(axis=0) / self.n
        self.probs_y = self.observed.sum(axis=1) / self.n
        self.expected = np.outer(self.probs_y, self.probs_x) * self.n
        self.residuals = \
            (self.observed - self.expected) / np.sqrt(self.expected)
        self.residuals = np.nan_to_num(self.residuals)
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
        p = ChiSqStats(self.master.discrete_data,
                       *(self.attrs[i].name for i in state)).p
        return 2 if np.isnan(p) else p

    def bar_length(self, score):
        return min(1, -math.log(score, 10) / 50) if 0 < score <= 1 else 0


class OWSieveDiagram(OWWidget):
    name = "Sieve Diagram"
    description = "Visualize the observed and expected frequencies " \
                  "for a combination of values."
    icon = "icons/SieveDiagram.svg"
    priority = 200

    class Inputs:
        data = Input("Data", Table, default=True)
        features = Input("Features", AttributeList)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    graph_name = "canvas"

    want_control_area = False

    settings_version = 1
    settingsHandler = DomainContextHandler()
    attr_x = ContextSetting(None, exclude_metas=False)
    attr_y = ContextSetting(None, exclude_metas=False)
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
        self.domain_model = DomainModel(valid_types=DomainModel.PRIMITIVE)
        combo_args = dict(
            widget=self.attr_box, master=self, contentsLength=12,
            callback=self.update_attr, sendSelectedValue=True, valueType=str,
            model=self.domain_model)
        fixed_size = (QSizePolicy.Fixed, QSizePolicy.Fixed)
        gui.comboBox(value="attr_x", **combo_args)
        gui.widgetLabel(self.attr_box, "\u2715", sizePolicy=fixed_size)
        gui.comboBox(value="attr_y", **combo_args)
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

    @classmethod
    def migrate_context(cls, context, version):
        if not version:
            settings.rename_setting(context, "attrX", "attr_x")
            settings.rename_setting(context, "attrY", "attr_y")
            settings.migrate_str_to_variable(context)

    @Inputs.data
    def set_data(self, data):
        """
        Discretize continuous attributes, and put all attributes and discrete
        metas into self.attrs.

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
            self.domain_model.set_domain(None)
        else:
            self.domain_model.set_domain(data.domain)
        self.attrs = [x for x in self.domain_model if isinstance(x, Variable)]
        if self.attrs:
            self.attr_x = self.attrs[0]
            self.attr_y = self.attrs[len(self.attrs) > 1]
        else:
            self.attr_x = self.attr_y = None
            self.areas = []
            self.selection = set()
        self.openContext(self.data)
        if self.data:
            self.discrete_data = self.sparse_to_dense(data, True)
        self.resolve_shown_attributes()
        self.update_graph()
        self.update_selection()

        self.vizrank.initialize()
        self.vizrank_button.setEnabled(
            self.data is not None and len(self.data) > 1 and
            len(self.data.domain.attributes) > 1 and not self.data.is_sparse())

    def set_attr(self, attr_x, attr_y):
        self.attr_x, self.attr_y = attr_x, attr_y
        self.update_attr()

    def update_attr(self):
        """Update the graph and selection."""
        self.selection = set()
        self.discrete_data = self.sparse_to_dense(self.data)
        self.update_graph()
        self.update_selection()

    def sparse_to_dense(self, data, init=False):
        """
        Extracts two selected columns from sparse matrix.
        GH-2260
        """
        def discretizer(data):
            if any(attr.is_continuous for attr in chain(data.domain, data.domain.metas)):
                discretize = Discretize(
                    method=EqualFreq(n=4), remove_const=False,
                    discretize_classes=True, discretize_metas=True)
                return discretize(data)
            return data

        if not data.is_sparse() and not init:
            return self.discrete_data
        if data.is_sparse():
            attrs = {self.attr_x,
                     self.attr_y}
            new_domain = data.domain.select_columns(attrs)
            data = Table.from_table(new_domain, data)
            data.X = data.X.toarray()
        return discretizer(data)

    @Inputs.features
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
        features = [f for f in self.input_features if f in self.domain_model]
        if not features:
            self.warning(
                "Features from the input signal are not present in the data")
            return
        old_attrs = self.attr_x, self.attr_y
        self.attr_x, self.attr_y = [f for f in (features * 2)[:2]]
        self.attr_box.setEnabled(False)
        if (self.attr_x, self.attr_y) != old_attrs:
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
            self.Outputs.selected_data.send(None)
            self.Outputs.annotated_data.send(create_annotated_table(self.data, []))
            return

        filts = []
        for i, area in enumerate(self.areas):
            if i in self.selection:
                width = 4
                val_x, val_y = area.value_pair
                filts.append(
                    filter.Values([
                        filter.FilterDiscrete(self.attr_x.name, [val_x]),
                        filter.FilterDiscrete(self.attr_y.name, [val_y])
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
        idset = set(selection.ids)
        sel_idx = [i for i, id in enumerate(self.data.ids) if id in idset]
        if self.discrete_data is not self.data:
            selection = self.data[sel_idx]
        self.Outputs.selected_data.send(selection)
        self.Outputs.annotated_data.send(create_annotated_table(self.data, sel_idx))

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
            def _oper(attr, txt):
                if self.data.domain[attr.name] is ddomain[attr.name]:
                    return "="
                return " " if txt[0] in "<≥" else " in "

            return (
                "<b>{attr_x}{xeq}{xval_name}</b>: {obs_x}/{n} ({p_x:.0f} %)".
                format(attr_x=to_html(attr_x.name),
                       xeq=_oper(attr_x, xval_name),
                       xval_name=to_html(xval_name),
                       obs_x=fmt(chi.probs_x[x] * n),
                       n=int(n),
                       p_x=100 * chi.probs_x[x]) +
                "<br/>" +
                "<b>{attr_y}{yeq}{yval_name}</b>: {obs_y}/{n} ({p_y:.0f} %)".
                format(attr_y=to_html(attr_y.name),
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
                self.attr_x is None or self.attr_y is None:
            return

        ddomain = self.discrete_data.domain
        attr_x, attr_y = self.attr_x, self.attr_y
        disc_x, disc_y = ddomain[attr_x.name], ddomain[attr_y.name]
        view = self.canvasView

        chi = ChiSqStats(self.discrete_data, disc_x, disc_y)
        max_ylabel_w = max((width(val) for val in disc_y.values), default=0)
        max_ylabel_w = min(max_ylabel_w, 200)
        x_off = width(attr_x.name) + max_ylabel_w
        y_off = 15
        square_size = min(view.width() - x_off - 35, view.height() - y_off - 80)
        square_size = max(square_size, 10)
        self.canvasView.setSceneRect(0, 0, view.width(), view.height())
        if not disc_x.values or not disc_y.values:
            text_ = "Features {} and {} have no values".format(disc_x, disc_y) \
                if not disc_x.values and \
                   not disc_y.values and \
                          disc_x != disc_y \
                else \
                    "Feature {} has no values".format(
                        disc_x if not disc_x.values else disc_y)
            text(text_, view.width() / 2 + 70, view.height() / 2,
                 Qt.AlignRight | Qt.AlignVCenter)
            return
        n = chi.n
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
        text(attr_y.name, 0, y_off + square_size / 2,
             Qt.AlignLeft | Qt.AlignVCenter, bold=True, vertical=True)
        text(attr_x.name, x_off + square_size / 2, bottom,
             Qt.AlignHCenter | Qt.AlignTop, bold=True)
        bottom += 30
        xl = text("χ²={:.2f}, p={:.3f}".format(chi.chisq, chi.p),
                  0, bottom)
        # Assume similar height for both lines
        text("N = " + fmt(chi.n), 0, bottom - xl.boundingRect().height())

    def get_widget_name_extension(self):
        if self.data is not None:
            return "{} vs {}".format(self.attr_x.name, self.attr_y.name)

    def send_report(self):
        self.report_plot()


def main():
    # pylint: disable=missing-docstring
    import sys
    from AnyQt.QtWidgets import QApplication
    a = QApplication(sys.argv)
    ow = OWSieveDiagram()
    ow.show()
    data = Table(r"zoo.tab")
    ow.set_data(data)
    a.exec_()
    ow.saveSettings()

if __name__ == "__main__":
    main()
