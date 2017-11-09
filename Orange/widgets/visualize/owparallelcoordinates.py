import itertools
import textwrap

from os import path

import numpy as np
from scipy.stats import kendalltau

from Orange.data import Table, TimeVariable
from Orange.util import color_to_hex
from Orange.widgets import gui, widget, settings
from Orange.widgets.utils.annotated_data import create_annotated_table
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.plotly_widget import Plotly

from AnyQt.QtCore import Qt, QObject, pyqtSlot
from AnyQt.QtWidgets import QMessageBox

from plotly.graph_objs import Parcoords


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class ParallelCoordinates(Plotly):
    BASEDIR = path.join(path.dirname(__file__),
                        '_' + path.splitext(path.basename(__file__))[0])
    CSS_FILE = path.join(BASEDIR, 'style.css')
    JS_FILE = path.join(BASEDIR, 'script.js')

    def __init__(self, parent):

        class _Bridge(QObject):
            @pyqtSlot('QVariantList')
            def update_axes_info(_, axes_info):
                self._owwidget.selected_attrs = [name for name, _ in axes_info]
                self._owwidget.constraint_range = {name: range
                                                   for name, range in axes_info
                                                   if range}
                self._owwidget.commit()

        self._owwidget = parent
        super().__init__(parent, _Bridge(),
                         style=self.CSS_FILE, javascript=self.JS_FILE)

    def plot(self, data, *, padding_right=0):
        return super().plot(data, scroll_zoom=False,
                            layout=dict(margin=dict(l=50, t=50, b=10, r=20 + padding_right)),
                            modeBarButtons=[])


class OWParallelCoordinates(widget.OWWidget):
    name = "Parallel Coordinates"
    description = "Parallel coordinates display of multi-dimensional data."
    icon = "icons/ParallelCoordinates.svg"
    priority = 900
    inputs = [("Data", Table, 'set_data', widget.Default),
              ("Features", widget.AttributeList, 'set_shown_attributes')]
    outputs = [("Selected Data", Table, widget.Default),
               ("Annotated Data", Table),
               ("Features", widget.AttributeList)]

    graph_name = 'graph'
    settingsHandler = settings.DomainContextHandler()

    autocommit = settings.Setting(True)
    selected_attrs = settings.ContextSetting([])
    color_attr = settings.ContextSetting('')
    constraint_range = settings.ContextSetting({})

    autocommit = settings.Setting(default=True)

    UserAdviceMessages = [
        widget.Message('You can select subsets of data based on value intervals '
                       'by dragging on the corresponding dimensions\' axes.\n\n'
                       'You can reset the selection by clicking somewhere '
                       'outside the selected interval on the axis.',
                       'subset-selection')
    ]

    class Warning(widget.OWWidget.Warning):
        too_many_selected_dimensions = widget.Msg(
            'Too many dimensions selected ({}). Only first {} shown.')

    class Information(widget.OWWidget.Information):
        dataset_sampled = widget.Msg('Showing a random sample of your data.')

    OPTIMIZATION_N_DIMS = (3, 9)
    MAX_N_DIMS = 20

    def __init__(self):
        super().__init__()
        self.graph = ParallelCoordinates(self)
        self.mainArea.layout().addWidget(self.graph)

        self.model = DomainModel(valid_types=DomainModel.PRIMITIVE)
        self.colormodel = DomainModel(valid_types=DomainModel.PRIMITIVE)

        box = gui.vBox(self.controlArea, 'Lines')
        combo = gui.comboBox(box, self, 'color_attr',
                             sendSelectedValue=True,
                             label='Color:', orientation=Qt.Horizontal,
                             callback=self.update_plot)
        combo.setModel(self.colormodel)

        box = gui.vBox(self.controlArea, 'Dimensions')
        view = gui.listView(box, self, 'selected_attrs', model=self.model,
                            callback=self.update_plot)
        view.setSelectionMode(view.ExtendedSelection)
        # Prevent drag selection. Otherwise, each new addition to selectio`n
        # the mouse passes over triggers a webview redraw. Sending lots of data
        # around multiple times on large datasets results in stalling and crashes.
        view.mouseMoveEvent = (lambda event:
            None if view.state() == view.DragSelectingState else
                super(view.__class__, view).mouseMoveEvent(event))

        self.optimize_button = gui.button(
            box, self, 'Optimize Selected Dimensions',
            callback=self.optimize,
            tooltip='Optimize visualized dimensions by maximizing cumulative '
                    'Kendall rank correlation coefficient.')

        gui.auto_commit(self.controlArea, self, 'autocommit', '&Apply')

    def set_data(self, data):
        self.data = data
        self.graph.clear()

        self.closeContext()

        model = self.model
        colormodel = self.colormodel

        self.sample = None
        self.selected_attrs = None
        self.color_attr = None

        if data is not None and len(data) and len(data.domain.variables):
            self.sample = slice(None) if len(data) < 2000 else np.random.choice(np.arange(len(data)), 2000, replace=False)
            model.set_domain(data.domain)
            colormodel.set_domain(data.domain)
            self.color_attr = None
            selected_attrs = (model.data(model.index(i, 0))
                              for i in range(min(self.OPTIMIZATION_N_DIMS[1],
                                                 model.rowCount())))
            self.selected_attrs = [attr for attr in selected_attrs
                                   if isinstance(attr, str)]
        else:
            model.set_domain(None)
            colormodel.set_domain(None)

        self.Information.dataset_sampled(
            shown=False if data is None else len(data))

        self.openContext(data.domain)

        self.update_plot()
        self.commit()

    def clear(self):
        self.graph.clear()
        self.commit()

    def update_plot(self):
        data = self.data
        if data is None or not len(data):
            self.clear()
            return

        self.optimize_button.setDisabled(not self.is_optimization_valid())

        self.Warning.too_many_selected_dimensions(
            len(self.selected_attrs), self.MAX_N_DIMS,
            shown=len(self.selected_attrs) > self.MAX_N_DIMS)
        selected_attrs = self.selected_attrs[:self.MAX_N_DIMS]

        sample = self.sample

        dimensions = []
        for attr in selected_attrs:
            attr = data.domain[attr]
            values = data.get_column_view(attr)[0][sample]
            dim = dict(label=attr.name,
                       values=values,
                       constraintrange=self.constraint_range.get(attr.name))
            if attr.is_discrete:
                dim.update(tickvals=np.arange(len(attr.values)),
                           ticktext=attr.values)
            elif isinstance(attr, TimeVariable):
                tickvals = [np.nanmin(values),
                            np.nanmedian(values),
                            np.nanmax(values)]
                ticktext = [attr.repr_val(i)
                            for i in tickvals]
                dim.update(tickvals=tickvals,
                           ticktext=ticktext)
            dimensions.append(dim)

        # Compute color legend
        line = dict()
        padding_right = 40
        if self.color_attr:
            attr = data.domain[self.color_attr]
            values = data.get_column_view(attr)[0][sample]
            line.update(color=values, showscale=True)
            title = '<br>'.join(textwrap.wrap(attr.name.strip(), width=7,
                                              max_lines=4, placeholder='…'))
            if attr.is_discrete:
                padding_right = 90
                colors = [color_to_hex(i) for i in attr.colors]
                values_short = [textwrap.fill(value, width=9,
                                              max_lines=1, placeholder='…')
                                for value in attr.values]
                self.graph.exposeObject('discrete_colorbar',
                                        dict(colors=colors,
                                             title=title,
                                             values=attr.values,
                                             values_short=values_short))
                line.update(showscale=False,
                            colorscale=list(zip(np.linspace(0, 1, len(attr.values)),
                                                colors)))
            else:
                padding_right = 0
                self.graph.exposeObject('discrete_colorbar', {})
                line.update(colorscale=list(zip((0, 1),
                                                (color_to_hex(i) for i in attr.colors[:-1]))),
                            colorbar=dict(title=title))
                if isinstance(attr, TimeVariable):
                    tickvals = [np.nanmin(values),
                                np.nanmedian(values),
                                np.nanmax(values)]
                    ticktext = [attr.repr_val(i)
                                for i in tickvals]
                    line.update(colorbar=dict(title=title,
                                              tickangle=-90,
                                              tickvals=tickvals,
                                              ticktext=ticktext))
        self.graph.plot([Parcoords(line=line,
                                   dimensions=dimensions)],
                        padding_right=padding_right)

    def set_shown_attributes(self, attrs):
        self.selected_attrs = attrs
        self.update_plot()

    def commit(self):
        selected_data, annotated_data = None, None
        data = self.data
        if data is not None and len(data):

            mask = np.ones(len(data), dtype=bool)
            for attr, (min, max) in self.constraint_range.items():
                values = data.get_column_view(attr)[0]
                mask &= (values >= min) & (values <= max)

            selected_data = data[mask]
            annotated_data = create_annotated_table(data, mask)

        self.send('Selected Data', selected_data)
        self.send('Annotated Data', annotated_data)
        self.send('Features', widget.AttributeList(self.selected_attrs))

    def is_optimization_valid(self):
        return (self.OPTIMIZATION_N_DIMS[0] <= len(self.selected_attrs) <= self.OPTIMIZATION_N_DIMS[1])

    def optimize(self):
        """ Optimizes the order of selected dimensions. """
        data = self.data
        if data is None or not len(data):
            return

        if not self.is_optimization_valid():
            QMessageBox(QMessageBox.Warning,
                        "Parallel Coordinates Optimization",
                        "Can only optimize when the number of selected dimensions "
                        "is between {} and {}. "
                        "Sorry.".format(*self.OPTIMIZATION_N_DIMS),
                        QMessageBox.Abort, self).exec()
            return

        self.optimize_button.blockSignals(True)

        R = {}
        Rc = {}
        sample = slice(None) if len(data) < 300 else np.random.choice(np.arange(len(data)), 300, replace=False)

        for attr1 in self.selected_attrs:
            if self.color_attr:
                Rc[attr1] = kendalltau(data.get_column_view(attr1)[0][sample],
                                       data.get_column_view(self.color_attr)[0][sample],
                                       nan_policy='omit')[0]
            for attr2 in self.selected_attrs:
                if (attr1, attr2) in R or attr1 == attr2:
                    continue
                R[(attr1, attr2)] = R[(attr2, attr1)] = \
                    kendalltau(data.get_column_view(attr1)[0][sample],
                               data.get_column_view(attr2)[0][sample],
                               nan_policy='omit')[0]

        # First dimension is the one with the highest correlation with the
        # color attribute; the last dimension the one with the lowest
        # correlation with the first dimension.
        # If there is no color attribute, first and last are the two dimensions
        # with the lowest correlation.
        # In either case, the rest are filled in in the order of maximal
        # cumulative correlation.
        if self.color_attr:
            head = max(Rc.items(), key=lambda i: i[1])[0]
            tail = min(((key, value)
                        for key, value in R.items()
                        if key[0] == head), key=lambda i: i[1])[0][1]
        else:
            head, tail = min(R.items(), key=lambda i: i[1])[0]

        def cumsum(permutation):
            return sum(R[(attr1, attr2)]
                       for attr1, attr2 in pairwise((head,) + permutation + (tail,)))

        body = max(itertools.permutations(set(self.selected_attrs) - set([head, tail])),
                   key=cumsum)

        self.selected_attrs = (head,) + body + (tail,)
        self.update_plot()

        self.optimize_button.blockSignals(False)

    def send_report(self):
        self.report_items((
            ('Dimensions', [i.name for i in self.selected_attrs]),
            ('Color', self.color_attr)))
        self.report_plot()


if __name__ == "__main__":
    from AnyQt.QtWidgets import QApplication
    from Orange.data import Domain, ContinuousVariable, DiscreteVariable
    a = QApplication([])
    ow = OWParallelCoordinates()
    ow.show()

    N_RANDOM = 15
    x = np.random.random(size=10000) * 10
    X = np.column_stack((x, -x, np.sin(x), np.cos(x), np.arctan(x), np.exp(x), x*x, np.sqrt(x), x > 5,
                         np.random.random((len(x), N_RANDOM))))
    domain = Domain([ContinuousVariable(i)
                     for i in 'x -x sin(x) cos(x) atan(x) exp(x) x*x sqrt(x)'.split()] +
                    [DiscreteVariable('x > 5', values=['false', 'true'])] +
                    [ContinuousVariable('random ' + str(i))
                     for i in range(N_RANDOM)])
    data = Table(domain, X)

    ow.set_data(data)
    ow.handleNewSignals()

    a.exec_()

    ow.saveSettings()
