"""A scatter plot example using Highcharts"""

from itertools import chain

import numpy as np

from Orange.data import Table
from Orange.widgets import gui, settings, widget, highcharts


class Scatterplot(highcharts.Highchart):
    """
    Scatterplot extends Highchart and just defines some sane defaults:
    * enables scroll-wheel zooming,
    * enables rectangle (+ individual point) selection,
    * sets the chart type to 'scatter' (could also be 'bubble' or as
      appropriate; Se Highcharts JS docs)
    * sets the selection callback. The callback is passed a list (array)
      of indices of selected points for each data series the chart knows
      about.
    """
    def __init__(self, selection_callback, **kwargs):
        super().__init__(enable_zoom=True,
                         enable_select='xy+',
                         chart_type='scatter',
                         selection_callback=selection_callback,
                         **kwargs)


class OWScatterPlot(widget.OWWidget):
    """Example scatter plot visualization using Highcharts"""
    name = 'Simple Scatter Plot'
    description = 'An example scatter plot visualization using Highcharts.'
    icon = "icons/ScatterPlot.svg"

    inputs = [("Data", Table, "set_data")]
    outputs = [("Selected Data", Table)]

    attr_x = settings.Setting('')
    attr_y = settings.Setting('')

    graph_name = 'scatter'

    def __init__(self):
        super().__init__()
        self.data = None
        self.indices = None
        self.n_selected = 0
        self.series_rows = []
        # Create the UI controls for selecting axes attributes
        box = gui.vBox(self.controlArea, 'Axes')
        self.cbx = gui.comboBox(box, self, 'attr_x',
                                label='X:',
                                orientation='horizontal',
                                callback=self.replot,
                                sendSelectedValue=True)
        self.cby = gui.comboBox(box, self, 'attr_y',
                                label='Y:',
                                orientation='horizontal',
                                callback=self.replot,
                                sendSelectedValue=True)
        gui.label(self.controlArea, self, '%(n_selected)d points are selected',
                  box='Info')
        gui.rubber(self.controlArea)

        # Create an instance of Scatter plot. Initial Highcharts configuration
        # can be passed as '_'-delimited keyword arguments. See Highcharts
        # class docstrings and Highcharts API documentation for more info and
        # usage examples.
        self.scatter = Scatterplot(selection_callback=self.on_selection,
                                   xAxis_gridLineWidth=0,
                                   yAxis_gridLineWidth=0,
                                   title_text='Scatterplot example',
                                   tooltip_shared=False,
                                   # In development, we can enable debug mode
                                   # and get right-click-inspect and related
                                   # console utils available:
                                   debug=True)
        # Just render an empty chart so it shows a nice 'No data to display'
        # warning
        self.scatter.chart()

        self.mainArea.layout().addWidget(self.scatter)

    def set_data(self, data):
        self.data = data

        # When the widget receives new data, we need to:

        # ... reset the combo boxes ...

        def init_combos():
            self.cbx.clear()
            self.cby.clear()
            for var in data.domain if data is not None else []:
                if var.is_primitive():
                    self.cbx.addItem(gui.attributeIconDict[var], var.name)
                    self.cby.addItem(gui.attributeIconDict[var], var.name)

        init_combos()

        # If the data is actually None, we should just
        # ... reset the scatter plot, selected indices ...
        if data is None:
            self.scatter.clear()
            self.indices = None
            self.commit()
            return

        # ... else, select the first two attributes and replot the scatter.
        if len(data.domain) >= 2:
            self.attr_x = self.cbx.itemText(0)
            self.attr_y = self.cbx.itemText(1)
        self.replot()

    def replot(self):
        # Brace yourself ...

        if self.data is None or not self.attr_x or not self.attr_y:
            # Sanity checks failed; nothing to do
            return

        data = self.data
        attr_x, attr_y = data.domain[self.attr_x], data.domain[self.attr_y]

        # Highcharts widget accepts an options dict. This dict is converted
        # to options Object Highcharts JS uses in its examples. All keys are
        # **exactly the same** as for Highcharts JS.
        options = dict(series=[])

        # For our scatter plot, we need data in a standard numpy 2D array,
        # with x and y values in the two columns ...
        cols = []
        for attr in (attr_x, attr_y):
            subset = data[:, attr]
            cols.append(subset.Y if subset.Y.size else subset.X)
        # ... that's our X here
        X = np.column_stack(cols)

        # Highcharts point selection returns indexes of selected points per
        # each input series. Thus we should maintain a "map" of such indices
        # into the original data table.
        self.series_rows = []

        # If data has a discrete class, we want to color nodes by it, and we
        # do so by constructing a separate instance series for each class
        # value. This is one way of doing it. If you know of a better one,
        # you must be so lucky and I envy you!!
        if data.domain.has_discrete_class:
            y = data[:, data.domain.class_var].Y.ravel()
            for yval, yname in enumerate(data.domain.class_var.values):
                rows = (y == yval).nonzero()[0]
                self.series_rows.append(rows)
                options['series'].append(dict(data=X[rows], name=yname))
        # If data doesn't have a discrete class, just use the whole data as
        # a single series (colored with default color — no gradient fill in
        # this example).
        else:
            self.series_rows.append(np.arange(len(X)))
            options['series'].append(dict(data=X, showInLegend=False))

        # Besides the options dict, Highcharts can also be passed keyword
        # parameters, where each parameter is split on underscores in
        # simulated object hierarchy. This works:
        kwargs = dict(
            xAxis_title_text=attr_x.name,
            yAxis_title_text=attr_y.name,
            tooltip_headerFormat=(
                '<span style="color:{point.color}">\u25CF</span> '
                '{series.name} <br/>'),
            tooltip_pointFormat=(
                '<b>{attr_x.name}:</b> {{point.x}}<br/>'
                '<b>{attr_y.name}:</b> {{point.y}}<br/>').format_map(locals()))
        # If any of selected attributes is discrete, we correctly scatter it
        # as a categorical
        if attr_x.is_discrete:
            kwargs['xAxis_categories'] = attr_x.values
        if attr_y.is_discrete:
            kwargs['yAxis_categories'] = attr_y.values

        # That's it, we can scatter our scatter by calling its chart method
        # with the parameters we'd constructed
        self.scatter.chart(options, **kwargs)

    def on_selection(self, indices):
        # When points on the scatter plot are selected, this method is called.
        # Variable indices contains indices of selected points **per each
        # input series** (series in the options object above).
        # Luckily, we kept original data indices that form each of the
        # series ...
        self.indices = list(chain.from_iterable(
            self.series_rows[i][selected]
            for i, selected in enumerate(indices)
        ))

        # Let's give the user some feedback
        self.n_selected = len(self.indices)

        # And that's it, we can commit the output!
        self.commit()

    def commit(self):
        self.send('Selected Data',
                  self.data[self.indices] if self.indices else None)

    def send_report(self):
        self.report_data('Data', self.data)
        self.report_raw('Scatter plot', self.scatter.svg())


def main():
    from PyQt4.QtGui import QApplication
    app = QApplication([])
    ow = OWScatterPlot()
    data = Table("iris")
    ow.set_data(data)
    ow.show()
    app.exec_()


if __name__ == "__main__":
    main()
