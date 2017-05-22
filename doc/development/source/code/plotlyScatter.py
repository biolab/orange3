from Orange.data import Table
from Orange.widgets.widget import OWWidget

from Orange.widgets.utils.plotly_widget import Plotly

import plotly.graph_objs as go


class ScatterPlot(Plotly):
    """
    ScatterPlot class extends Plotly class so we can make necessary
    adjustments, here override the two methods below.
    """
    def on_selected_range(self, ranges):
        """
        This callback is called whenever the selection changes and its
        argument is a list of extremes of each data dimension.
        """
        print('X [min, max]:', ranges[0])
        print('Y [min, max]:', ranges[1])

    def on_selected_points(self, indices):
        """
        This callback is called whenever the selection changes and its
        argument is the list of indexes of selected points per each input
        trace.
        """
        for i, points in enumerate(indices, 1):
            print('Selected points on trace %d:' % i, points.tolist())


class OWPlotlyScatter(OWWidget):
    name = "Plotly Iris Scatter"
    description = "Iris dataset scatter plot demo using Plotly JS"

    def __init__(self):
        # Create an instance of ScatterPlot and put it in the Orange
        # widget's main area
        self.plot = ScatterPlot()
        self.mainArea.layout().addWidget(self.plot)
        # Then plot "iris" dataset on it
        self.plot_iris()

    def plot_iris(self):
        data = Table('iris')

        x = data[:, 'sepal length'].X.ravel()
        y = data[:, 'sepal width'].X.ravel()
        iris = data[:, 'iris'].Y.ravel()
        iris_names = data.domain['iris'].values
        symbol_shapes = ('circle', 'square', 'triangle')

        # Plot each flower type as a distinct trace of go.Scatter type
        traces = [
            go.Scatter(x=x[iris == i],
                       y=y[iris == i],
                       name=name,
                       mode='markers',
                       marker=dict(symbol=symbol))
            for i, (name, symbol) in enumerate(zip(iris_names,
                                                   symbol_shapes))]
        self.plot.plot(data=traces,
                       layout=dict(xaxis=dict(title='Sepal length'),
                                   yaxis=dict(title='Sepal width')))


if __name__ == "__main__":
    from AnyQt.QtWidgets import QApplication
    app = QApplication([])
    ow = OWPlotlyScatter()
    ow.show()
    app.exec_()
