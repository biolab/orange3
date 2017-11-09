import re
from copy import deepcopy
from os import path, environ as env
from collections.abc import Sequence, MutableMapping

import numpy as np

from plotly.offline import plot
from plotly.graph_objs import *

from AnyQt.QtCore import QT_VERSION_STR, QUrl, QObject, pyqtSlot

from Orange.widgets.utils.webview import WebviewWidget


if QT_VERSION_STR < '5.5':
    raise RuntimeError('Plotly-based plots require Qt >= 5.3')


_DEFAULT_CONFIG = dict(
    # Reference
    # https://github.com/plotly/plotly.js/blob/540994d8dd527f65160a5d56ddec73eca2106b7a/src/plot_api/plot_config.js#L21-L111
    fillframe=True,
    doubleClick=False,
    showTips=False,
    sendData=False,
    displayLogo=False, displaylogo=False,
    scrollZoom=True,
    logging=False,
    # Reference:
    # https://github.com/plotly/plotly.js/blob/master/src/components/modebar/buttons.js
    modeBarButtonsToRemove=[
        'toImage', 'sendDataToCloud', 'autoScale2d',
        'hoverClosestCartesian', 'hoverCompareCartesian',
        'resetCameraLastSave3d', 'hoverClosest3d', 'hoverClosestGeo',
        'hoverClosestGl2d', 'hoverClosestPie', 'toggleHover',
        'resetViews'],
    modeBarButtons=[
        ['select2d', 'lasso2d',],
        ['pan2d',],
        ['zoom2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d',],
        # ['zoomInGeo', 'zoomOutGeo', 'resetGeo'],
        # ['zoom3d', 'resetCameraDefault3d'], ],
    ]
)

_DEFAULT_LAYOUT = dict(
    hovermode='closest',
    dragmode='select',
    showlegend=True,
    legend=dict(orientation='h',
                borderwidth=.5,),
    margin=dict(l=50, r=20, b=30, t=50, pad=0),
)

_MAX_TRACES_FOR_HORIZ_LEGEND = 10


def _merge_dicts(master, update):
    """Merge dicts recursively in place (``master`` is modified)"""
    for k, v in master.items():
        if k in update:
            if isinstance(v, MutableMapping) and isinstance(update[k], MutableMapping):
                update[k] = _merge_dicts(v, update[k])
    master.update(update)
    return master


class Plotly(WebviewWidget):
    """
    A Plotly-based visualization widget.

    Parameters
    ----------
    parent : QObject
    bridge : QObject
        An object exposed in JS as ``window.pybridge`` whose Qt
        signals / slots are available for calling.
    style : str
        Path to CSS stylesheet file or CSS style rules directly.
    javascript : str or list of str
        JavaScript code or path to .js file to evaluate in the global scope.
        Or a list thereof.
    """
    def __init__(self, parent=None, bridge=None, *, style='', javascript=''):

        class _Bridge(QObject):
            @pyqtSlot('QVariantList')
            def on_selected_points(_, indices):
                self.on_selected_points([np.sort(np.array(selected)).astype(int)
                                         for selected in indices])

            @pyqtSlot('QVariantList')
            def on_selected_range(_, ranges):
                self.on_selected_range(ranges)

        if bridge is not None:
            # Patch existing user-passed bridge with our selection callbacks
            assert isinstance(bridge, QObject), 'bridge needs to be a QObject'
            _Bridge = type(bridge.__class__.__name__,
                           bridge.__class__.__mro__,
                           dict(bridge.__dict__,
                                on_selected_points=_Bridge.on_selected_points,
                                on_selected_range=_Bridge.on_selected_range))

        super().__init__(parent=parent,
                         bridge=_Bridge(),
                         url=QUrl.fromLocalFile(path.join(path.dirname(__file__),
                                                          '_plotly', 'html.html')))
        if style:
            if path.isfile(style):
                with open(style) as f:
                    style = f.read()
            # Add each of the rules of the stylesheet into the document
            # If you feel this is overly complicated, be my guest to improve it
            for i, style in enumerate(filter(None, style.replace(' ', '').replace('\n', '').split('}'))):
                self.evalJS("document.styleSheets[0].insertRule('%s }', 0);" % style)

        if javascript:
            if isinstance(javascript, str):
                javascript = [javascript]
            for js in javascript:
                if path.isfile(js):
                    with open(js) as f:
                        js = f.read()
                self.evalJS(js)

    def plot(self, data, layout=None, *, scroll_zoom=False, **config_opts):
        """
        Plot a new plot in the webview.

        Parameters
        ----------
        data : list or dict
            Plotly data as accepted by ``plotly.offline.plot()`` function.
        layout : dict
            Plotly layout configuration. Can also be passed in
            ``data['layout']`` when `data` is a dict.
        scroll_zoom : bool
            Enable scroll wheel zoom.
        **config_opts
            Plot view options, as accepted by ``plotly.offline.plot(config)``.
        """
        if isinstance(data, Sequence) and not isinstance(data, dict):
            data = dict(data=data)

        default_layout = deepcopy(_DEFAULT_LAYOUT)
        if len(data['data']) > _MAX_TRACES_FOR_HORIZ_LEGEND:
            default_layout['legend']['orientation'] = 'v'
        if any('lines' in s.mode for s in data['data']
               if isinstance(s, Scatter)):
            default_layout['hovermode'] = 'y'

        data['layout'] = _merge_dicts(_merge_dicts(default_layout,
                                                   data.get('layout', {})),
                                      layout or {})

        html = plot(data, output_type='div', include_plotlyjs=False,
                    auto_open=False, show_link=False,
                    config=dict(_DEFAULT_CONFIG,
                                scroll_zoom=scroll_zoom,
                                logging=env.get('ORANGE_DEBUG'),
                                **config_opts))
        _, script = html.split('<script type="text/javascript">')
        script = script[:-len('</script>')]
        script = re.sub(r'(\bPlotly\.newPlot\(")[^"]+', r'\1container', script, 1)
        self.evalJS(script + '; _after_plot();')

    def clear(self):
        """Clear the current plot, if any."""
        self.evalJS('Plotly.purge(container);')

    def on_selected_points(self, indices):
        """OVERRIDE THIS

        Called on selection.

        Parameters
        ----------
        indices : list of ndarray
            Sorted indices of selected points for each trace of data.
        """

    def on_selected_range(self, ranges):
        """OVERRIDE THIS

        Called on selection.

        Parameters
        ----------
        ranges : list
            List of [min_value, max_value] pairs for each data dimension.
        """


if __name__ == '__main__':
    from AnyQt.QtWidgets import QApplication
    app = QApplication([])

    w = Plotly()
    w.on_selected_points = print
    w.show()
    x, y = np.random.random((2, 100))
    w.plot([
        Scatter(x=x, y=y, mode='markers'),
        Scatter(x=x, y=x, mode='markers'),
        Scatter(x=y, y=x, mode='markers'),
        Scatter(x=y, y=y, mode='markers'),
    ])

    app.exec()
