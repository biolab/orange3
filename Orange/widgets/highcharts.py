"""
This module provides Highcharts class, which is a thin wrapper around
Highcharts JS library.
"""
from json import dumps as json

from collections import defaultdict
from collections.abc import MutableMapping

from os.path import join, dirname

import numpy as np

from AnyQt.QtCore import QObject, pyqtSlot

from Orange.widgets.gui import WebviewWidget


def _Autotree():
    return defaultdict(_Autotree)


def _merge_dicts(master, update):
    """Merge dicts recursively in place (``master`` is modified)"""
    for k, v in master.items():
        if k in update:
            if isinstance(v, MutableMapping) and isinstance(update[k], MutableMapping):
                update[k] = _merge_dicts(v, update[k])
    master.update(update)
    return master


def _kwargs_options(kwargs):
    """Transforma a dict into a hierarchical dict.

    Example
    -------
    >>> (_kwargs_options(dict(a_b_c=1, a_d_e=2, x=3)) ==
    ...  dict(a=dict(b=dict(c=1), d=dict(e=2)), x=3))
    True
    """
    kwoptions = _Autotree()
    for kws, val in kwargs.items():
        cur = kwoptions
        kws = kws.split('_')
        for kw in kws[:-1]:
            cur = cur[kw]
        cur[kws[-1]] = val
    return kwoptions


class Highchart(WebviewWidget):
    """Create a Highcharts webview widget.

    Parameters
    ----------
    parent: QObject
        Qt parent object, if any.
    bridge: QObject
        Exposed as ``window.pybridge`` in JavaScript.
    options: dict
        Default options for this chart. See Highcharts docs. Some
        options are already set in the default theme.
    highchart: str
        One of `Chart`, `StockChart`, or `Map` Highcharts JS types.
    enable_zoom: bool
        Enables scroll wheel zooming and right-click zoom reset.
    enable_select: str
        If '+', allow series' points to be selected by clicking
        on the markers, bars or pie slices. Can also be one of
        'x', 'y', or 'xy' (all of which can also end with '+' for the
        above), in which case it indicates the axes on which
        to enable rectangle selection. The list of selected points
        for each input series (i.e. a list of arrays) is
        passed to the ``selection_callback``.
        Each selected point is represented as its index in the series.
        If the selection is empty, the callback parameter is a single
        empty list.
    javascript: str
        Additional JavaScript code to evaluate beforehand. If you
        need something exposed in the global namespace,
        assign it as an attribute to the ``window`` object.
    debug: bool
        Enables right-click context menu and inspector tools.
    **kwargs:
        The additional options. The underscores in argument names imply
        hierarchy, e.g., keyword argument such as ``chart_type='area'``
        results in the following object, in JavaScript::

            {
                chart: {
                    type: 'area'
                }
            }

        The original `options` argument is updated with options from
        these kwargs-derived objects.
    """

    _HIGHCHARTS_HTML = join(dirname(__file__), '_highcharts', 'chart.html')

    def __init__(self,
                 parent=None,
                 bridge=None,
                 options=None,
                 *,
                 highchart='Chart',
                 enable_zoom=False,
                 enable_select=False,
                 selection_callback=None,
                 javascript='',
                 debug=False,
                 **kwargs):
        options = (options or {}).copy()
        enable_select = enable_select or ''

        if not isinstance(options, dict):
            raise ValueError('options must be dict')
        if enable_select not in ('', '+', 'x', 'y', 'xy', 'x+', 'y+', 'xy+'):
            raise ValueError("enable_select must be '+', 'x', 'y', or 'xy'")
        if enable_select and not selection_callback:
            raise ValueError('enable_select requires selection_callback')

        super().__init__(parent, bridge, debug=debug)

        self.highchart = highchart
        self.enable_zoom = enable_zoom
        enable_point_select = '+' in enable_select
        enable_rect_select = enable_select.replace('+', '')
        if enable_zoom:
            _merge_dicts(options, _kwargs_options(dict(
                mapNavigation_enableMouseWheelZoom=True,
                mapNavigation_enableButtons=False)))
        if enable_select:
            self._selection_callback = selection_callback
            _merge_dicts(options, _kwargs_options(dict(
                chart_events_click='/**/unselectAllPoints/**/')))
        if enable_point_select:
            _merge_dicts(options, _kwargs_options(dict(
                plotOptions_series_allowPointSelect=True,
                plotOptions_series_point_events_click='/**/clickedPointSelect/**/')))
        if enable_rect_select:
            _merge_dicts(options, _kwargs_options(dict(
                chart_zoomType=enable_rect_select,
                chart_events_selection='/**/rectSelectPoints/**/')))
        if kwargs:
            _merge_dicts(options, _kwargs_options(kwargs))

        with open(self._HIGHCHARTS_HTML) as html:
            self.setHtml(html.read() % dict(javascript=javascript,
                                            options=json(options)),
                         self.toFileURL(dirname(self._HIGHCHARTS_HTML)) + '/')

    def contextMenuEvent(self, event):
        """ Zoom out on right click. Also disable context menu."""
        if self.enable_zoom:
            self.evalJS('chart.zoomOut(); 0;')
        super().contextMenuEvent(event)

    def exposeObject(self, name, obj):
        if isinstance(obj, np.ndarray):
            # Highcharts chokes on NaN values. Instead it prefers 'null' for
            # points it is not intended to show.
            obj = obj.astype(object)
            obj[np.isnan(obj)] = None
        super().exposeObject(name, obj)

    def chart(self, options=None, *,
              highchart=None, javascript='', javascript_after='', **kwargs):
        """ Populate the webview with a new Highcharts JS chart.

        Parameters
        ----------
        options, highchart, javascript, **kwargs:
            The parameters are the same as for the object constructor.
        javascript_after: str
            Same as `javascript`, except that the code is evaluated
            after the chart, available as ``window.chart``, is created.

        Notes
        -----
        Passing ``{ series: [{ data: some_data }] }``, if ``some_data`` is
        a numpy array, it is **more efficient** to leave it as numpy array
        instead of converting it ``some_data.tolist()``, which is done
        implicitly.
        """
        options = (options or {}).copy()
        if not isinstance(options, MutableMapping):
            raise ValueError('options must be dict')

        if kwargs:
            _merge_dicts(options, _kwargs_options(kwargs))
        self.exposeObject('pydata', options)
        highchart = highchart or self.highchart or 'Chart'
        self.evalJS('''
            {javascript};
            window.chart = new Highcharts.{highchart}(pydata); 0;
            {javascript_after};
        '''.format(javascript=javascript,
                   javascript_after=javascript_after,
                   highchart=highchart,))

    def clear(self):
        """Remove all series from the chart"""
        self.evalJS('''
            if (window.chart) {
                while(chart.series.length > 0) {
                    chart.series[0].remove(false);
                }
                chart.redraw();
            }; 0;
        ''')

    @pyqtSlot('QVariantList')
    def _on_selected_points(self, points):
        self._selection_callback([np.sort(selected).astype(int)
                                  for selected in points])

    def svg(self):
        """
        Returns div that is container of a chart.
        This method overrides svg method from WebView because
        SVG itself does not contain chart labels (title, axis labels, ...)
        """
        html = self.html()
        return html[html.index('<div id="container"'):html.rindex('</div>') + 6]


def main():
    """ A simple test. """
    from AnyQt.QtGui import QApplication
    app = QApplication([])

    def _on_selected_points(points):
        print(len(points), points)

    w = Highchart(enable_zoom=True, enable_select='xy+',
                  selection_callback=_on_selected_points,
                  debug=True)
    w.chart(dict(series=[dict(data=np.random.random((100, 2)))]),
            credits_text='BTYB Yours Truly',
            title_text='Foo plot',
            chart_type='scatter')
    w.show()
    app.exec()


if __name__ == '__main__':
    main()

