"""
This module provides Highcharts class, which is a thin wrapper around
Highcharts JS library.
"""
from json import dumps as json

from collections import defaultdict
from collections.abc import MutableMapping, Mapping, Set, Sequence, Iterable

from os.path import join, dirname
from urllib.parse import urljoin
from urllib.request import pathname2url

import numpy as np

from PyQt4.QtCore import QUrl, QObject, pyqtProperty, pyqtSlot, QTimer
from PyQt4.QtGui import QColor

from Orange.widgets.webview import WebView


def _Autotree():
    return defaultdict(_Autotree)


def _to_primitive_types(d):
    # pylint: disable=too-many-return-statements
    if isinstance(d, np.integer):
        return int(d)
    if isinstance(d, (float, np.floating)):
        return float(d) if not np.isnan(d) else None
    if isinstance(d, (str, int, bool)):
        return d
    if isinstance(d, np.ndarray):
        # Highcharts chokes on NaN values. Instead it prefers 'null' for
        # points it is not intended to show.
        new = d.astype(object)
        new[np.isnan(d)] = None
        return new.tolist()
    if isinstance(d, Mapping):
        return {k: _to_primitive_types(d[k]) for k in d}
    if isinstance(d, Set):
        return {k: 1 for k in d}
    if isinstance(d, (Sequence, Iterable)):
        return [_to_primitive_types(i) for i in d]
    if d is None:
        return None
    if isinstance(d, QColor):
        return d.name()
    raise TypeError


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


class Highchart(WebView):
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

    _HIGHCHARTS_HTML = urljoin(
        'file:', pathname2url(join(join(dirname(__file__), '_highcharts'), 'chart.html')))

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

        super().__init__(parent, bridge,
                         debug=debug,
                         url=QUrl(self._HIGHCHARTS_HTML))
        self.debug = debug
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
            self.frame.addToJavaScriptWindowObject('__highchart', self)
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

        super_evalJS = super().evalJS
        super_evalJS('window.__js_queue = [];')
        self._is_init = False

        def evalOptions():
            super_evalJS(javascript)
            super_evalJS('''
                var options = {options};
                fixupOptionsObject(options);
                Highcharts.setOptions(options);
            '''.format(options=json(options)))
            self._is_init = True

        self.frame.loadFinished.connect(evalOptions)

    def contextMenuEvent(self, event):
        """ Zoom out on right click. Also disable context menu."""
        if self.enable_zoom:
            self.evalJS('chart.zoomOut();')
        if self.debug:
            super().contextMenuEvent(event)

    @staticmethod
    def _JSObject_factory(obj):
        pyqt_type = type(obj).__mro__[-2]
        if isinstance(obj, (list, np.ndarray)):
            pyqt_type = 'QVariantList'
        elif isinstance(obj, Mapping):
            pyqt_type = 'QVariantMap'
        else:
            raise TypeError("Can't expose object of type {}. Too easy. Use "
                            "evalJS method instead.".format(type(obj)))

        class _JSObject(QObject):
            """ This class hopefully prevent options data from being marshalled
            into a string-like dumb (JSON) object when passed into JavaScript. """
            def __init__(self, parent, obj):
                super().__init__(parent)
                self._obj = obj

            @pyqtProperty(pyqt_type)
            def _options(self):
                return self._obj

        return _JSObject

    def exposeObject(self, name, obj):
        """Expose the object `obj` as ``window.<name>`` in JavaScript.

        If the object contains any string values that start and end with
        literal ``/**/``, those are evaluated as JS expressions the result
        value replaces the string in the object.

        The exposure, as defined here, represents a snapshot of object at
        the time of execution. Any future changes on the original Python
        object are not (necessarily) visible in its JavaScript counterpart.

        Parameters
        ----------
        name: str
            The global name the object is exposed as.
        obj: object
            The object to expose. Must contain only primitive types, such as:
            int, float, str, bool, list, dict, set, numpy.ndarray.
        """
        try:
            obj = _to_primitive_types(obj)
        except TypeError:
            raise TypeError(
                'object must consist of primitive types '
                '(allowed: int, float, str, bool, list, '
                'dict, set, numpy.ndarray, ...)') from None

        pydata = self._JSObject_factory(obj)(self, obj)
        self.frame.addToJavaScriptWindowObject('_' + name, pydata)
        self.evalJS('''
            window.{0} = window._{0}._options;
            fixupOptionsObject({0});
        '''.format(name))

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
            window.chart = new Highcharts.{highchart}(pydata);
            {javascript_after};
        '''.format(javascript=javascript,
                   javascript_after=javascript_after,
                   highchart=highchart,))

    def evalJS(self, javascript):
        """ Asynchronously evaluate JavaScript code. """
        _ENQUEUE = '__js_queue.push(function() { %s; });'
        evalJS = super().evalJS

        def _dequeue():
            if not self._is_init:
                QTimer.singleShot(1, _dequeue)
                return
            return evalJS('while (__js_queue.length) (__js_queue.shift())();')

        evalJS(_ENQUEUE % javascript)
        return _dequeue()

    def clear(self):
        """Remove all series from the chart"""
        self.evalJS('''
            if (window.chart) {
                while(chart.series.length > 0) {
                    chart.series[0].remove(false);
                }
                chart.redraw();
            }
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
        html = self.frame.toHtml()
        return html[html.index('<div id="container"'):html.rindex('</div>') + 6]


def main():
    """ A simple test. """
    from PyQt4.QtGui import QApplication
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

