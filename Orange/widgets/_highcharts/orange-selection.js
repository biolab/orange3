/**
 * Our selection-handling functionality.
 */

Highcharts.Chart.prototype.deselectPointsIfNot = function(accumulate) {
    // If no Shift or Ctrl modifier, clear the existing selection
    if (!accumulate) {
        var points = this.getSelectedPoints();
        for (var i = 0; i < points.length; ++i) {
            points[i].select(false, true);
        }
    }
};

Highcharts.Chart.prototype.getSelectedPointsForExport = function() {
    /**
     * The original getSelectedPoints object is too complex for QWebView
     * bridge. Let's just take what we need.
     */
    var points = [],
        selected = this.getSelectedPoints();
    for (var i = 0; i < this.series.length; ++i)
        points.push([]);
    for (var i = 0; i < selected.length; ++i) {
        var p = selected[i];
        points[p.series.index].push(p.index);
    }
    return points;
};

function unselectAllPoints(e) {
    // Only handle left click on the canvas area, if no modifier pressed
    if (e.ctrlKey  ||
        e.shiftKey ||
        !(e.which == 1 &&
          e.target.parentElement &&
          e.target.parentElement.tagName.toLowerCase() == 'svg'))
        return true;
    this.deselectPointsIfNot(false);
    __self._on_selected_points([]);
}

function clickedPointSelect(e) {
    var chart = this.series.chart;
    chart.deselectPointsIfNot(e.shiftKey || e.ctrlKey);
    var points = chart.getSelectedPointsForExport();
    if (this.selected) { // Already selected, this click should deselect
        var selected = points[this.series.index];
        selected.splice(selected.indexOf(this.index), 1);
    } else
        points[this.series.index].push(this.index);
    __self._on_selected_points(points);
    return true;
}

function rectSelectPoints(e) {
    if (!(e.originalEvent && e.originalEvent.which == 1))
        return true;
    e.preventDefault();  // Don't zoom

    var no_xAxis = !e.xAxis || !e.xAxis.length,
        no_yAxis = !e.yAxis || !e.yAxis.length,
        xMin = no_xAxis || e.xAxis[0].min,
        xMax = no_xAxis || e.xAxis[0].max,
        yMin = no_yAxis || e.yAxis[0].min,
        yMax = no_yAxis || e.yAxis[0].max,
        series = this.series,
        accumulate = e.originalEvent.shiftKey || e.originalEvent.ctrlKey,
        newstate = e.originalEvent.ctrlKey ? undefined /* =toggle */ : true;

    this.deselectPointsIfNot(accumulate);

    // Select the points
    for (var i=0; i < series.length; ++i) {
        var points = series[i].points;
        for (var j=0; j < points.length; ++j) {
            var point = points[j], x = point.x, y = point.y;
            if ((no_xAxis || (x >= xMin && x <= xMax)) &&
                (no_yAxis || (y >= yMin && y <= yMax)) &&
                point.series.visible) {
                point.select(newstate, true);
            }
        }
    }

    __self._on_selected_points(this.getSelectedPointsForExport());
    return false;  // Don't zoom
}
