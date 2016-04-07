/**
 * Our general helpers for Highcharts, JS, QWebView bridge ...
 */

function fixupOptionsObject(obj) {
    /**
     * Replace any strings with their eval'd value if they
     * start with an empty JavaScript block comment, i.e. these
     * four characters:
     */                 /**/
    if (typeof obj === 'undefined' || obj === null)
        return;

    var keys = Object.keys(obj);
    for (var i=0; i<keys.length; ++i) {
        var key = keys[i],
            val = obj[key];

        if (typeof val === 'string' && val.indexOf('/**/') == 0) {
            obj[key] = eval(val)
        } else if (val.constructor === Object ||
                   val.constructor === Array && key === 'series')
            fixupOptionsObject(val);
    }
}
