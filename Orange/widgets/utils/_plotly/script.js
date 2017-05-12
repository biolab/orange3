const container = document.querySelector('#container');

/* Autosize on window resize */
window.addEventListener('resize', function() {
    Plotly.Plots.resize(container);
}, true);


function _saveModState(ev) {
   window._shiftKey = ev.shiftKey;
   window._ctrlKey = ev.ctrlKey;
   window._altKey = ev.altKey;
}
window.addEventListener('keydown', _saveModState);
window.addEventListener('keyup', _saveModState);

function _after_plot() {
    // Connect events
    function log(ev_name) {
        return function(ev) { console.log(ev_name, ev) };
    }
    [
        'selected',
        'redraw',
        'relayout',
        'afterplot',
        'autosize',
        'deselect',
        'hover',
        'unhover',
        'restyle'
    ].map(function(ev_name) {
        container.on('plotly_' + ev_name,
                     window['on_plotly_' + ev_name] || log('plotly_' + ev_name));
    });
    container.on('plotly_relayout', function(event) {
        console.log('got', event);
    });
}

function on_plotly_selected(eventData) {
    console.log('plotly_selected', eventData);
    var indices = [],
        points = eventData.points;
    for (var i = 0; i < container.data.length; ++i)
        indices.push([]);
    for (var i = 0; i < points.length; ++i) {
        var pt = points[i];
        indices[pt.curveNumber].push(pt.pointNumber);
    }
    pybridge.on_selected_points(indices);
    pybridge.on_selected_range([
        eventData.range.x || [],
        eventData.range.y || [],
        eventData.range.z || [],
    ]);
}
