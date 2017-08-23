NodeList.prototype.forEach = Array.prototype.forEach;


function on_plotly_restyle() {
    console.log('restyle');
    pybridge.update_axes_info(container.data[0].dimensions.map(function(dim) {
        return [dim.label, dim.constraintrange || []]
    }));
}


function on_plotly_afterplot() {
    ellipsize_labels();
    add_discrete_legend();
}


/**
 * Ellipsize dimension labels
 *
 * Remove once this is fixed:
 * https://github.com/plotly/plotly.js/issues/1703
 */
function ellipsize_labels() {
    var MAX_WIDTH = window.innerWidth / (container.data[0].dimensions.length + 1) - 20;
    console.log('relayout');
    document.querySelectorAll('.axisHeading .axisTitle').forEach(function (el) {
        el.textContent = el.__data__.label;

        while (el.textContent.length > 2 &&
               el.getComputedTextLength() > MAX_WIDTH) {
            el.textContent = el.textContent.slice(0, -2);
            el.textContent += '…';
        }

        // Add <title> element so tooltip works in SVG
        var title = document.createElementNS("http://www.w3.org/2000/svg", "title");
        el.insertBefore(title, el.firstChild);
        title.textContent = el.__data__.label;
    });
}


/**
 * Show or hide discrete legend. Expects window.discrete_colorbar config object.
 *
 * Remove once this is fixed:
 * https://github.com/plotly/plotly.js/issues/1968
 */
function add_discrete_legend() {
    // If not discrete colorbar, nothing to do here
    if (Object.keys(window.discrete_colorbar || {}).length === 0) {
        document.getElementById('customjs-container').innerHTML = '';
        return;
    }

    var html = '',
        colors = discrete_colorbar.colors,
        values = discrete_colorbar.values,
        values_short = discrete_colorbar.values_short;
    for (var i=0; i < colors.length; ++i) {
        html += '<div title="' + values[i].replace('"', '&quot;') + '"><span style="background:' + colors[i] + '"></span>&nbsp;' + values_short[i].replace('<', '&lt;') + '</div>';
    }
    html = '<div class="discrete-legend"><span class="title">' + discrete_colorbar.title + '</span><br>' + html + '</div>';
    document.getElementById('customjs-container').innerHTML = html;
}
