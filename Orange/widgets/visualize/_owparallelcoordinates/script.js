NodeList.prototype.forEach = Array.prototype.forEach;


function on_plotly_restyle() {
    console.log('restyle');
    pybridge.update_axes_info(container.data[0].dimensions.map(function(dim) {
        return [dim.label, dim.constraintrange || []]
    }));
}


function on_plotly_afterplot() {
    // Ellipsizes dimension labels
    // Can be removed once this is fixed:
    // https://github.com/plotly/plotly.js/issues/1703
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
