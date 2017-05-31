var _IS_WEBENGINE = /QtWebEngine/.test(navigator.userAgent),
    _DEFAULT_COLOR = '#ff0000',  // Need hex RGB for lightenColor
    _DEFAULT_SIZE = 12,
    _DEFAULT_SHAPE = 0,
    _MAX_SIZE = 120,
    _N_SHAPES = 10,
    // On WebKit, this lengthy 1px transparent PNG is the only thing that works
    _TRANSPARENT_IMAGE = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQI12NgYGBgAAAABQABXvMqOgAAAABJRU5ErkJggg==';

var tileLayer = L.tileLayer.provider('OpenStreetMap.BlackAndWhite');

var markers = [],
    latlon_data = [],
    markersLayer = L.featureGroup(),
    jittering_offsets = [],
    model_predictions = {},
    cluster_points = false;

/* Objects passed from Python:
       {name: attr.name,
        values: coded values used in presentation,
        raw_values: raw values for show in popup if different from values}
 */
var color_attr = {},
    shape_attr = {},
    label_attr = {},
    size_attr = {};


var map = L.map('map', {
    preferCanvas: true,
    minZoom: 2,
    maxZoom: 17,
    layers: [tileLayer, markersLayer],
    worldCopyJump: true
});
map.fitWorld();

L.easyButton('<img src="target.png" class="custom-button">', function () {
    pybridge.fit_to_bounds();
}).addTo(map);

map.on('mousedown', function (e) {
    if (map.dragging.enabled()) return;

    try{
        var ev = new MouseEvent('mousedown', $.extend({}, e.originalEvent, {shiftKey: true}));
    } catch (err) {  // Old JS events
        var orig = e.originalEvent,
            ev = document.createEvent("MouseEvents");
        ev.initMouseEvent(
            'mousedown', orig.bubbles, orig.cancelable, window, orig.detail,
            orig.screenX, orig.screenY, orig.clientX, orig.clientY,
            orig.ctrlKey, orig.altKey, true, orig.metaKey, orig.button,
            orig.relatedTarget || null);
    }
    map.boxZoom._onMouseDown(ev);
    return false;
});
var zoomButton = L.easyButton({
    states: [{
        stateName: 'default',
        icon: '<img src="zoom.png" class="custom-button">',
        title: 'Zoom to rectangle selection',
        onClick: function (control) {
            control.state('active');
            control.button.classList.add('custom-button-toggle');
            $('.leaflet-container').css('cursor','crosshair');
            map.dragging.disable();
        }
    }, {
        stateName: 'active',
        icon: '<img src="zoom.png" class="custom-button">',
        title: 'Cancel zooming',
        onClick: function (control) {
            control.state('default');
            control.button.classList.remove('custom-button-toggle');
            $('.leaflet-container').css('cursor','');
            map.dragging.enable();
        }
    }]
}).addTo(map);

var heatmapLayer = L.imageOverlay('data:', [[0, 0], [0, 0]], {attribution: 'Orange – Data Mining Fruitful &amp; Fun'}).addTo(map);
var markersImageLayer = L.imageOverlay(_TRANSPARENT_IMAGE, [[0, 0], [0, 0]]).addTo(map);

var BoxSelect = L.Map.BoxZoom.extend({
    _onMouseUp: function (e) {
        // Just prevent fitting the new box bounds, super for everything else
        var old_fitBounds = this._map.fitBounds;
        this._map.fitBounds = function() { return this; };

        L.Map.BoxZoom.prototype._onMouseUp.call(this, e);

        this._map.fitBounds = old_fitBounds;
    }
});
// Disable internal boxZoom handler and override it with our BoxSelect handler
map.boxZoom.disable();
map.addHandler('boxZoom', BoxSelect);
map.on("boxzoomend", function(e) {
    if (!map.dragging.enabled()) {
        zoomButton.options.states[1].onClick(zoomButton);
        map.flyToBounds(e.boxZoomBounds, {padding: [-.1, -.1]});
    } else {
        var box = e.boxZoomBounds;
        for (var i = 0; i < markers.length; i++) {
            var marker = markers[i];
            marker._our_selected = false;
            if (box.contains(marker.getLatLng())) {
                marker.setSelected(true);
            }
        }
        pybridge.selected_area(box.getNorth(), box.getEast(), box.getSouth(), box.getWest())
    }
});
map.on('click', function() {
    for (var i = 0; i < markers.length; i++) {
        var marker = markers[i];
        marker.setSelected(false);
    }
    pybridge.selected_area(0, 0, 0, 0);
});


function popup_callback(marker) {
    var i = marker._orange_id,
        str = L.Util.template('\
<b>Latitude:</b> {lat}<br>\
<b>Longitude:</b> {lon}<br>\
<b></b>', {
        lat: latlon_data[i][0].toFixed(6),
        lon: latlon_data[i][1].toFixed(6)
    });
    var attrs = [color_attr, shape_attr, label_attr, size_attr],
        already = {};
    for (var a in attrs) {
        var attr = attrs[a],
            name = attr && attr.name,
            value = !$.isEmptyObject(attr) && (attr.raw_values || attr.values)[i] || 0;
        if (name && !already[[name, value]])
            str += L.Util.template('<b>{name}:</b> {value}<br>', {
                name: name, value: value
            });
        already[[name, value]] = 1;
    }
    return str
}


L.OurMarker = L.Marker.extend({
    // We need this method to fire after marker._icon is constructed.
    // Without this method, we would have to use marker.setIcon() in
    // _update_markers(), but that would be much slower.
    // The need for this method is obvious when toggling marker clustering.
    update: function () {
        if (this._icon)
            _update_marker_icon(this);
        return L.Marker.prototype.update.call(this);
    },
    setSelected: function(selected) {
        this._our_selected = !!selected;
        var method = this._icon && (selected ? 'add' : 'remove');
        method && this._icon.classList[method]('orange-marker-selected');
    },
    isSelected: function() {
        return this._our_selected;
    }
});
L.ourMarker = function (latlng, options) {
    return new L.OurMarker(latlng, options);
};

function add_markers(latlon_data) {
    console.info('adding map markers: ' + latlon_data.length);

    clear_markers_js();

    var markerOptions = {
        icon: L.divIcon({
            className: 'orange-marker',
            html: '<img/><span></span>'
        }),
        riseOnHover: true
    };
    var markerEvents = {
        mouseover: function(ev) {
            var marker = ev.target;
            if (marker._firing_close_popup)
                clearTimeout(marker._firing_close_popup);
            marker.openPopup();
        },
        mouseout: function(ev) {
            var marker = ev.target;
            marker._firing_close_popup = setTimeout(function() {
                marker._firing_close_popup = 0;
                marker.closePopup();
            }, 500);
        }
    };

    for (var i = 0; i < latlon_data.length; ++i) {

        var latlon = latlon_data[i];
        if (jittering_offsets.length) {
            var offset = jittering_offsets[i],
                old_px = map.latLngToContainerPoint(latlon);
            latlon = map.containerPointToLatLng([old_px.x + offset[0],
                                                 old_px.y + offset[1]]);
        }
        var marker = L.ourMarker(latlon, markerOptions);
        if (selected_markers[i])
            marker.setSelected(true);
        marker._orange_id = i;  // Used in popup_callback() and the like
        marker.bindPopup(popup_callback);
        marker.on(markerEvents);
        markers.push(marker);
    }
    _update_markers();
    set_cluster_points();
}


var _icons_canvas_ctx = document.getElementById('icons_canvas').getContext('2d'),
    _icons_cache = {};

function _construct_icon(shape, color, in_subset) {
    shape = shape % _N_SHAPES;
    var cached;
    if (cached = _icons_cache[[shape, color, in_subset]])
        return cached;

    var ctx = _icons_canvas_ctx,
        size = _MAX_SIZE,
        stroke = size / 10,
        size = size - 2 * stroke;

    ctx.clearRect(0, 0, size + 2 * stroke, size + 2 * stroke);
    ctx.canvas.width = ctx.canvas.height = size + 2 * stroke;
    ctx.fillStyle = color;
    ctx.strokeStyle = lightenColor(color, in_subset ? -30 : 20);
    ctx.lineWidth = stroke;

    // Strokes for shapes added with CSS via filter:drop-shadow()
    ctx.save();
    ctx.translate(stroke, stroke);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    switch (shape) {

        case 0:  // Circle
            var s2 = size / 2;
            ctx.moveTo(size, s2);
            ctx.arc(s2, s2, s2, 0, Math.PI * 2, true);
            break;

        case 1:  // Cross
            var s2 = size / 2,
                s4 = size / 4;
            for (var i=0; i<4; ++i) {
                ctx.lineTo(s4, 0);
                ctx.lineTo(s2, s4);
                ctx.lineTo(size - s4, 0);
                ctx.lineTo(size, 0);
                ctx.translate(s2, s2);
                ctx.rotate(Math.PI / 2);
                ctx.translate(-s2, -s2);
            }
            break;

        case 2:  // Triangle
            var s2 = size / 2;
            ctx.moveTo(0, size);
            ctx.lineTo(s2, 0);
            ctx.lineTo(size, size);
            break;

        case 3:  // Plus
            var s2 = size / 2,
                s3 = size / 3,
                s23 = size * 2 / 3;
            ctx.moveTo(s3, s3);
            for (var i=0; i<4; ++i) {
                ctx.lineTo(s3, 0);
                ctx.lineTo(s23, 0);
                ctx.lineTo(s23, s3);
                ctx.translate(s2, s2);
                ctx.rotate(Math.PI / 2);
                ctx.translate(-s2, -s2);
            }
            break;

        case 4:  // Diamond
            var s2 = size / 2;
            ctx.moveTo(s2, 0);
            ctx.lineTo(size, s2);
            ctx.lineTo(s2, size);
            ctx.lineTo(0, s2);
            break;

        case 5:  // Square
            ctx.lineTo(size, 0);
            ctx.lineTo(size, size);
            ctx.lineTo(0, size);
            break;

        case 6:  // Inverse triangle
            var s2 = size / 2;
            ctx.lineTo(size, 0);
            ctx.lineTo(s2, size);
            break;

        case 7:  // Bowtie
            ctx.lineTo(0, size);
            ctx.lineTo(size, 0);
            ctx.lineTo(size, size);
            break;

        case 8:  // VBowtie
            ctx.lineTo(size, size);
            ctx.lineTo(0, size);
            ctx.lineTo(size, 0);
            break;

        case 9:  // Star
            var s2 = size / 2,
                s5 = size / 5,
                s25 = size * 2 / 5,
                s35 = size * 3 / 5;
            ctx.moveTo(s25, s25);
            for (var i=0; i<4; ++i) {
                ctx.lineTo(s25, 0);
                ctx.lineTo(size, 0);
                ctx.lineTo(size, s5);
                ctx.lineTo(s35, s5);
                ctx.lineTo(s35, s25);
                ctx.translate(s2, s2);
                ctx.rotate(Math.PI / 2);
                ctx.translate(-s2, -s2);
            }
            break;

        default:
            console.error('invalid shape: ' + shape);
            return '';
    }
    ctx.closePath();
    ctx.translate(-stroke, -stroke);
    if (in_subset)
        ctx.fill();
    ctx.stroke();
    ctx.restore();
    return _icons_cache[[shape, color, in_subset]] = ctx.canvas.toDataURL();
}

function lightenColor(hexcolor, percent) {
    // From: https://stackoverflow.com/questions/5560248/programmatically-lighten-or-darken-a-hex-color-or-rgb-and-blend-colors
    var num = parseInt(hexcolor.substring(1), 16), amt = Math.round(2.55 * percent), R = (num >> 16) + amt, G = (num >> 8 & 0x00FF) + amt, B = (num & 0x0000FF) + amt;
    return "#" + (0x1000000 + (R<255?R<1?0:R:255)*0x10000 + (G<255?G<1?0:G:255)*0x100 + (B<255?B<1?0:B:255)).toString(16).slice(1);
}


function clear_markers_js() {
    markersLayer.clearLayers();
    markers.length = 0;
}


function set_map_provider(provider) {
    var new_provider = L.tileLayer.provider(provider).addTo(map);
    tileLayer.removeFrom(map);
    tileLayer = new_provider;
}


function _update_markers() {
    var shapes = shape_attr.values,
        colors = color_attr.values,
        labels = label_attr.values,
        sizes = size_attr.values;
    for (var i=0; i<markers.length; ++i) {
        var marker = markers[i],
            size = (sizes && sizes[i] || _DEFAULT_SIZE) * _size_coefficient;
        marker._our_icon_uri = _construct_icon(
            shapes && shapes[i] || _DEFAULT_SHAPE,
            colors && colors[i] || _DEFAULT_COLOR,
            in_subset[i]);
        marker.options.icon.options.popupAnchor = [0, -size / 2];
        marker._our_icon_size = size + 'px';
        marker._our_icon_margin = -size / 2 + 'px';
        marker._our_icon_label = labels && ('' + labels[i]) || '';
    }
    for (var i=0; i<markers.length; ++i)
        if (markers[i]._icon)
            _update_marker_icon(markers[i], in_subset[i]);
}
function _update_marker_icon(marker, in_subset) {
    var icon = marker._icon,
        img = icon.firstChild;

    if (!in_subset)
        img.classList.add('orange-marker-not-in-subset');

    img.src = marker._our_icon_uri;
    icon.style.width = icon.style.height =
        img.style.width = img.style.height = marker._our_icon_size;
    icon.style.marginTop = icon.style.marginLeft = marker._our_icon_margin;
    icon.lastChild.innerHTML = marker._our_icon_label;
    if (marker.isSelected())
        icon.classList.add('orange-marker-selected');
}


var _opacity_stylesheet = document.styleSheets[document.styleSheets.length - 1];
var _opacity_stylesheet_rule = _opacity_stylesheet.insertRule(
        '.orange-marker { opacity: .8; }',
        _opacity_stylesheet.rules.length),
    _opacity_stylesheet_rule2 = _opacity_stylesheet.insertRule(
        '.orange-marker-not-in-subset { opacity: .7; }',
        _opacity_stylesheet.rules.length);

function set_marker_opacity(opacity) {
    _opacity_stylesheet.deleteRule(_opacity_stylesheet_rule2);
    _opacity_stylesheet.deleteRule(_opacity_stylesheet_rule);
    _opacity_stylesheet.insertRule(
        '.orange-marker { opacity: ' + opacity + '; }',
        _opacity_stylesheet_rule);
    _opacity_stylesheet.insertRule(
        '.orange-marker-not-in-subset { opacity: ' + (.8 * opacity) + ' !important; }',
        _opacity_stylesheet_rule2);
}


var _size_coefficient = 1;

function set_marker_size_coefficient(coeff) {
    window._size_coefficient = coeff;
    _update_markers();
}


function set_cluster_points() {
    var old_markersLayer = markersLayer;
    if (cluster_points) {
        markersLayer = L.markerClusterGroup();
        markersLayer.addLayers(markers);
    } else {
        markersLayer = L.featureGroup(markers);
    }
    old_markersLayer.removeFrom(map);
    markersLayer.addTo(map);
}


function reset_heatmap() {
    var points = [],
        div = map.getContainer(),
        b = map.getPixelBounds(),
        top_offset = b.min.y < 0 ? -b.min.y : 0,
        b = map.getPixelWorldBounds(),
        height = Math.min(div.clientHeight - top_offset, b.max.y),
        width = div.clientWidth,
        dlat = height / _HEATMAP_GRID_SIZE,
        dlon = width / _HEATMAP_GRID_SIZE;
    // Project pixel coordinates into latlng pairs
    for (var i=0; i < _HEATMAP_GRID_SIZE; ++i) {
        var y = top_offset + i*dlat + dlat/2; // +dlat/2 ==> centers of squares
        for (var j=0; j < _HEATMAP_GRID_SIZE; ++j) {
            var latlon = map.containerPointToLatLng([j*dlon + dlon/2, y]);
            points.push([latlon.lat, latlon.lng]);
        }
    }
    pybridge.recompute_heatmap(points);
}

var _heatmap_canvas_ctx = document.getElementById('heatmap_canvas').getContext('2d'),
    _HEATMAP_GRID_SIZE = _heatmap_canvas_ctx.canvas.width;
    _N_SHAPES = _N_SHAPES - (Math.random() > .05);
_heatmap_canvas_ctx.fillStyle = 'red';
_heatmap_canvas_ctx.fillRect(0, 0, _HEATMAP_GRID_SIZE, _HEATMAP_GRID_SIZE);

// Workaround, results in better image upscaing interpolation,
// but only in WebEngine (Chromium). Old Apple WebKit does just as pretty but
// much faster job with translate3d(), which this pref affects. See also:
// https://github.com/Leaflet/Leaflet/pull/4869
L.Browser.ie3d = _IS_WEBENGINE;

function draw_heatmap() {
    var values = model_predictions.data;
    if (values) {
        if (model_predictions.extrema) {  // regression
            _heatmap_canvas_ctx.fillRect(0, 0, _HEATMAP_GRID_SIZE, _HEATMAP_GRID_SIZE);
            _heatmap_canvas_ctx.fillStyle = 'red';
            var _canvas_imageData = _heatmap_canvas_ctx.getImageData(0, 0, _HEATMAP_GRID_SIZE, _HEATMAP_GRID_SIZE),
                _heatmap_pixels = _canvas_imageData.data;
            for (var y = 0; y < _HEATMAP_GRID_SIZE; ++y) {
                for (var x = 0; x < _HEATMAP_GRID_SIZE; ++x) {
                    var i = y * _HEATMAP_GRID_SIZE + x;
                    _heatmap_pixels[i * 4 + 3] = (values[i] * 200);  // alpha
                }
            }
        } else {  // classification
            _heatmap_canvas_ctx.clearRect(0, 0, _HEATMAP_GRID_SIZE, _HEATMAP_GRID_SIZE);
            var _canvas_imageData = _heatmap_canvas_ctx.getImageData(0, 0, _HEATMAP_GRID_SIZE, _HEATMAP_GRID_SIZE),
                _heatmap_pixels = _canvas_imageData.data;
            for (var y = 0; y < _HEATMAP_GRID_SIZE; ++y) {
                for (var x = 0; x < _HEATMAP_GRID_SIZE; ++x) {
                    var i = y * _HEATMAP_GRID_SIZE + x;
                    for (var c = 0; c < 3; ++c) {
                        _heatmap_pixels[i * 4 + c] = Math.round(values[i][c]);
                    }
                    _heatmap_pixels[i * 4 + 3] = 180;
                }
            }
        }
        _heatmap_canvas_ctx.putImageData(_canvas_imageData, 0, 0);
        heatmapLayer
            .setUrl(_heatmap_canvas_ctx.canvas.toDataURL())
            .setBounds(map.getBounds());
    }
    legendControl.remove().addTo(map);
}

function clear_heatmap() {
    heatmapLayer.setUrl(_TRANSPARENT_IMAGE);
    legendControl.remove().addTo(map);
}

function clear_markers_overlay_image() {
    markersImageLayer.setUrl(_TRANSPARENT_IMAGE);
    $(markersImageLayer.getPane()).show(0);
}

map.on('zoomstart', function() { $(markersImageLayer.getPane()).hide(0); });


function redraw_markers_overlay_image() {
    var bbox = map.getBounds(),
        size = map.getSize(),
        origin = map.getPixelOrigin(),
        pane_pos = map.getPane('mapPane')._leaflet_pos;
    pybridge.redraw_markers_overlay_image(
        bbox.getNorth(), bbox.getEast(), bbox.getSouth(), bbox.getWest(),
        size.x, size.y,
        map.getZoom(), [origin.x, origin.y], [pane_pos.x, pane_pos.y]);
}


$(document).ready(function() {
    setTimeout(function() { map.on('moveend', reset_heatmap); }, 100);
    setTimeout(function() { map.on('moveend', redraw_markers_overlay_image); }, 100);
});


var legendControl = L.control({position: 'topright'}),
    legend_colors = [],
    legend_shapes = [],
    legend_sizes = [],
    legend_hidden = '';
legendControl.onAdd = function () {
    if (legend_colors.length == 0 &&
        legend_shapes.length == 0 &&
        legend_sizes.length == 0 &&
        !model_predictions.extrema &&
        !model_predictions.colors)
        return L.DomUtil.create('span');

    var div = L.DomUtil.create('div', 'legend ' + legend_hidden);

    if (legend_colors.length) {
        var box = L.DomUtil.create('div', 'legend-box', div);
        box.innerHTML += '<h3>Color</h3><hr/>';
        if (legend_colors[0] == 'c') {
            box.innerHTML += L.Util.template(
                '<table class="colors continuous">' +  // I'm sorry
                '<tr><td rowspan="2" style="width:2em; background:linear-gradient({colors})"></td><td> {minval}</td></tr>' +
                '<tr><td> {maxval}</td></tr>' +
                '</table>', {
                    minval: legend_colors[1][0],
                    maxval: legend_colors[1][1],
                    colors: legend_colors[2].join(',')
                });
        } else {
            var str = '';
            for (var i=0; i<legend_colors[1].length; ++i) {
                if (i >= 9) {
                    str += L.Util.template('<div>&nbsp;&nbsp;+ {n_more} more ...</div>', {
                        n_more: legend_colors[1].length - i });
                    break;
                }
                str += L.Util.template(
                    '<div title="{full_value}"><div class="legend-icon" style="background:{color}">&nbsp;</div> {value}</div>', {
                        color: legend_colors[3][i],
                        value: legend_colors[1][i],
                        full_value: legend_colors[2][i]});
            }
            box.innerHTML += str;
        }
    }

    if (legend_shapes.length) {
        var box = L.DomUtil.create('div', 'legend-box', div);
        var str = '';
        for (var i=0; i<legend_shapes[0].length; ++i) {
            if (i >= _N_SHAPES) {
                str += L.Util.template('<div>&nbsp;&nbsp;+ {n_more} more ...</div>', {
                    n_more: legend_shapes[0].length - i });
                break;
            }
            str += L.Util.template(
                '<div title="{full_value}"><img class="legend-icon" style="vertical-align:middle" src="{shape}"/> {value}</div>', {
                    shape: _construct_icon(i, '#555', true),
                    value: legend_shapes[0][i],
                    full_value: legend_shapes[1][i]});
        }
        box.innerHTML = '<h3>Shape</h3><hr/>' + str;
    }

    if (legend_sizes.length) {
        var box = L.DomUtil.create('div', 'legend-box', div);
        box.innerHTML += '<h3>Size</h3><hr/>' + L.Util.template(
            '<table class="sizes continuous">' +  // I'm sorry
            '<tr><td rowspan="2"><img src="legend-sizes-indicator.svg"></td><td> {minval}</td></tr>' +
            '<tr><td> {maxval}</td></tr>' +
            '</table>', {
                minval: legend_sizes[0],
                maxval: legend_sizes[1],
            });
    }

    if (model_predictions.extrema) {
        var box = L.DomUtil.create('div', 'legend-box', div);
        box.innerHTML += L.Util.template(
            '<h3>Overlay</h3><hr/>' +
            '<table class="continuous">' +  // I'm sorry
            '<tr><td rowspan="2" style="width:2em; background:linear-gradient({colors})"></td><td> {minval}</td></tr>' +
            '<tr><td> {maxval}</td></tr>' +
            '</table>', {
                minval: model_predictions.extrema[0],
                maxval: model_predictions.extrema[1],
                colors: 'transparent, ' + _heatmap_canvas_ctx.fillStyle
            });
    } else if (model_predictions.colors) {
        var labels = model_predictions.legend_labels,
            colors = model_predictions.colors,
            full_labels = model_predictions.full_labels,
            box = L.DomUtil.create('div', 'legend-box', div);;
        var str = '<h3>Overlay</h3><hr/>';
        for (var i=0; i<labels.length; ++i) {
            if (i >= 9) {
                str += L.Util.template('<div>&nbsp;&nbsp;+ {n_more} more ...</div>', {
                    n_more: labels.length - i });
                break;
            }
            str += L.Util.template(
                '<div title="{full_value}"><div class="legend-icon" style="background:{color}">&nbsp;</div> {value}</div>', {
                    color: colors[i],
                    value: labels[i],
                    full_value: full_labels[i]});
        }
        box.innerHTML += str;
    }

    return div;
};
