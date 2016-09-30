// Polyfill for Array.fill
// Note: only full filling supported
if (!Array.prototype.fill) {
    Array.prototype.fill = function(value) {
        for (var i=0; i<this.length; ++i)
            this[i] = value;
        return this;
    }
}


var _DEFAULT_COLOR = 'red',
    _DEFAULT_SIZE = 30,
    _DEFAULT_SHAPE = 0;

var tileLayer = L.tileLayer.provider('OpenStreetMap.BlackAndWhite');

var markers = [],
    markersLayer = L.featureGroup(),
    jittering_offsets = [],
    cluster_points = false;

var color_attr = {},
    shape_attr = {},
    label_attr = {},
    size_attr = {};

var _SHAPES = [
    'orange-marker-circle',
    'orange-marker-cross',
    'orange-marker-triangle',
    'orange-marker-plus',
    'orange-marker-diamond',
    'orange-marker-square',
    'orange-marker-downtriangle',
    'orange-marker-star',
    'orange-marker-shogi',
    'orange-marker-heart',
    'orange-marker-bowtie',
    'orange-marker-sweet'
];

var orangeMarkerIcon = L.divIcon({
    className: 'orange-marker orange-marker-circle',
    html:'<span></span>'  // label span
});

var map = L.map('map', {
    preferCanvas: true,
    center: [51.505, -0.09],
    minZoom: 2,
    layers: [tileLayer, markersLayer],
    worldCopyJump: true,
    // Disable animation. Works better when lots of markers (over 9000)
    // This can't be set in add_markers() with L.setOptions() as by then the
    // event handlers are already installed
    zoomAnimation: false,
    // With disabled animation, the default wheel zooms seems too responsive
    wheelPxPerZoomLevel: 500,
    wheelDebounceTime: 200
});
map.fitWorld();
map.on('zoom', reposition_markers);

var heatmapLayer = L.imageOverlay('data:', [[0, 0], [0, 0]], {attribution: 'Orange – Data Mining Fruitful &amp; Fun'}).addTo(map);

var selected_markers = {};
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
map['boxZoom'].disable();
map.addHandler('boxZoom', BoxSelect);
map.on("boxzoomend", function(e) {
    for (var i = 0; i < markers.length; i++) {
        var marker = markers[i];
        if (e.boxZoomBounds.contains(marker.getLatLng())) {
            marker._icon.classList.add('orange-marker-selected');
            selected_markers[marker._orange_id] = 1;
        }
    }
    __self._selected_indices(Object.keys(selected_markers));
});
map.on('click', function() {
    for (var i = 0; i < markers.length; i++) {
        markers[i]._icon.classList.remove('orange-marker-selected');
    }
    __self._selected_indices([]);
});


function popup_callback(marker) {
    var i = marker._orange_id,
        str = L.Util.template('\
<b>Latitude:</b> {lat}<br>\
<b>Longitude:</b> {lon}<br>\
<b></b>', {
        lat: latlon_data[i][0],
        lon: latlon_data[i][1]
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

function add_markers(latlon_data) {
    console.info('adding map markers: ' + latlon_data.length);

    clear_markers();

    var markerOptions = {
        icon: orangeMarkerIcon,
        riseOnHover: true,
        html: '<span></span>'  // This firstChild is for labels
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
        var marker = L.marker(latlon_data[i], markerOptions);
        marker._orange_id = i;  // Used in popup_callback() and the like
        marker.bindPopup(popup_callback);
        marker.on(markerEvents);

        markers.push(marker);
    }
    set_cluster_points();
    set_jittering();
    set_marker_sizes();
    map.fitBounds(markersLayer.getBounds().pad(.1));
}

function clear_markers() {
    markersLayer.clearLayers();
    markers.length = 0;
}

function reposition_markers() {
    if (!markers.length)
        return;
    if (jittering_offsets.length) {
        if (markers.length != jittering_offsets.length || markers.length != latlon_data.length)
            return console.error('markers.length != jittering_offsets.length || markers.length != latlon_data.length ???');
        var data = latlon_data,
            div = map.getContainer(),
            w = div.clientWidth,
            h = div.clientHeight;
        for (var i = 0; i < markers.length; ++i) {
            var offset = jittering_offsets[i],
                old_px = map.latLngToContainerPoint(data[i]),
                new_pt = map.containerPointToLatLng([old_px.x + h * offset[0],
                                                     old_px.y + w * offset[1]]);
            markers[i].setLatLng(new_pt);
        }
    }
}

var jittering_percent = 0;

function set_jittering() {
    percent = jittering_percent / 100;
    jittering_offsets.length = 0;
    if (percent == 0)
        return;
    for (var i = 0; i < latlon_data.length; ++i) {
        jittering_offsets.push([(Math.random() - .5) * percent,
                                (Math.random() - .5) * percent]);
    }
    reposition_markers();
}

function clear_jittering() {
    for (var i = 0; i < markers.length; ++i) {
        markers[i].setLatLng(latlon_data[i]);
    }
}


function set_map_provider(provider) {
    var new_provider = L.tileLayer.provider(provider).addTo(map);
    tileLayer.removeFrom(map);
    tileLayer = new_provider;
}


function set_marker_shapes(shape_indices) {
    if (!shape_indices.length)
        shape_indices = Array(markers.length).fill(_DEFAULT_SHAPE);
    if (markers.length != shape_indices.length)
        return console.error('markers.length != shape_indices.length ???');
    for (var i = 0; i < shape_indices.length; ++i) {
        var classList = markers[i]._icon.classList;
        classList.remove.apply(classList, _SHAPES);
        var ind = shape_indices[i];
        if (ind >= _SHAPES.length)
            ind = _SHAPES.length - 1;
        classList.add(_SHAPES[ind]);
    }
}


function set_marker_colors(css_colors) {
    if (!css_colors.length)
        css_colors = Array(markers.length).fill(_DEFAULT_COLOR);
    if (markers.length != css_colors.length)
        return console.error('markers.length != hex_colors.length ???');
    for (var i = 0; i < css_colors.length; ++i) {
        markers[i]._icon.style.color = css_colors[i];
    }
}


function set_marker_sizes(font_sizes) {
    if (!font_sizes || !font_sizes.length)
        font_sizes = Array(markers.length).fill(_DEFAULT_SIZE);
    if (markers.length != font_sizes.length)
        return console.error('markers.length != hex_colors.length ???');

    // Markers need to be display=block for getComputedStyle().height to work.
    // Yet they have to be display=inline for marker label span to not be
    // pushed on the next line. So the rule that is inserted here, is reverted
    // at the end.
    var stylesheet = document.styleSheets[document.styleSheets.length - 1],
        stylesheet_rule = stylesheet.insertRule(
            '.orange-marker:before { display: inline-block; }',
            stylesheet.rules.length);

    for (var i = 0; i < font_sizes.length; ++i) {
        var marker = markers[i];
        marker._icon.style.fontSize = font_sizes[i] + 'px';
        var computed = window.getComputedStyle(marker._icon, ':before'),
            w = parseFloat(computed.width), h = parseFloat(computed.height),
            opts = marker.options.icon.options;
        // Offset the center of the marker and popup anchor
        opts.iconAnchor = L.point(w / 2, h / 2);
        opts.popupAnchor = L.point(0, -h / 2);
        marker.setIcon(marker.options.icon);
    }

    stylesheet.deleteRule(stylesheet_rule);
}


function set_marker_labels(labels) {
    if (!labels.length)
        labels = Array(markers.length).fill('');
    if (markers.length != labels.length)
        return console.error('markers.length != hex_colors.length ???');
    for (var i = 0; i < labels.length; ++i) {
        markers[i]._icon.firstChild.innerHTML = labels[i];
    }
}


var opacity_stylesheet = document.styleSheets[document.styleSheets.length - 1];
var opacity_stylesheet_rule = opacity_stylesheet.insertRule(
    '.orange-marker { opacity: .8; }',
    opacity_stylesheet.rules.length);

function set_marker_opacity(opacity) {
    opacity_stylesheet.deleteRule(opacity_stylesheet_rule);
    opacity_stylesheet.insertRule(
        '.orange-marker { opacity: ' + opacity + '; }',
        opacity_stylesheet_rule);
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
        dlat = height / HEATMAP_GRID_SIZE,
        dlon = width / HEATMAP_GRID_SIZE;
    // Project pixel coordinates into latlng pairs
    for (var i=0; i < HEATMAP_GRID_SIZE; ++i) {
        var y = top_offset + i*dlat + dlat/2; // +dlat/2 ==> centers of squares
        for (var j=0; j < HEATMAP_GRID_SIZE; ++j) {
            var latlon = map.containerPointToLatLng([j*dlon + dlon/2, y]);
            points.push([latlon.lat, latlon.lng]);
        }
    }
    __self.latlon_viewport_extremes(points);
}

var canvas = document.getElementById('heatmap_canvas'),
    HEATMAP_GRID_SIZE = canvas.width,
    canvas_ctx = canvas.getContext('2d');
canvas_ctx.fillStyle = 'red';
canvas_ctx.fillRect(0, 0, HEATMAP_GRID_SIZE, HEATMAP_GRID_SIZE);
var canvas_imageData = canvas_ctx.getImageData(0, 0, HEATMAP_GRID_SIZE, HEATMAP_GRID_SIZE),
    heatmap_pixels = canvas_imageData.data;

// Workaround, results in better image upscaing interpolation,
// but only in WebEngine (Chromium). Old Apple WebKit does just as pretty but
// much faster job with translate3d(), which this pref affects. See also:
// https://github.com/Leaflet/Leaflet/pull/4869
L.Browser.ie3d = /QtWebEngine/.test(navigator.userAgent);

function draw_heatmap() {
    var values = model_predictions.data;
    for (var y = 0; y < HEATMAP_GRID_SIZE; ++y) {
        for (var x = 0; x < HEATMAP_GRID_SIZE; ++x) {
            var i = y * HEATMAP_GRID_SIZE + x;
            heatmap_pixels[i * 4 + 3] = (values[i] * 200);  // alpha
        }
    }
    canvas_ctx.putImageData(canvas_imageData, 0, 0);
    heatmapLayer
        .setUrl(canvas.toDataURL())
        .setBounds(map.getBounds());
}

function clear_heatmap() {
    // On WebKit, this lengthy 1px transparent PNG is the only thing that works
    heatmapLayer.setUrl('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQI12NgYGBgAAAABQABXvMqOgAAAABJRU5ErkJggg==');
}


$(document).ready(function() {
    setTimeout(function() { map.on('moveend', reset_heatmap); }, 100);
});
