;

function __unpack_js_object(obj) {
    if (obj.name) {
        window[obj.name] = obj.obj;
        fixupPythonObject(window[obj.name]);
        console.log('received obj: ' + obj.name + ' ' + obj.id);
        __channel.objects.__js_object_channel.mark_exposed(obj.id);
    }
}


function __unpack_start_js_objects(objs) {
    for (var i=0; i < objs.length; ++i) {
        __unpack_js_object(objs[i])
    }
}


new QWebChannel(qt.webChannelTransport, function(channel) {
    window.__channel = channel;
    window.__self = channel.objects.__self;

    var timer = setInterval(function() {
        if (!window.__load_finished)
            return;
        clearInterval(timer);

        // If an object is already present on our channel, expose it
        __unpack_start_js_objects(__channel.objects.__js_object_channel.get_start_objects);

        // And subscribe to all further object exposing
        channel.objects.__js_object_channel.objectChanged.connect(
            __unpack_js_object);

        for (var variable in channel.objects) {
            // Skip internal UUID objects and our __js_object_channel
            // which was already handled above
            if (variable[0] == '{' || variable == '__js_object_channel')
                continue;

            // Assign all other objects to window directly
            window[variable] = channel.objects[variable];
        }

        try {
            __self._load_really_finished();
        } catch (err) {
            log.error('Not using WebviewWidget. No problem. Enjoy!')
        }
    }, 50);
});
