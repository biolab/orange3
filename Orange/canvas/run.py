import os
import sys
import argparse
import logging

from AnyQt.QtCore import QEvent
from AnyQt.QtWidgets import QApplication

from orangecanvas import utils, config
from orangecanvas.registry import WidgetRegistry, cache
from orangecanvas.scheme.node import UserMessage
from orangecanvas.scheme import signalmanager

# Imported to make webwidget addons work
from Orange.widgets.utils.webview import WebviewWidget  # pylint: disable=unused-import


def main(argv=None):
    app = QApplication(list(argv) if argv else [])
    argv = app.arguments()

    parser = argparse.ArgumentParser(
        description=(
            "Run an orange workflow without showing a GUI and exit "
            "when it completes.\n\n"
            "WARNING: This is experimental as Orange is not designed to run "
            "non-interactive."
        )
    )
    parser.add_argument("--log-level", "-l", metavar="LEVEL", type=int,
                        default=logging.CRITICAL, dest="log_level")
    parser.add_argument("--config", default="Orange.canvas.config.Config",
                        type=str)
    parser.add_argument("file")
    args = parser.parse_args(argv[1:])

    log_level = args.log_level
    filename = args.file
    logging.basicConfig(level=log_level)

    cfg_class = utils.name_lookup(args.config)
    cfg: config.Config = cfg_class()
    config.set_default(cfg)
    config.init()
    reg = WidgetRegistry()
    widget_discovery = cfg.widget_discovery(
        reg, cached_descriptions=cache.registry_cache()
    )
    widget_discovery.run(cfg.widgets_entry_points())
    model = cfg.workflow_constructor()
    model.set_runtime_env(
        "basedir", os.path.abspath(os.path.dirname(filename))
    )
    sigprop = model.findChild(signalmanager.SignalManager)
    sigprop.pause()  # Pause signal propagation during load

    with open(filename, "rb") as f:
        model.load_from(f, registry=reg)

    # Ensure all widgets are created (this is required for the workflow
    # to even start - relies to much on OWWidget behaviour).
    for _ in map(model.widget_for_node, model.nodes):
        pass

    sigprop.resume()  # Resume inter-widget signal propagation

    def on_finished():
        severity = 0
        for node in model.nodes:
            for msg in node.state_messages():
                if msg.contents and msg.severity == msg.Error:
                    print(msg.contents, msg.message_id, file=sys.stderr)
                    severity = msg.Error
        if severity == UserMessage.Error:
            app.exit(1)
        else:
            app.exit()
    sigprop.finished.connect(on_finished)

    rval = app.exec_()
    model.clear()
    # Notify the workflow model to 'close'.
    QApplication.sendEvent(model, QEvent(QEvent.Close))
    app.processEvents()
    return rval


if __name__ == "__main__":
    sys.exit(main(sys.argv))
