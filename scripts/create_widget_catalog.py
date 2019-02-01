from copy import copy
import json
from os import path, makedirs

from AnyQt.QtGui import QImage, QPainter
from AnyQt.QtWidgets import QGraphicsScene, QApplication, QWidget, QGraphicsView, QHBoxLayout
from AnyQt.QtCore import QRectF, Qt, QTimer

from Orange.canvas import config
from Orange.canvas.canvas.items import NodeItem
from Orange.canvas.help import HelpManager
from Orange.canvas.registry.qt import QtWidgetDiscovery, QtWidgetRegistry


class WidgetCatalog:
    def __init__(self, output_dir, image_url_prefix):
        self.output_dir = output_dir
        self.image_url_prefix = image_url_prefix

        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
        QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
        self.app = QApplication([])

        print("Generating widget repository")
        self.registry = self.__get_widget_registry()
        print("Locating help files")
        self.help_manager = HelpManager()
        self.help_manager.set_registry(self.registry)

        # Help manager needs QApplication running to get the urls
        # of widget documentation. 5 seconds should be enough.
        QTimer.singleShot(5000, self.app.quit)
        self.app.exec()

        self.__scene = QGraphicsScene()
        self.__nodes = []
        print("Ready to go")

    def create(self):
        print("Generating catalog")
        try:
            makedirs(path.join(self.output_dir, "icons"))
        except FileExistsError:
            pass

        result = []
        for category in self.registry.categories():
            widgets = []
            result.append((category.name, widgets))
            for widget in category.widgets:
                widgets.append({
                    "text": widget.name,
                    "doc": self.__get_help(widget),
                    "img": self.__get_icon(widget, category),
                    "keyword": widget.keywords,
                })

        with open(path.join(self.output_dir, "widgets.json"), 'wt') as f:
            json.dump(result, f, indent=1)
        print("Done")

    @staticmethod
    def __get_widget_registry():
        widget_discovery = QtWidgetDiscovery()
        widget_registry = QtWidgetRegistry()
        widget_discovery.found_category.connect(
            widget_registry.register_category
        )
        widget_discovery.found_widget.connect(
            widget_registry.register_widget
        )
        widget_discovery.run(config.widgets_entry_points())

        # Fixup category.widgets list
        for cat, widgets in widget_registry._categories_dict.values():
            cat.widgets = widgets

        return widget_registry

    def __get_icon(self, widget, category=None):
        # Set remove inputs/outputs so the "ears" are not drawn
        widget = copy(widget)
        widget.inputs = []
        widget.outputs = []

        w = IconWidget()
        w.set_widget(widget, category)
        w.show()
        #self.app.processEvents()
        filename = "icons/{}.png".format(widget.qualified_name)
        w.render_as_png(path.join(self.output_dir, filename))
        w.hide()

        if self.image_url_prefix:
            return self.image_url_prefix + filename
        else:
            return filename

    def __get_help(self, widget):
        query = dict(id=widget.qualified_name)
        try:
            return self.help_manager.search(query).url()
        except KeyError:
            return None


class IconWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setFixedSize(50, 50)

        view = QGraphicsView()
        self.layout().addWidget(view)
        self.scene = QGraphicsScene(view)
        view.setScene(self.scene)

    def set_widget(self, widget_description, category_description):
        node = NodeItem(widget_description)
        if category_description is not None:
            node.setWidgetCategory(category_description)
        self.scene.addItem(node)

    def render_as_png(self, filename):
        png = self.__transparent_png()
        painter = QPainter(png)
        painter.setRenderHint(QPainter.Antialiasing, 1)
        self.scene.render(painter, QRectF(0, 0, 50, 50), QRectF(-25, -25, 50, 50))
        painter.end()
        png.save(filename)

    def __transparent_png(self):
        # PyQt is stupid and does not increment reference count on bg
        w = h = 50
        self.__bg = bg = b"\xff\xff\x00" * w * h * 4
        return QImage(bg, w, h, QImage.Format_ARGB32)


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser(usage="usage: %prog --output <outputdir> [options]")
    parser.add_option('--url-prefix', dest="prefix",
                      help="prefix to prepend to image urls")
    parser.add_option('--output', dest="output",
                      help="path where widgets.json will be created")

    options, args = parser.parse_args()
    if not options.output:
        parser.error("Please specify the output dir")

    w = WidgetCatalog(output_dir=options.output, image_url_prefix=options.prefix)
    w.create()
