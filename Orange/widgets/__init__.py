"""

"""
import pkg_resources


# Entry point for main Orange categories/widgets discovery
def widget_discovery(discovery):
    #from . import data
    dist = pkg_resources.get_distribution("Orange")
    pkgs = ["Orange.widgets.data",
            "Orange.widgets.visualize",
            "Orange.widgets.classify",
            "Orange.widgets.regression",
            "Orange.widgets.evaluate",
            "Orange.widgets.unsupervised",
            "Orange.widgets.prototypes"]
    for pkg in pkgs:
        discovery.process_category_package(pkg, distribution=dist)
