"""

"""
import os
import sysconfig

import pkg_resources

import Orange


# Entry point for main Orange categories/widgets discovery
def widget_discovery(discovery):
    dist = pkg_resources.get_distribution("Orange")
    pkgs = [
        "Orange.widgets.data",
        "Orange.widgets.visualize",
        "Orange.widgets.classify",
        "Orange.widgets.regression",
        "Orange.widgets.evaluate",
        "Orange.widgets.unsupervised",
    ]
    for pkg in pkgs:
        discovery.process_category_package(pkg, distribution=dist)


WIDGET_HELP_PATH = (
    ("{DEVELOP_ROOT}/doc/build/htmlhelp/index.html", None),
#     os.path.join(sysconfig.get_path("data"),
#                  "share", "doc", "Orange-{}".format(Orange.__version__)),
    ("http://docs.orange.biolab.si/3/visual-programming/", "")
)
