"""

"""
import pkg_resources


# Entry point for main Orange categories/widgets discovery
def widget_discovery(discovery):
    from . import data
    dist = pkg_resources.get_distribution("Orange")
    for pkg in [data]:
        discovery.process_category_package(pkg, distribution=dist)
