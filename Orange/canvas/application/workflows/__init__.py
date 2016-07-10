"""
Examples of Orange workflows

"""
import os
import logging
import types
from itertools import chain

import pkg_resources

log = logging.getLogger(__name__)


def list_schemes(package):
    """Return a list of example workflows.
    """
    resources = pkg_resources.resource_listdir(package.__name__, ".")
    resources = list(filter(is_ows, resources))
    return sorted(resources)


def is_ows(filename):
    return filename.endswith(".ows")


def default_entry_point():
    dist = pkg_resources.get_distribution("Orange3")
    ep = pkg_resources.EntryPoint("Orange Canvas", __name__, dist=dist)
    return ep


def workflow_entry_points():
    """Return an iterator over all example workflows.
    """
    default = default_entry_point()
    return chain([default],
                 pkg_resources.iter_entry_points("orange.widgets.tutorials"),
                 pkg_resources.iter_entry_points("orange.widgets.workflows"))


def example_workflows():
    """Return all known example workflows.
    """
    all_workflows = []
    for ep in workflow_entry_points():
        workflows = None
        try:
            workflows = ep.load()
        except pkg_resources.DistributionNotFound as ex:
            log.warning("Could not load workflows from %r (%r)",
                        ep.dist, ex)
            continue
        except ImportError:
            log.error("Could not load workflows from %r",
                      ep.dist, exc_info=True)
            continue
        except Exception:
            log.error("Could not load workflows from %r",
                      ep.dist, exc_info=True)
            continue

        if isinstance(workflows, types.ModuleType):
            package = workflows
            workflows = list_schemes(workflows)
            workflows = [ExampleWorkflow(wf, package, ep.dist)
                         for wf in workflows]
        elif isinstance(workflows, (types.FunctionType, types.MethodType)):
            try:
                workflows = example_workflows()
            except Exception as ex:
                log.error("A callable entry point (%r) raised an "
                          "unexpected error.",
                          ex, exc_info=True)
                continue
            workflows = [ExampleWorkflow(wf, package=None, distribution=ep.dist)
                         for wf in workflows]

        all_workflows.extend(workflows)

    return all_workflows


class ExampleWorkflow(object):
    def __init__(self, resource, package=None, distribution=None):
        self.resource = resource
        self.package = package
        self.distribution = distribution

    def abspath(self):
        """Return absolute filename for the scheme if possible else
        raise an ValueError.

        """
        if self.package is not None:
            return pkg_resources.resource_filename(self.package.__name__,
                                                   self.resource)
        elif isinstance(self.resource, str):
            if os.path.isabs(self.resource):
                return self.resource

        raise ValueError("cannot resolve resource to an absolute name")

    def stream(self):
        """Return the workflow file as an open stream.
        """
        if self.package is not None:
            return pkg_resources.resource_stream(self.package.__name__,
                                                 self.resource)
        elif isinstance(self.resource, str):
            if os.path.isabs(self.resource) and os.path.exists(self.resource):
                return open(self.resource, "rb")

        raise ValueError
