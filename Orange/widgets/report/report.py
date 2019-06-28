from collections import OrderedDict

from orangewidget.report.report import *
from orangewidget.report import report as __report

from Orange.data.sql.table import SqlTable

__all__ = __report.__all__ + [
    "DataReport", "describe_data", "describe_data_brief",
    "describe_domain", "describe_domain_brief",
]

del __report


class DataReport(Report):
    """
    A report subclass that adds data related methods to the Report.
    """

    def report_data(self, name, data=None):
        """
        Add description of data table to the report.

        See :obj:`describe_data` for details.

        The first argument, `name` can be omitted.

        :param name: report section name (can be omitted)
        :type name: str or tuple or OrderedDict
        :param data: data whose description is added to the report
        :type data: Orange.data.Table
        """

        name, data = self._fix_args(name, data)
        self.report_items(name, describe_data(data))

    def report_domain(self, name, domain=None):
        """
        Add description of domain to the report.

        See :obj:`describe_domain` for details.

        The first argument, `name` can be omitted.

        :param name: report section name (can be omitted)
        :type name: str or tuple or OrderedDict
        :param domain: domain whose description is added to the report
        :type domain: Orange.data.Domain
        """
        name, domain = self._fix_args(name, domain)
        self.report_items(name, describe_domain(domain))

    def report_data_brief(self, name, data=None):
        """
        Add description of data table to the report.

        See :obj:`describe_data_brief` for details.

        The first argument, `name` can be omitted.

        :param name: report section name (can be omitted)
        :type name: str or tuple or OrderedDict
        :param data: data whose description is added to the report
        :type data: Orange.data.Table
        """
        name, data = self._fix_args(name, data)
        self.report_items(name, describe_data_brief(data))


# For backwards compatibility Report shadows the one from the base
Report = DataReport


def describe_domain(domain):
    """
    Return an :obj:`OrderedDict` describing a domain

    Description contains keys "Features", "Meta attributes" and "Targets"
    with the corresponding clipped lists of names. If the domain contains no
    meta attributes or targets, the value is `False`, which prevents it from
    being rendered by :obj:`~Orange.widgets.report.render_items`.

    :param domain: domain
    :type domain: Orange.data.Domain
    :rtype: OrderedDict
    """

    def clip_attrs(items, s):
        return clipped_list([a.name for a in items], 1000,
                            total_min=10, total=" (total: {{}} {})".format(s))

    return OrderedDict(
        [("Features", clip_attrs(domain.attributes, "features")),
         ("Meta attributes", bool(domain.metas) and
          clip_attrs(domain.metas, "meta attributes")),
         ("Target", bool(domain.class_vars) and
          clip_attrs(domain.class_vars, "targets variables"))])


def describe_data(data):
    """
    Return an :obj:`OrderedDict` describing the data

    Description contains keys "Data instances" (with the number of instances)
    and "Features", "Meta attributes" and "Targets" with the corresponding
    clipped lists of names. If the domain contains no meta attributes or
    targets, the value is `False`, which prevents it from being rendered.

    :param data: data
    :type data: Orange.data.Table
    :rtype: OrderedDict
    """
    items = OrderedDict()
    if data is None:
        return items
    if isinstance(data, SqlTable):
        items["Data instances"] = data.approx_len()
    else:
        items["Data instances"] = len(data)
    items.update(describe_domain(data.domain))
    return items


def describe_domain_brief(domain):
    """
    Return an :obj:`OrderedDict` with the number of features, metas and classes

    Description contains "Features" and "Meta attributes" with the number of
    featuers, and "Targets" that contains either a name, if there is a single
    target, or the number of targets if there are multiple.

    :param domain: data
    :type domain: Orange.data.Domain
    :rtype: OrderedDict
    """
    items = OrderedDict()
    if domain is None:
        return items
    items["Features"] = len(domain.attributes) or "None"
    items["Meta attributes"] = len(domain.metas) or "None"
    if domain.has_discrete_class:
        items["Target"] = "Class '{}'".format(domain.class_var.name)
    elif domain.has_continuous_class:
        items["Target"] = "Numeric variable '{}'". \
            format(domain.class_var.name)
    elif domain.class_vars:
        items["Targets"] = len(domain.class_vars)
    else:
        items["Targets"] = False
    return items


def describe_data_brief(data):
    """
    Return an :obj:`OrderedDict` with a brief description of data.

    Description contains keys "Data instances" with the number of instances,
    "Features" and "Meta attributes" with the corresponding numbers, and
    "Targets", which contains a name, if there is a single target, or the
    number of targets if there are multiple.

    :param data: data
    :type data: Orange.data.Table
    :rtype: OrderedDict
    """
    items = OrderedDict()
    if data is None:
        return items
    if isinstance(data, SqlTable):
        items["Data instances"] = data.approx_len()
    else:
        items["Data instances"] = len(data)
    items.update(describe_domain_brief(data.domain))
    return items
