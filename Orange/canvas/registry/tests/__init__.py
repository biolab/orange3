"""
"""


def small_testing_registry():
    """Return a small registry with a few widgets for testing.
    """
    from ..description import WidgetDescription, CategoryDescription
    from .. import WidgetRegistry

    registry = WidgetRegistry()

    data_desc = CategoryDescription.from_package(
        "Orange.OrangeWidgets.Data"
    )

    file_desc = WidgetDescription.from_module(
        "Orange.OrangeWidgets.Data.OWFile"
    )
    discretize_desc = WidgetDescription.from_module(
        "Orange.OrangeWidgets.Data.OWDiscretize"
    )

    classify_desc = CategoryDescription.from_package(
        "Orange.OrangeWidgets.Classify"
    )

    bayes_desc = WidgetDescription.from_module(
        "Orange.OrangeWidgets.Classify.OWNaiveBayes"
    )

    registry.register_category(data_desc)
    registry.register_category(classify_desc)
    registry.register_widget(file_desc)
    registry.register_widget(discretize_desc)
    registry.register_widget(bayes_desc)
    return registry
