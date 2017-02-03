import numpy


def check_results_adequacy(results, error_group, check_nan=True):
    error_group.add_message("invalid_results")
    error_group.invalid_results.clear()

    def anynan(a):
        return numpy.any(numpy.isnan(a))

    if results is None:
        return None
    if results.data is None:
        error_group.invalid_results(
            "Results do not include information on test data")
    elif not results.data.domain.has_discrete_class:
        error_group.invalid_results(
            "Discrete outcome variable is required")
    elif check_nan and (anynan(results.actual) or
                        anynan(results.predicted) or
                        (results.probabilities is not None and
                         anynan(results.probabilities))):
        error_group.invalid_results(
            "Results contains invalid values")
    else:
        return results
