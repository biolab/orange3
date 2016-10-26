def check_results_adequacy(results, error_group):
    error_group.add_message("invalid_results")
    error_group.invalid_results.clear()
    if results is None:
        return None
    if results.data is None:
        error_group.invalid_results(
            "Results do not include information on test data")
    elif not results.data.domain.has_discrete_class:
        error_group.invalid_results(
            "Discrete outcome variable is required")
    else:
        return results
