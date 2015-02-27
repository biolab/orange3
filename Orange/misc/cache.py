def single_cache(f):
    last_args = ()
    last_kwargs = set()
    last_result = None

    def cached(*args, **kwargs):
        nonlocal last_args, last_kwargs, last_result
        if len(last_args) != len(args) or \
                not all(x is y for x, y in zip(args, last_args)) or \
                last_kwargs != set(kwargs) or \
                any(last_kwargs[k] != kwargs[k] for k in last_kwargs):
            last_result = f(*args, **kwargs)
            last_args, last_kwargs = args, kwargs
        return last_result

    return cached
