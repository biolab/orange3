"""Common caching methods, using `lru_cahce` sometimes has its downsides."""
from functools import wraps, lru_cache
import weakref


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


def memoize_method(*lru_args, **lru_kwargs):
    """Memoize methods without keeping reference to `self`.

    Parameters
    ----------
    lru_args
    lru_kwargs

    Returns
    -------

    See Also
    --------
    https://stackoverflow.com/questions/33672412/python-functools-lru-cache-with-class-methods-release-object

    """
    def _decorator(func):

        @wraps(func)
        def _wrapped_func(self, *args, **kwargs):
            self_weak = weakref.ref(self)
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.

            @wraps(func)
            @lru_cache(*lru_args, **lru_kwargs)
            def _cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)

            setattr(self, func.__name__, _cached_method)
            return _cached_method(*args, **kwargs)

        return _wrapped_func

    return _decorator
