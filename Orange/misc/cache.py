"""Common caching methods, using `lru_cache` sometimes has its downsides."""
from functools import wraps, lru_cache
from typing import MutableMapping
import weakref


def single_cache(func):
    """Cache with size 1."""
    last_args = ()
    last_kwargs = set()
    last_result = None

    @wraps(func)
    def _cached(*args, **kwargs):
        nonlocal last_args, last_kwargs, last_result
        if len(last_args) != len(args) or \
                not all(x is y for x, y in zip(args, last_args)) or \
                last_kwargs != set(kwargs) or \
                any(last_kwargs[k] != kwargs[k] for k in last_kwargs):
            last_result = func(*args, **kwargs)
            last_args, last_kwargs = args, kwargs
        return last_result

    return _cached


def memoize_method(*lru_args, **lru_kwargs):
    """Memoize methods without keeping reference to `self`.

    Using ordinary lru_cache on methods keeps a reference to the object in the cache,
    creating a cycle that keeps the object from getting garbage collected.

    Parameters
    ----------
    lru_args
    lru_kwargs

    See Also
    --------
    https://stackoverflow.com/questions/33672412/python-functools-lru-cache-with-class-methods-release-object

    """
    def _decorator(func):

        @lru_cache(*lru_args, **lru_kwargs)
        def _cached_method(self_weak, *args, **kwargs):
            return func(self_weak(), *args, **kwargs)

        @wraps(func)
        def _wrapped_func(self, *args, **kwargs):
            return _cached_method(weakref.ref(self), *args, **kwargs)

        _wrapped_func.cache_clear = _cached_method.cache_clear
        _wrapped_func.cache_info = _cached_method.cache_info
        return _wrapped_func

    return _decorator


class IDWeakrefCache:
    """
    Cache that caches keys according to their id() for speed. It also stores
    weak references to the keys to ensure that the same keys are being accessed.
    """

    def __init__(self, cache: MutableMapping):
        self._cache = cache

    def __setitem__(self, keys, value):
        self._cache[tuple(map(id, keys))] = \
            value, [weakref.ref(k) for k in keys]

    def __getitem__(self, keys):
        key = tuple(map(id, keys))
        if key not in self._cache:
            raise KeyError()
        shared, weakrefs = self._cache[key]
        for r in weakrefs:
            if r() is None:
                del self._cache[key]
                raise KeyError()
        return shared

    def clear(self):
        self._cache.clear()
