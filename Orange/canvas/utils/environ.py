import warnings

from Orange.misc import environ

warnings.warn(
    f"'{__name__}' is deprecated and will be removed on the future. "
    "Use 'Orange.misc.environ' instead",
    FutureWarning, stacklevel=2
)
buffer_dir = environ.cache_dir()
widget_settings_dir = environ.widget_settings_dir()
