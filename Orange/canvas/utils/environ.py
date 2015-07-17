import warnings

from Orange.misc import environ

warnings.warn("'{}' is deprecated please do not import it"
              .format(__name__),
              DeprecationWarning, stacklevel=2)

buffer_dir = environ.cache_dir()
widget_settings_dir = environ.widget_settings_dir()
