import warnings
from AnyQt.QtCore import QStandardPaths as _paths

warnings.warn(
    f"{__name__} module is deprecated.", DeprecationWarning, stacklevel=2
)
_location = _paths.writableLocation

Desktop = _location(_paths.DesktopLocation)
Documents = _location(_paths.DocumentsLocation)
Music = _location(_paths.MusicLocation)
Movies = _location(_paths.MoviesLocation)
Pictures = _location(_paths.PicturesLocation)
Home = _location(_paths.HomeLocation)
Cache = _location(_paths.CacheLocation)
AppData = _location(_paths.AppLocalDataLocation)
