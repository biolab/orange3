from AnyQt.QtCore import QT_VERSION

if QT_VERSION > 0x50000:
    from AnyQt.QtCore import QStandardPaths as _paths
    _location = _paths.writableLocation
else:
    from AnyQt.QtGui import QDesktopServices as _paths
    _location = _paths.storageLocation

Desktop = _location(_paths.DesktopLocation)
Documents = _location(_paths.DocumentsLocation)
Music = _location(_paths.MusicLocation)
Movies = _location(_paths.MoviesLocation)
Pictures = _location(_paths.PicturesLocation)
Home = _location(_paths.HomeLocation)
Cache = _location(_paths.CacheLocation)
AppData = _location(_paths.DataLocation)
