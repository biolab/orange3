#
# sitecustomize added by orange3 installer.
#
# (Ana)conda python distribution expects it is 'activated', it does not really
# support unactivated invocations (although a similar facility to this was
# included in anaconda python 3.6 and earlier).

import sys
import os


def normalize(path):
    return os.path.normcase(os.path.normpath(path))


extra_paths = [
    r"Library\bin",
]

paths = os.environ.get("PATH", "").split(os.path.pathsep)
paths = [normalize(path) for path in paths]

for path in extra_paths:
    path = os.path.join(sys.prefix, path)

    if os.path.isdir(path) and normalize(path) not in paths:
        os.environ["PATH"] = os.pathsep.join((path, os.environ.get("PATH", "")))
