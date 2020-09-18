import abc
import os
from typing import Optional, Mapping, NamedTuple, Type, Dict


class _DataType:
    def __eq__(self, other):
        """Equal if `other` has the same type and all elements compare equal."""
        if type(self) is not type(other):
            return False
        return super().__eq__(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((type(self), super().__hash__()))


class PathItem(abc.ABC):
    """
    Abstract data type representing an optionally variable prefixed path.
    Has only two type members: `AbsPath` and `VarPath`
    """
    def exists(self, env: Mapping[str, str]) -> bool:
        """Does path exists when evaluated in `env`."""
        return self.resolve(env) is not None

    @abc.abstractmethod
    def resolve(self, env: Mapping[str, str]) -> Optional[str]:
        """Resolve (evaluate) path to an absolute path. Return None if path
        does not resolve or does not exist.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def as_dict(self) -> Dict[str, str]:
        """Encode item as dict"""
        raise NotImplementedError

    @staticmethod
    def from_dict(data: Mapping[str, str]) -> 'PathItem':
        """Inverse of `as_dict`"""
        try:
            type_ = data["type"]
            if type_ == "AbsPath":
                return AbsPath(data["path"])
            elif type_ == "VarPath":
                return VarPath(data["name"], data["relpath"])
            else:
                raise ValueError(f"{type_}: unknown type")
        except KeyError as err:
            raise ValueError() from err

    # Forward declarations for type members
    AbsPath: Type['AbsPath']
    VarPath: Type['VarPath']


class AbsPath(_DataType, NamedTuple("AbsPath", [("path", str)]), PathItem):
    """
    An absolute path (no var env substitution).
    """
    def __new__(cls, path):
        path = os.path.abspath(os.path.normpath(path))
        if os.name == "nt":
            # Always store paths using a cross platform compatible sep
            path = path.replace(os.path.sep, "/")
        return super().__new__(cls, path)

    def resolve(self, env: Mapping[str, str]) -> Optional[str]:
        return self.path if os.path.exists(self.path) else None

    def as_dict(self) -> Dict[str, str]:
        return {"type": "AbsPath", "path": self.path}


class VarPath(_DataType, NamedTuple("VarPath", [("name", str), ("relpath", str)]),
              PathItem):
    """
    A variable prefix path. `name` is the prefix name and `relpath` the path
    relative to prefix.
    """
    def __new__(cls, name, relpath):
        relpath = os.path.normpath(relpath)
        if relpath.startswith(os.path.pardir):
            raise ValueError("invalid relpath '{}'".format(relpath))
        if os.name == "nt":
            relpath = relpath.replace(os.path.sep, "/")
        return super().__new__(cls, name, relpath)

    def resolve(self, env: Mapping[str, str]) -> Optional[str]:
        prefix = env.get(self.name, None)
        if prefix is not None:
            path = os.path.join(prefix, self.relpath)
            return path if os.path.exists(path) else None
        return None

    def as_dict(self) -> Dict[str, str]:
        return {"type": "VarPath", "name": self.name, "relpath": self.relpath}


PathItem.AbsPath = AbsPath
PathItem.VarPath = VarPath


def infer_prefix(path, env) -> Optional[VarPath]:
    """
    Create a PrefixRelative item inferring a suitable prefix name and relpath.

    Parameters
    ----------
    path : str
        File system path.
    env : List[Tuple[str, str]]
        A sequence of (NAME, basepath) pairs. The sequence is searched
        for a item such that basepath/relpath == path and the
        VarPath(NAME, relpath) is returned.
        (note: the first matching prefixed path is chosen).

    Returns
    -------
    varpath : VarPath
    """
    abspath = os.path.abspath(path)
    for sname, basepath in env:
        if isprefixed(basepath, abspath):
            relpath = os.path.relpath(abspath, basepath)
            return VarPath(sname, relpath)
    return None


def isprefixed(prefix, path):
    """
    Is `path` contained within the directory `prefix`.

    >>> isprefixed("/usr/local/", "/usr/local/shared")
    True
    """
    normalize = lambda path: os.path.normcase(os.path.normpath(path))
    prefix, path = normalize(prefix), normalize(path)
    if not prefix.endswith(os.path.sep):
        prefix = prefix + os.path.sep
    return os.path.commonprefix([prefix, path]) == prefix


def samepath(p1, p2):
    # type: (str, str) -> bool
    """
    Return True if the paths `p1` and `p2` match after case and path
    normalization.
    """
    return pathnormalize(p1) == pathnormalize(p2)


def pathnormalize(p):
    """
    Normalize a path (apply both path and case normalization.
    """
    return os.path.normcase(os.path.normpath(p))


def prettyfypath(path):
    """
    Return the path with the $HOME prefix shortened to '~/' if applicable.

    Example
    -------
    >>> prettyfypath("/home/user/file.dat")
    '~/file.dat'
    """
    home = os.path.expanduser("~/")
    home_n = pathnormalize(home)
    path_n = pathnormalize(path)
    if path_n.startswith(home_n):
        path = os.path.join("~", os.path.relpath(path, home))
    return path
