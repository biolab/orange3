"""
========
Document
========

The :mod:`document` package contains classes for visual interactive editing
of a :class:`Scheme` instance.

The :class:`.SchemeEditWidget` is the main widget used for editing. It
uses classes defined in :mod:`canvas` to display the scheme. It also
supports undo/redo functionality.

"""

__all__ = ["quickmenu", "schemeedit"]

from .schemeedit import SchemeEditWidget
