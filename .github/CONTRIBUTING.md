Contributing
============

Thanks for taking the time to contribute to Orange!

Please submit contributions in accordance with the flow explained in the
[GitHub Guides].

[GitHub Guides]: https://guides.github.com/


Reporting bugs
--------------
Please report bugs according to established [bug reporting guidelines].
At least, include a method to reproduce the bug (if consistently
reproducible) and a screenshot (if applicable).

[bug reporting guidelines]: https://www.google.com/search?q=reporting+bugs


Coding style
------------

Roughly conform to [PEP-8] style guide for Python code. Whenever PEP-8 is
undefined, adhere to [Google Python Style Guide].

In addition, we add the following guidelines:

* Only ever `import *` to make objects available in another namespace,
  preferably in *\_\_init\_\_.py*. Everywhere else use explicit object
  imports.
* Use [Napoleon]-comaptible (e.g. NumPy style) docstrings, preferably with
  [tests].
* When instantiating Qt widgets, pass static property values as
  [keyword args to the constructor] instead of calling separate property
  setters later. For example, do:

      view = QListView(alternatingRowColors=True,
                       selectionMode=QAbstractItemView.ExtendedSelection)

  instead of:

      view = QListView()
      view.setAlternatingRowColors(True)
      view.setSelectionMode(QAbstractItemView.ExtendedSelection)

* Each Orange widget module, or better still, each Python module (within
  reason) should have a `__name__ == '__main__'`-fenced code block that
  shows/tests the gist of that module in a user-friendly way.

[PEP-8]: https://www.python.org/dev/peps/pep-0008/
[Google Python Style Guide]: https://google.github.io/styleguide/pyguide.html
[Napoleon]: http://www.sphinx-doc.org/en/stable/ext/napoleon.html
[keyword args to the constructor]: http://pyqt.sourceforge.net/Docs/PyQt5/qt_properties.html


Tests
-----
[tests]: #tests
Ensure the tests pass by running:

    python setup.py test

If you contribute new code, make [unit tests] for it in _Orange/tests_.

Prefer [doctests] for public APIs. Note, we unit-test doctests with
`NORMALIZE_WHITESPACE` and `ELLIPSIS` options enabled, so you can use them
implicitly.

[unit tests]: https://en.wikipedia.org/wiki/Unit_testing
[doctests]: https://en.wikipedia.org/wiki/Doctest


Commit messages
---------------
Make a separate commit for each logical change you introduce. We prefer
short commit messages with descriptive titles. For a general format see
[Commit Guidelines]. E.g.:

> io: Fix reader for XYZ file format
>
> The reader didn't work correctly in such-and-such case.

The commit title (first line) should concisely explain _WHAT_ is the change.
If the reasons for the change aren't reasonably obvious, also explain the
_WHY_ and _HOW_ in the commit body.

The commit title should start with a tag which concisely conveys what
Python package, module, or class the introduced change pertains to.

**ProTip**: Examine project's [commit history] to see examples of commit
messages most probably acceptable to that project.

[Commit Guidelines]: http://git-scm.com/book/ch5-2.html#Commit-Guidelines
[commit history]: https://github.com/biolab/orange3/commits/master


Pull requests
-------------
Implement new features in separate topic branches:

    git checkout master
    git checkout -b my-new-feature   # spin a branch off of current branch

When you are asked to make changes to your pull request, and you add the
commits that implement those changes, squash commits that fit together.

E.g., if your pull request looks like this:

    d43ef09   Some feature I made
    b803d26   reverts part of previous commit
    77d5ad3   Some other bugfix
    9e30343   Another new feature
    1d5b3bc   fix typo (in previous commit)

interactively rebase the commits onto the master branch:

    git rebase --interactive master

and mark `fixup` or `squash` the commits that are just minor patches on
previous commits (interactive rebase also allows you to reword and reorder
commits). The resulting example pull request should look clean:

    b432f18   some_module: Some feature I made
    85d5a0a   other.module: Some other bugfix
    439e303   OWSomeWidget: Another new feature

Read more [about squashing commits].

[about squashing commits]: https://www.google.com/search?q=git+squash+commits
