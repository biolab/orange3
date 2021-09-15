.. py:currentmodule:: Orange.data.io

################################
Loading and saving data (``io``)
################################

:obj:`Orange.data.Table` supports loading from several file formats:

* Comma-separated values (\*.csv) file,
* Tab-separated values (\*.tab, \*.tsv) file,
* Excel spreadsheet (\*.xls, \*.xlsx),
* Python pickle.

In addition, the text-based files (CSV, TSV) can be compressed with gzip,
bzip2 or xz (e.g. \*.csv.gz).


Header Format
=============

The data in CSV, TSV, and Excel files can be described in an extended
three-line header format, or a condensed single-line header format.


Three-line header format
------------------------

A three-line header consists of:

1. **Feature names** on the first line. Feature names can include any combination
   of characters.

2. **Feature types** on the second line. The type is determined automatically,
   or, if set, can be any of the following:

   * ``discrete`` (or ``d``) — imported as :obj:`Orange.data.DiscreteVariable`,
   * a space-separated **list of discrete values**, like "``male female``",
     which will result in :obj:`Orange.data.DiscreteVariable` with those values
     and in that order. If the individual values contain a space character, it
     needs to be escaped (prefixed) with, as common, a backslash ('\\') character.
   * ``continuous`` (or ``c``) — imported as :obj:`Orange.data.ContinuousVariable`,
   * ``string`` (or ``s``, or ``text``) — imported as :obj:`Orange.data.StringVariable`,
   * ``time`` (or ``t``) — imported as :obj:`Orange.data.TimeVariable`, if the
     values parse as `ISO 8601 <https://en.wikipedia.org/wiki/ISO_8601>`_ date/time formats,

3. **Flags** (optional) on the third header line. Feature's flag can be empty,
   or it can contain, space-separated, a consistent combination of:

   * ``class`` (or ``c``) — feature will be imported as a class variable.
     Most algorithms expect a single class variable.
   * ``meta`` (or ``m``) — feature will be imported as a meta-attribute, just
     describing the data instance but not actually used for learning,
   * ``weight`` (or ``w``) — the feature marks the weight of examples (in
     algorithms that support weighted examples),
   * ``ignore`` (or ``i``) — feature will not be imported,
   * ``<key>=<value>`` are custom attributes recognized in specific contexts, for instance ``color``, which defines the color palette when the variable is visualized, or ``type=image`` which signals that the variable contains a path to an image.

Example of iris dataset in Orange's three-line format
(:download:`iris.tab <../../../../Orange/datasets/iris.tab>`).

.. literalinclude:: ../../../../Orange/datasets/iris.tab
   :lines: 1-7


Single-line header format
-------------------------

Single-line header consists of feature names prefixed by an optional "``<flags>#``"
string, i.e. flags followed by a hash ('#') sign. The flags can be a consistent
combination of:

* ``c`` for class feature (also known as a target variable or dependent variable),
* ``i`` for feature to be ignored,
* ``m`` for meta attributes (not used in learning),
* ``C`` for features that are continuous (numeric),
* ``D`` for features that are discrete (categorical),
* ``T`` for features that represent date and/or time in one of the ISO 8601
  formats,
* ``S`` for string features.

If some (all) names or flags are omitted, the names, types, and flags are
discerned automatically, and correctly (most of the time).
