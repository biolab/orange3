This directory contains messsage files and configuration files for translating
Orange to other languages. Currently, Orange is translated to Slovenian.

Orange is translated using `trubar`, which is pip-installable.

### Updating translations

If CI tests report that the translation is out of date, you can update it if you wish.

From directory orange3, run `trubar collect -s Orange i18n/si/msgs.jaml` and see the changes in the message file, `i18n/si/msgs.jaml` e.g. using `git diff`.

Message files are in [a simplified version of YAML](http://janezd.github.io/trubar/message-files/). Obsolete messages will be removed and new messages will be added. You will need to translate the latter or mark them as not needing translation. To do so, change `null`'s to:

- a translation,
- `true` if translation is OK for Slovenian, but the string may need to be changed for some other languages,
- `false` if the string must not be translated, most often because it is a string literal.

When in doubt, do nothing and someone else will look into it some day.

### Translating Orange

#### Preparation (only once)

1. Copy Orange to another directory, say ~/dev/si/orange3, or clone it from GitHub.
2. Create a virtual environment that uses Orange from that directory.

#### Translating

1. Copy the newest Orange sources to your new directory, e.g. ~/dev/si/orange3.
2. `cd` to `i18n` and run `./trans.sh <language> <directory>`, e.g.  `./trans.sh si ~/dev/si/orange3`.
3. Activate the appropriate virtual environment and run Orange.
