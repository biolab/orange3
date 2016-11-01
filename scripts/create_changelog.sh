#!/bin/bash

VERSION=$(grep '^VERSION' setup.py | sed -E "s/.*'([^']*)'/\1/")
RELEASE_DATE=$(date "+%Y-%m-%d")

echo "[$VERSION] - $RELEASE_DATE"
echo "--------------------"
echo '##### Enhancements'
git log stable..master --first-parent --format='%s %b' |
sed -E 's/.*#([0-9]+).*\[ENH\] *(.*)/\* \2 ([#\1](\.\.\/\.\.\/pull\/\1))/' |
    grep -E '^\*'
echo
echo "##### Bugfixes"
git log stable..master --first-parent --format='%s %b' |
    sed -E 's/.*#([0-9]+).*\[FIX\] *(.*)/\* \2 ([#\1](\.\.\/\.\.\/pull\/\1))/' |
    grep -E '^\*'
