#!/bin/bash

git diff-index --quiet --cached HEAD || git diff-files --quiet || { echo "Please, commit your changes or stash them before updating canvas."; echo "Aborting"; exit 1; }

if [ ! -f tip.tar.gz ]; then
  wget https://bitbucket.org/biolab/orange/get/tip.tar.gz
fi
rm -rf biolab-orange-*
tar -xzf tip.tar.gz
mv biolab-orange-*/Orange/OrangeCanvas canvas
rm canvas/orng*
mv canvas/main.py canvas/__main__.py
2to3 -n -W --no-diffs canvas
echo biolab-orange-* | awk -F"-" '{print $3}' > canvas/source-revision.txt
rev=`cat canvas/source-revision.txt`
rm -rf ../Orange/canvas
mv canvas ../Orange/
rm -rf biolab-orange-*
git checkout -b merge-canvas-$rev || exit 1
git add --all ../Orange/canvas
git commit -am "Update canvas to $rev"

echo "Please apply the patches using:"
echo
echo "  git am patches/*.patch"
echo
echo "Fix any conflicts that arise."
echo
echo "When you are done, update the patches using:"
echo
echo "  git format-patch -o patches -7"
echo "  git add --all patches"
echo "  git commit"
echo
echo "To merge updated canvas to master, run:"
echo
echo "  git checkout master"
echo "  git merge --squash merge-canvas-$rev"
echo "  git commit"
